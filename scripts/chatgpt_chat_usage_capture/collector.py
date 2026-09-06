"""Fixture-backed Chat history collection, backfill, and incremental refresh."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional
from uuid import uuid4

from .adapter import ChatGPTHistoryAdapter, FixtureTransport
from .config import AccountConfig, CollectorConfig
from .ledger import Ledger
from .models import ConversationSummary, MessageRecord
from .privacy import SURFACE_CHAT, evidence_identity, observation_projection
from .reconstruct import reconstruct_attempts
from .timeutil import ensure_utc, isoformat_utc, parse_datetime

Clock = Callable[[], datetime]


@dataclass
class RunResult:
    run_id: str
    mode: str
    result: str
    coverage: str
    new_attempts: int
    updated_attempts: int
    deduplicated_attempts: int
    conversations_seen: int
    pages_fetched: int
    missed_intervals: int
    warnings: list[str]
    requests: list[dict[str, Any]]


class Collector:
    def __init__(
        self,
        config: CollectorConfig,
        ledger: Ledger,
        *,
        adapter: ChatGPTHistoryAdapter | None = None,
        fixture_root: str | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.config = config
        self.ledger = ledger
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.fixture_root = fixture_root
        self.adapter = adapter

    def _adapter_for(self, account: AccountConfig) -> ChatGPTHistoryAdapter:
        if self.adapter is not None:
            return self.adapter
        if self.fixture_root is None:
            raise RuntimeError(
                "live Playwright collection is not enabled in this vertical slice; "
                "pass --fixture-root for fixture-backed discovery"
            )
        transport = FixtureTransport(self.fixture_root)
        return ChatGPTHistoryAdapter(
            transport,
            expected_identity={
                "provider_user_id": account.expected_provider_user_id,
                "workspace_id": account.expected_workspace_id,
            },
        )

    def inspect_capabilities(self, account_id: Optional[str] = None) -> dict[str, Any]:
        account = self.config.account(account_id)
        adapter = self._adapter_for(account)
        identity = adapter.inspect_session()
        capabilities = adapter.capabilities
        return {
            "account_id": account.id,
            "surface": SURFACE_CHAT,
            "identity": identity,
            "capabilities": {
                "adapter_version": capabilities.adapter_version,
                "index_scopes": list(capabilities.index_scopes),
                "archive_behavior": capabilities.archive_behavior,
                "project_coverage": capabilities.project_coverage,
                "modern_detail": capabilities.modern_detail,
                "pagination": capabilities.pagination,
                "legacy_support": capabilities.legacy_support,
                "branch_visibility": capabilities.branch_visibility,
                "model_metadata": capabilities.model_metadata,
                "quota_metadata": capabilities.quota_metadata,
            },
        }

    def collect(
        self,
        *,
        account_id: Optional[str] = None,
        mode: str = "refresh",
        since: Optional[datetime] = None,
        missed_intervals: int = 0,
        scheduled_for: Optional[datetime] = None,
    ) -> RunResult:
        account = self.config.account(account_id)
        adapter = self._adapter_for(account)
        now = ensure_utc(self.clock())
        run_id = str(uuid4())
        warnings: list[str] = []
        self.ledger.upsert_account(
            {
                "collector_account_id": account.id,
                "provider_user_id": account.expected_provider_user_id,
                "workspace_id": account.expected_workspace_id,
                "quota_owner_id": account.quota_owner_id,
                "surface": SURFACE_CHAT,
                "auth_state": "ready",
                "plan_policy_id": account.plan_policy_id,
                "enabled": account.enabled,
                "profile_path": str(account.browser.profile_path),
            }
        )
        identity = adapter.inspect_session()
        if identity.get("auth_state") in {"auth_required", "identity_mismatch"}:
            with self.ledger.transaction():
                self.ledger.record_run_start(
                    run_id=run_id,
                    account_id=account.id,
                    mode=mode,
                    started_at=now,
                    scheduled_for=scheduled_for,
                    missed_intervals=missed_intervals,
                )
                self.ledger.finish_run(
                    run_id,
                    ended_at=self.clock(),
                    result=identity["auth_state"],
                    coverage="paused",
                    error_class=identity["auth_state"],
                    details={"identity": identity},
                )
            return RunResult(
                run_id=run_id,
                mode=mode,
                result=identity["auth_state"],
                coverage="paused",
                new_attempts=0,
                updated_attempts=0,
                deduplicated_attempts=0,
                conversations_seen=0,
                pages_fetched=0,
                missed_intervals=missed_intervals,
                warnings=[identity["auth_state"]],
                requests=list(getattr(adapter.transport, "requests", [])),
            )
        backfill_start = since or (now - account.scheduler.initial_backfill_duration)
        with self.ledger.transaction():
            self.ledger.record_run_start(
                run_id=run_id,
                account_id=account.id,
                mode=mode,
                started_at=now,
                scheduled_for=scheduled_for,
                missed_intervals=missed_intervals,
            )
        scopes = ["active"]
        if account.collection.include_archived:
            scopes.append("archived")
        pages_fetched = 0
        conversations: dict[str, ConversationSummary] = {}
        coverage = "validated_page"
        for scope in scopes:
            discovered, page_count, scope_coverage, scope_warnings = self._discover_scope(
                adapter=adapter,
                account=account,
                scope=scope,
                run_id=run_id,
                now=now,
                backfill_start=backfill_start,
            )
            pages_fetched += page_count
            warnings.extend(scope_warnings)
            if scope_coverage != "validated_page":
                coverage = scope_coverage
            for summary in discovered:
                conversations[summary.conversation_id] = summary
        outstanding = self.ledger.pending_conversations(account.id)
        for row in outstanding:
            conversation_id = row["conversation_id"]
            if conversation_id not in conversations:
                conversations[conversation_id] = ConversationSummary(
                    conversation_id=conversation_id,
                    created_at=parse_datetime(row["created_at"]),
                    updated_at=parse_datetime(row["updated_at"]),
                    is_archived=bool(row["is_archived"]),
                    workspace_id=row["workspace_id"],
                    project_id=None,
                    surface=row["surface"] or "unknown",
                    origin=row["origin"],
                    has_versions=None,
                    current_node=None,
                    coverage=row["page_coverage"] or "partial",
                )
        new_attempts = 0
        updated_attempts = 0
        deduplicated_attempts = 0
        for summary in conversations.values():
            if summary.surface not in {SURFACE_CHAT, "unknown"}:
                warnings.append(f"excluded_surface:{summary.conversation_id}:{summary.surface}")
                continue
            if summary.surface == "unknown" and account.collection.require_surface_evidence:
                warnings.append(f"unknown_surface:{summary.conversation_id}")
            fetched, page_count, conv_warnings = self._fetch_conversation(
                adapter=adapter,
                account=account,
                summary=summary,
                run_id=run_id,
                now=now,
            )
            pages_fetched += page_count
            warnings.extend(conv_warnings)
            stats = self._ingest_messages(
                account=account,
                conversation_id=summary.conversation_id,
                messages=fetched,
                run_id=run_id,
                now=now,
            )
            new_attempts += stats["inserted"]
            updated_attempts += stats["updated"]
            deduplicated_attempts += stats["deduplicated"]
        result = "complete" if coverage == "validated_page" and not any(
            warning.startswith("budget") or warning.startswith("repeated_cursor")
            for warning in warnings
        ) else "partial"
        with self.ledger.transaction():
            self.ledger.finish_run(
                run_id,
                ended_at=ensure_utc(self.clock()),
                result=result,
                new_attempts=new_attempts,
                updated_attempts=updated_attempts,
                deduplicated_attempts=deduplicated_attempts,
                coverage=coverage if result == "complete" else "partial",
                details={"warnings": warnings[:32], "conversations": len(conversations)},
            )
            if result == "complete":
                for scope in scopes:
                    self.ledger.upsert_discovery_state(
                        account_id=account.id,
                        scope=scope,
                        last_complete_discovery_started_at=now,
                        last_complete_discovery_finished_at=ensure_utc(self.clock()),
                        continuation=None,
                        watermark_at=now,
                        coverage="complete",
                    )
        return RunResult(
            run_id=run_id,
            mode=mode,
            result=result,
            coverage=coverage if result == "complete" else "partial",
            new_attempts=new_attempts,
            updated_attempts=updated_attempts,
            deduplicated_attempts=deduplicated_attempts,
            conversations_seen=len(conversations),
            pages_fetched=pages_fetched,
            missed_intervals=missed_intervals,
            warnings=warnings,
            requests=list(getattr(adapter.transport, "requests", [])),
        )

    def _discover_scope(
        self,
        *,
        adapter: ChatGPTHistoryAdapter,
        account: AccountConfig,
        scope: str,
        run_id: str,
        now: datetime,
        backfill_start: datetime,
    ) -> tuple[list[ConversationSummary], int, str, list[str]]:
        archived = scope == "archived"
        offset = 0
        limit = 100
        seen_ids: set[str] = set()
        collected: list[ConversationSummary] = []
        warnings: list[str] = []
        pages = 0
        coverage = "validated_page"
        seen_continuations: set[str] = set()
        state = self.ledger.get_discovery_state(account.id, scope) or {}
        last_complete = parse_datetime(state.get("last_complete_discovery_started_at"))
        if last_complete is None:
            cutoff = backfill_start
        else:
            cutoff = last_complete - account.scheduler.overlap_duration
        while True:
            page = adapter.list_conversations(archived=archived, offset=offset, limit=limit)
            pages += 1
            if page.coverage != "validated_page":
                coverage = page.coverage
            warnings.extend(page.warnings)
            continuation_key = str(page.continuation)
            if continuation_key in seen_continuations and page.continuation is not None:
                warnings.append("repeated_cursor")
                coverage = "partial"
                self.ledger.record_gap(
                    gap_id=evidence_identity("gap", account.id, f"{scope}:repeated_cursor"),
                    account_id=account.id,
                    scope=scope,
                    reason="repeated_cursor",
                    now=now,
                )
                break
            if page.continuation is not None:
                seen_continuations.add(continuation_key)
            reached_cutoff = True
            for summary in page.items:
                if summary.conversation_id in seen_ids:
                    continue
                seen_ids.add(summary.conversation_id)
                if summary.updated_at is None:
                    warnings.append(f"unresolved_update_time:{summary.conversation_id}")
                    collected.append(summary)
                    continue
                if summary.updated_at >= cutoff:
                    collected.append(summary)
                    reached_cutoff = False
                else:
                    # Keep scanning only when ordering is validated; fixtures use updated desc.
                    continue
            page_start = len(collected) - len(page.items) if page.items else len(collected)
            with self.ledger.transaction():
                for summary in collected[page_start:]:
                    if summary.conversation_id:
                        self.ledger.upsert_conversation(account.id, summary, run_id)
                        self.ledger.insert_observation(
                            account_id=account.id,
                            source_kind=f"index:{scope}",
                            source_id=summary.conversation_id,
                            revision=isoformat_utc(summary.updated_at) or "unknown",
                            surface=summary.surface,
                            conversation_id=summary.conversation_id,
                            payload=observation_projection(
                                {
                                    "id": summary.conversation_id,
                                    "update_time": isoformat_utc(summary.updated_at),
                                    "create_time": isoformat_utc(summary.created_at),
                                    "is_archived": summary.is_archived,
                                    "workspace_id": summary.workspace_id,
                                    "surface": summary.surface,
                                },
                                source_kind=f"index:{scope}",
                                run_id=run_id,
                                evidence_id=summary.conversation_id,
                            ),
                            observed_at=now,
                            run_id=run_id,
                        )
                self.ledger.upsert_discovery_state(
                    account_id=account.id,
                    scope=scope,
                    continuation=None if page.continuation is None else str(page.continuation),
                    coverage=page.coverage,
                )
            if page.exhausted or page.continuation is None:
                break
            if reached_cutoff and page.coverage == "validated_page" and not page.warnings:
                break
            if pages >= account.collection.max_http_attempts_per_run:
                warnings.append("budget_exhausted")
                coverage = "partial"
                break
            try:
                offset = int(page.continuation)
            except (TypeError, ValueError):
                warnings.append("non_numeric_continuation")
                coverage = "partial"
                break
        return collected, pages, coverage, warnings

    def _fetch_conversation(
        self,
        *,
        adapter: ChatGPTHistoryAdapter,
        account: AccountConfig,
        summary: ConversationSummary,
        run_id: str,
        now: datetime,
    ) -> tuple[list[MessageRecord], int, list[str]]:
        warnings: list[str] = []
        pages = 0
        records: list[MessageRecord] = []
        payload = adapter.fetch_conversation(summary.conversation_id)
        pages += 1
        from .adapter import adapt_message_page, iter_mapping_messages

        # Surface propagation: conversations discovered by the Chat collector
        # are Chat conversations.  Default unknown / None surfaces to
        # SURFACE_CHAT so that _ingest_messages, reconstruct_attempts, and
        # reporting all see the correct surface, avoiding zero-count attempts.
        effective_surface = (
            summary.surface if summary.surface == SURFACE_CHAT else SURFACE_CHAT
        )

        if isinstance(payload.get("mapping"), dict):
            mapping_warnings: list[str] = []
            records.extend(
                iter_mapping_messages(
                    payload["mapping"],
                    conversation_id=summary.conversation_id,
                    warnings=mapping_warnings,
                    conversation_surface=effective_surface,
                )
            )
            warnings.extend(mapping_warnings)
        page = adapt_message_page(
            payload,
            conversation_id=summary.conversation_id,
            conversation_surface=effective_surface,
        )
        records.extend(page.items)
        warnings.extend(page.warnings)
        cursor = page.continuation
        seen_cursors: set[str] = set()
        while cursor:
            if str(cursor) in seen_cursors:
                warnings.append("repeated_cursor")
                break
            seen_cursors.add(str(cursor))
            next_page = adapter.fetch_messages(summary.conversation_id, before=str(cursor))
            pages += 1
            records.extend(next_page.items)
            warnings.extend(next_page.warnings)
            if "repeated_cursor" in next_page.warnings:
                break
            if next_page.exhausted or next_page.continuation is None:
                break
            if pages >= account.collection.max_pages_per_conversation_per_run:
                warnings.append("conversation_page_budget")
                break
            cursor = next_page.continuation
        coverage = "partial" if warnings else "validated_page"
        self.ledger.mark_conversation_fetched(account.id, summary.conversation_id, coverage)
        self.ledger.insert_observation(
            account_id=account.id,
            source_kind="conversation",
            source_id=summary.conversation_id,
            revision=isoformat_utc(summary.updated_at) or run_id,
            surface=summary.surface,
            conversation_id=summary.conversation_id,
            payload=observation_projection(
                {
                    "id": summary.conversation_id,
                    "update_time": isoformat_utc(summary.updated_at),
                    "surface": summary.surface,
                    "message_count": len(records),
                },
                source_kind="conversation",
                run_id=run_id,
                evidence_id=summary.conversation_id,
            ),
            observed_at=now,
            run_id=run_id,
        )
        return records, pages, warnings

    def _ingest_messages(
        self,
        *,
        account: AccountConfig,
        conversation_id: str,
        messages: list[MessageRecord],
        run_id: str,
        now: datetime,
    ) -> dict[str, int]:
        unique: dict[str, MessageRecord] = {}
        for record in messages:
            unique[record.message_id] = record
        with self.ledger.transaction():
            for record in unique.values():
                self.ledger.upsert_message(account.id, record)
                self.ledger.insert_observation(
                    account_id=account.id,
                    source_kind="message",
                    source_id=f"{conversation_id}:{record.message_id}",
                    revision=isoformat_utc(record.created_at) or record.message_id,
                    surface=record.surface,
                    conversation_id=conversation_id,
                    payload=observation_projection(
                        {
                            "id": record.message_id,
                            "conversation_id": conversation_id,
                            "parent": record.parent_id,
                            "author": {"role": record.role},
                            "create_time": isoformat_utc(record.created_at),
                            "status": record.status,
                            "end_turn": record.end_turn,
                            "metadata": record.metadata,
                            "surface": record.surface,
                        },
                        source_kind="message",
                        run_id=run_id,
                        evidence_id=record.message_id,
                    ),
                    observed_at=now,
                    run_id=run_id,
                )
            stored = self.ledger.messages_for(account.id, conversation_id)
            attempts = reconstruct_attempts(
                stored,
                mapping_version=self.config.model_mapping.version,
                mapping_rules=self.config.model_mapping.exact_rules,
                conversation_id=conversation_id,
            )
            inserted = updated = deduplicated = 0
            for attempt in attempts:
                status = self.ledger.upsert_attempt(account.id, attempt)
                if status == "inserted":
                    inserted += 1
                else:
                    deduplicated += 1
                    updated += 1
        return {"inserted": inserted, "updated": updated, "deduplicated": deduplicated}
