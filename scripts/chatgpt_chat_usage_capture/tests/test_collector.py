"""Focused tests for ChatGPT Chat usage collector (D1-752)."""

from __future__ import annotations

import copy
import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.chatgpt_chat_usage_capture.config import CollectorConfig, parse_config
from scripts.chatgpt_chat_usage_capture.ledger import Ledger
from scripts.chatgpt_chat_usage_capture.privacy import (
    SURFACE_CHAT,
    ADAPTER_VERSION,
    assert_no_secrets,
    classify_surface,
    sanitize_metadata,
    sanitize_identity,
    sanitize_mapping,
    observation_projection,
)
from scripts.chatgpt_chat_usage_capture.adapter import (
    AdapterError,
    ChatGPTHistoryAdapter,
    FixtureTransport,
    adapt_message_page,
    message_from_node,
)
from scripts.chatgpt_chat_usage_capture.collector import Collector
from scripts.chatgpt_chat_usage_capture.reporting import build_report
from scripts.chatgpt_chat_usage_capture.timeutil import ensure_utc, parse_iso_duration, isoformat_utc
from scripts.chatgpt_chat_usage_capture.models import ConversationSummary, AttemptRecord


FIXTURE_ROOT = Path(__file__).parent / "fixtures"


class SessionTransport:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        assert method == "GET"
        assert path == "/api/auth/session"
        return self.payload


class IncompleteDetailTransport:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        assert method == "GET"
        query = dict(params or {})
        self.requests.append({"method": method, "path": path, "params": query})
        if path == "/api/auth/session":
            return {
                "user": {"id": "user-abc123"},
                "workspace_id": "ws-xyz",
                "quota_owner_id": "user-abc123",
                "surface": "chat",
            }
        if path == "/backend-api/conversations":
            if query.get("is_archived") == "true":
                return {"items": [], "total": 0, "offset": 0}
            return {
                "items": [
                    {
                        "id": "conv-incomplete",
                        "create_time": "2026-09-05T10:00:00Z",
                        "update_time": "2026-09-05T12:00:00Z",
                        "surface": "chat",
                        "workspace_id": "ws-xyz",
                    }
                ],
                "total": 1,
                "offset": 0,
            }
        if path == "/backend-api/conversations/conv-incomplete":
            return {
                "conversation_id": "conv-incomplete",
                "surface": "chat",
                "messages": [
                    {
                        "id": "msg-incomplete",
                        "author": {"role": "assistant"},
                        "create_time": "2026-09-05T12:00:00Z",
                        "metadata": {
                            "model_slug": "gpt-5.6-astra-pro",
                            "generation_id": "gen-incomplete",
                        },
                        "status": "finished_successfully",
                        "end_turn": True,
                    }
                ],
                "page_info": {"has_previous_page": True},
            }
        raise AssertionError(f"unexpected request: {method} {path} {query}")


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

MINIMAL_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "application": {
        "database_path": ":memory:",
    },
    "accounts": [
        {
            "id": "test-account",
            "enabled": True,
            "surface": "chat",
            "expected_provider_user_id": "user-abc123",
            "expected_workspace_id": "ws-xyz",
            "quota_owner_id": "user-abc123",
            "plan_policy_id": "pro200-chat-2026-09-05",
            "browser": {
                "adapter": "fixture_history",
            },
            "scheduler": {},
            "collection": {},
        }
    ],
    "accounting": {},
    "model_mapping": {
        "version": "test",
        "canonical_families": ["astra_pro", "sol_pro", "other_chat", "unknown"],
        "exact_rules": [],
        "unknown_behavior": "preserve_and_report",
    },
    "quota_policies": [
        {
            "id": "pro200-chat-2026-09-05",
            "status": "documented_seed_requires_account_verification",
            "surface": "chat",
            "buckets": [
                {
                    "id": "pro200-astra-chat",
                    "families": ["astra_pro"],
                    "capacity": 200,
                    "unit": "message",
                    "window": {"type": "unknown"},
                },
                {
                    "id": "pro200-sol-chat",
                    "families": ["sol_pro"],
                    "capacity": 200,
                    "unit": "message",
                    "window": {"type": "unknown"},
                },
            ],
        }
    ],
    "reporting": {"default_lookback": "P7D"},
    "retention": {"sanitized_observation_days": 45, "normalized_attempt_days": 180},
}


def _make_config(overrides: dict[str, Any] | None = None) -> CollectorConfig:
    payload = dict(MINIMAL_CONFIG)
    if overrides:
        payload.update(overrides)
    return parse_config(payload, source_path=Path("/tmp/test-config.yaml"))


# ---------------------------------------------------------------------------
# Privacy
# ---------------------------------------------------------------------------

class TestPrivacy:
    def test_surface_classification_chat_first(self):
        assert classify_surface("chat") == SURFACE_CHAT
        assert classify_surface("chatgpt") == SURFACE_CHAT
        assert classify_surface("codex") == "codex"
        assert classify_surface("work") == "work"
        assert classify_surface("voice") == "voice"
        assert classify_surface("unknown") == "unknown"
        assert classify_surface(None, default=None) == "unknown"

    def test_sanitize_mapping_removes_content(self):
        payload = {"id": "msg-1", "content": "secret text", "title": "My Title", "role": "user"}
        result = sanitize_mapping(payload)
        assert "id" in result
        assert "content" not in result
        assert "title" not in result
        assert "role" in result

    def test_sanitize_author_redacts_name(self):
        payload = {"author": {"role": "user", "name": "John Doe", "email": "j@d.com"}}
        result = sanitize_mapping(payload)
        assert result["author"]["role"] == "user"
        assert result["author"]["name"] == "[redacted]"

    def test_sanitize_identity_preserves_allowlist(self):
        payload = {
            "provider_user_id": "user-123",
            "workspace_id": "ws-456",
            "quota_owner_id": "user-123",
            "email": "test@example.com",
        }
        result = sanitize_identity(payload)
        assert result["provider_user_id"] == "user-123"
        assert result["workspace_id"] == "ws-456"
        assert "accessToken" not in result

    def test_assert_no_secrets_removes_sk_pattern(self):
        # Model slugs like 'gpt-5.6-astra-pro' should not trigger secret assertion
        assert_no_secrets({"model_slug": "gpt-5.6-astra-pro"})
        assert_no_secrets({"model_slug": "gpt-6-sol-pro"})

    def test_assert_no_secrets_blocks_bearer(self):
        with pytest.raises(Exception):
            assert_no_secrets({"auth": "Bearer sk-abc123"})

    def test_observation_projection_adds_provenance(self):
        result = observation_projection(
            {"id": "conv-1", "update_time": "2026-01-01T00:00:00Z", "surface": "chat"},
            source_kind="test",
            run_id="run-1",
            evidence_id="ev-1",
        )
        assert result["surface"] == SURFACE_CHAT
        assert result["provenance"]["adapter_version"] == ADAPTER_VERSION
        assert result["provenance"]["source_kind"] == "test"

    def test_metadata_projection_drops_content_and_unknown_values(self):
        raw_metadata = {
            "requested_model": "gpt-5.6-astra-pro",
            "generation_id": "gen-001",
            "request_id": "req-001",
            "surface": "chat",
            "prompt": "PRIVATE_PROMPT_9f7c",
            "citations": [
                {
                    "title": "PRIVATE_CITATION_TITLE_9f7c",
                    "url": "https://private.example/9f7c",
                }
            ],
            "private_note": "PRIVATE_METADATA_9f7c",
            "finish_details": {"reason": "PRIVATE_FINISH_DETAIL_9f7c"},
            "model_slug": {"content": "PRIVATE_NESTED_CONTENT_9f7c"},
        }

        result = observation_projection(
            {"id": "msg-001", "metadata": raw_metadata, "surface": "chat"},
            source_kind="message",
            run_id="run-1",
            evidence_id="msg-001",
        )
        projected = result["metadata"]

        assert projected == {
            "requested_model": "gpt-5.6-astra-pro",
            "generation_id": "gen-001",
            "request_id": "req-001",
            "surface": "chat",
        }
        serialized = json.dumps(result)
        for private_value in (
            "PRIVATE_PROMPT_9f7c",
            "PRIVATE_CITATION_TITLE_9f7c",
            "PRIVATE_METADATA_9f7c",
            "PRIVATE_FINISH_DETAIL_9f7c",
            "PRIVATE_NESTED_CONTENT_9f7c",
        ):
            assert private_value not in serialized
        assert "metadata.prompt:string" in result["provenance"]["unknown_fields"]

    def test_metadata_projection_is_closed_for_non_scalar_allowlisted_fields(self):
        projected = sanitize_metadata(
            {
                "generation_id": "gen-002",
                "request_id": ["PRIVATE_LIST_VALUE_9f7c"],
                "status": {"detail": "PRIVATE_STATUS_9f7c"},
                "is_complete": True,
            }
        )

        assert projected == {"generation_id": "gen-002", "is_complete": True}


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class TestAdapter:
    def test_fixture_transport_session(self):
        transport = FixtureTransport(FIXTURE_ROOT)
        payload = transport.request("GET", "/api/auth/session")
        assert payload["user"]["id"] == "user-abc123"

    def test_fixture_transport_disallows_unknown_path(self):
        transport = FixtureTransport(FIXTURE_ROOT)
        with pytest.raises(AdapterError):
            transport.request("GET", "/backend-api/secret/endpoint")

    def test_adapter_inspect_session(self):
        transport = FixtureTransport(FIXTURE_ROOT)
        adapter = ChatGPTHistoryAdapter(
            transport,
            expected_identity={
                "provider_user_id": "user-abc123",
                "workspace_id": "ws-xyz",
                "quota_owner_id": "user-abc123",
            },
        )
        identity = adapter.inspect_session()
        assert identity["auth_state"] == "ready"
        assert identity["surface"] == SURFACE_CHAT
        assert identity["provider_user_id"] == "user-abc123"

    @pytest.mark.parametrize(
        "field",
        ("provider_user_id", "workspace_id", "quota_owner_id"),
    )
    def test_adapter_identity_mismatch(self, field):
        transport = FixtureTransport(FIXTURE_ROOT)
        expected = {
            "provider_user_id": "user-abc123",
            "workspace_id": "ws-xyz",
            "quota_owner_id": "user-abc123",
        }
        expected[field] = "wrong-value"
        adapter = ChatGPTHistoryAdapter(transport, expected_identity=expected)
        identity = adapter.inspect_session()
        assert identity["auth_state"] == "identity_mismatch"
        assert f"{field}_mismatch" in identity["identity_errors"]

    @pytest.mark.parametrize(
        "field",
        ("provider_user_id", "workspace_id", "quota_owner_id"),
    )
    def test_adapter_requires_all_expected_identity_fields(self, field):
        transport = FixtureTransport(FIXTURE_ROOT)
        expected = {
            "provider_user_id": "user-abc123",
            "workspace_id": "ws-xyz",
            "quota_owner_id": "user-abc123",
        }
        expected.pop(field)
        adapter = ChatGPTHistoryAdapter(transport, expected_identity=expected)
        identity = adapter.inspect_session()
        assert identity["auth_state"] == "unconfigured"
        assert f"missing_expected_{field}" in identity["identity_errors"]

    def test_adapter_requires_observed_quota_owner(self):
        session = json.loads((FIXTURE_ROOT / "session.json").read_text())
        session.pop("quota_owner_id")
        adapter = ChatGPTHistoryAdapter(
            SessionTransport(session),
            expected_identity={
                "provider_user_id": "user-abc123",
                "workspace_id": "ws-xyz",
                "quota_owner_id": "user-abc123",
            },
        )

        identity = adapter.inspect_session()

        assert identity["auth_state"] == "identity_mismatch"
        assert "missing_observed_quota_owner_id" in identity["identity_errors"]

    def test_list_conversations_active(self):
        transport = FixtureTransport(FIXTURE_ROOT)
        adapter = ChatGPTHistoryAdapter(transport)
        page = adapter.list_conversations(archived=False, offset=0, limit=100)
        assert page.exhausted is True
        assert len(page.items) == 2
        assert all(isinstance(item, ConversationSummary) for item in page.items)
        assert all(item.surface == SURFACE_CHAT for item in page.items)

    def test_fetch_conversation_with_mapping(self):
        transport = FixtureTransport(FIXTURE_ROOT)
        adapter = ChatGPTHistoryAdapter(transport)
        payload = adapter.fetch_conversation("conv-001")
        assert isinstance(payload["mapping"], dict)
        assert len(payload["mapping"]) == 4

    def test_missing_detail_cursor_is_incomplete(self):
        page = adapt_message_page(
            {
                "messages": [
                    {
                        "id": "msg-001",
                        "author": {"role": "assistant"},
                        "metadata": {"generation_id": "gen-001"},
                    }
                ],
                "page_info": {"has_previous_page": True},
            },
            conversation_id="conv-001",
            conversation_surface=SURFACE_CHAT,
        )

        assert page.coverage == "unrecognized"
        assert page.exhausted is False
        assert page.continuation is None
        assert "has_previous_page_without_start_cursor" in page.warnings

    def test_unknown_200_detail_shape_is_incomplete(self):
        page = adapt_message_page(
            {"status": "ok", "future_shape": {"items": []}},
            conversation_id="conv-001",
            conversation_surface=SURFACE_CHAT,
        )

        assert page.coverage == "unrecognized"
        assert page.exhausted is False
        assert "unrecognized_detail_shape" in page.warnings


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------

class TestLedger:
    @pytest.fixture
    def ledger(self):
        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
            path = f.name
        ledger = Ledger(path)
        yield ledger
        ledger.close()
        Path(path).unlink(missing_ok=True)

    def test_upsert_account(self, ledger):
        ledger.upsert_account({
            "collector_account_id": "test-1",
            "provider_user_id": "user-123",
            "workspace_id": "ws-1",
            "quota_owner_id": "user-123",
            "surface": SURFACE_CHAT,
            "auth_state": "ready",
            "plan_policy_id": "pro200",
        })
        # Should not raise

    def test_message_ledger_receives_only_projected_metadata(self, ledger):
        ledger.upsert_account(
            {
                "collector_account_id": "test-metadata",
                "quota_owner_id": "user-123",
                "surface": SURFACE_CHAT,
                "auth_state": "ready",
            }
        )
        warnings: list[str] = []
        record = message_from_node(
            {
                "id": "node-001",
                "message": {
                    "id": "msg-001",
                    "author": {"role": "assistant"},
                    "metadata": {
                        "generation_id": {
                            "prompt": "PRIVATE_GENERATION_PROMPT_ledger_9f7c"
                        },
                        "request_id": "req-001",
                        "prompt": "PRIVATE_PROMPT_ledger_9f7c",
                        "citations": [
                            {"title": "PRIVATE_CITATION_ledger_9f7c"}
                        ],
                        "private_note": "PRIVATE_METADATA_ledger_9f7c",
                    },
                },
            },
            conversation_id="conv-001",
            warnings=warnings,
            conversation_surface=SURFACE_CHAT,
        )

        assert record is not None
        ledger.upsert_message("test-metadata", record)
        stored = ledger.messages_for("test-metadata", "conv-001")

        assert stored[0].metadata == {
            "request_id": "req-001",
        }
        assert stored[0].generation_id is None
        assert "PRIVATE_GENERATION_PROMPT_ledger_9f7c" not in json.dumps(
            stored[0].metadata
        )
        assert "PRIVATE_PROMPT_ledger_9f7c" not in json.dumps(stored[0].metadata)
        assert "PRIVATE_CITATION_ledger_9f7c" not in json.dumps(stored[0].metadata)

    def test_rejects_non_chat_surface(self, ledger):
        with pytest.raises(ValueError, match="surface=chat"):
            ledger.upsert_account({
                "collector_account_id": "test-2",
                "provider_user_id": "user-456",
                "workspace_id": "ws-2",
                "quota_owner_id": "user-456",
                "surface": "codex",
                "auth_state": "ready",
                "plan_policy_id": "pro200",
            })

    def test_record_run_start_and_finish(self, ledger):
        ledger.upsert_account({
            "collector_account_id": "test-3",
            "quota_owner_id": "user-789",
            "surface": SURFACE_CHAT,
            "auth_state": "ready",
        })
        now = datetime.now(timezone.utc)
        ledger.record_run_start(run_id="run-1", account_id="test-3", mode="refresh", started_at=now)
        ledger.finish_run("run-1", ended_at=now, result="complete", new_attempts=5)

    def test_discovery_state_lifecycle(self, ledger):
        ledger.upsert_discovery_state(
            account_id="test-3",
            scope="active",
            last_complete_discovery_started_at=datetime.now(timezone.utc),
            continuation="50",
            coverage="validated_page",
        )
        state = ledger.get_discovery_state("test-3", "active")
        assert state is not None
        assert state["continuation"] == "50"
        assert state["coverage"] == "validated_page"

    def test_upsert_attempt_insert_and_deduplicate(self, ledger):
        ledger.upsert_account({
            "collector_account_id": "test-4",
            "quota_owner_id": "user-000",
            "surface": SURFACE_CHAT,
            "auth_state": "ready",
        })
        attempt = AttemptRecord(
            attempt_id="att-001",
            conversation_id="conv-001",
            identity_basis="generation",
            time_basis="user_message",
            attempt_time=datetime.now(timezone.utc),
            earliest_possible_at=datetime.now(timezone.utc),
            latest_possible_at=datetime.now(timezone.utc),
            requested_model_raw="gpt-5.6-astra-pro",
            requested_mode_raw=None,
            requested_reasoning_effort_raw=None,
            recorded_final_model_raw="gpt-5.6-astra-pro",
            resolved_model_raw=None,
            requested_family="astra_pro",
            recorded_final_family="astra_pro",
            resolved_family=None,
            mapping_version="test",
            outcome="completed",
            completed_answer=True,
            generation_started=True,
            surface=SURFACE_CHAT,
            origin=None,
            aliases=(("generation", "gen-1"),),
            evidence_message_ids=("msg-1", "msg-2"),
            revision=1,
            warnings=(),
        )
        status = ledger.upsert_attempt("test-4", attempt)
        assert status == "inserted"
        # Same attempt_id should deduplicate
        status = ledger.upsert_attempt("test-4", attempt)
        assert status == "deduplicated"

    def test_scheduler_state(self, ledger):
        ledger.upsert_scheduler_state(
            "test-4",
            refresh_interval="PT1H",
            next_due_at=isoformat_utc(datetime.now(timezone.utc)),
            lease_owner="cli",
            lease_token="tok-123",
            lease_until=isoformat_utc(datetime.now(timezone.utc) + timedelta(minutes=25)),
        )
        state = ledger.get_scheduler_state("test-4")
        assert state is not None
        assert state["lease_owner"] == "cli"


# ---------------------------------------------------------------------------
# Collector integration (fixture-backed)
# ---------------------------------------------------------------------------

class TestCollectorIntegration:
    @pytest.fixture
    def config_and_ledger(self):
        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
            path = f.name
        config = _make_config()
        ledger = Ledger(path)
        yield config, ledger
        ledger.close()
        Path(path).unlink(missing_ok=True)

    def test_collect_with_fixtures(self, config_and_ledger):
        config, ledger = config_and_ledger
        collector = Collector(config, ledger, fixture_root=str(FIXTURE_ROOT))
        result = collector.collect(mode="refresh")
        assert result.result in ("complete", "partial")
        assert result.conversations_seen == 2
        assert result.new_attempts >= 0
        # Should have captured attempts from both conversations
        assert result.pages_fetched >= 3  # 1 index + 2 conversations

    def test_inspect_capabilities(self, config_and_ledger):
        config, ledger = config_and_ledger
        collector = Collector(config, ledger, fixture_root=str(FIXTURE_ROOT))
        result = collector.inspect_capabilities()
        assert result["surface"] == SURFACE_CHAT
        assert result["identity"]["auth_state"] == "ready"
        assert "capabilities" in result

    def test_identity_mismatch_does_not_persist_configured_identity_as_observed(self):
        payload = copy.deepcopy(MINIMAL_CONFIG)
        payload["accounts"][0]["expected_provider_user_id"] = "wrong-user"
        config = parse_config(payload, source_path=Path("/tmp/test-config.yaml"))
        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
            path = f.name
        ledger = Ledger(path)
        try:
            result = Collector(
                config,
                ledger,
                fixture_root=str(FIXTURE_ROOT),
            ).collect(mode="refresh")

            assert result.result == "identity_mismatch"
            row = ledger.conn.execute(
                """
                SELECT provider_user_id, workspace_id, auth_state
                FROM accounts
                WHERE collector_account_id=?
                """,
                ("test-account",),
            ).fetchone()
            assert row["provider_user_id"] is None
            assert row["workspace_id"] is None
            assert row["auth_state"] == "identity_mismatch"
            assert ledger.conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 0
        finally:
            ledger.close()
            Path(path).unlink(missing_ok=True)

    def test_incomplete_detail_does_not_advance_watermark(self, config_and_ledger):
        config, ledger = config_and_ledger
        transport = IncompleteDetailTransport()
        adapter = ChatGPTHistoryAdapter(
            transport,
            expected_identity={
                "provider_user_id": "user-abc123",
                "workspace_id": "ws-xyz",
                "quota_owner_id": "user-abc123",
            },
        )

        result = Collector(config, ledger, adapter=adapter).collect(mode="refresh")

        assert result.result == "partial"
        assert "detail_coverage:unrecognized" in result.warnings
        state = ledger.get_discovery_state("test-account", "active")
        assert state is not None
        assert state["coverage"] == "partial"
        assert state["last_complete_discovery_started_at"] is None
        assert state["watermark_at"] is None
        assert any(
            gap["reason"] == "detail_pagination_incomplete"
            for gap in ledger.list_coverage_gaps("test-account")
        )
        pending = ledger.conn.execute(
            """
            SELECT pending, page_coverage
            FROM conversation_state
            WHERE collector_account_id=? AND conversation_id=?
            """,
            ("test-account", "conv-incomplete"),
        ).fetchone()
        assert pending["pending"] == 1

    def test_report_after_collect(self, config_and_ledger):
        config, ledger = config_and_ledger
        collector = Collector(config, ledger, fixture_root=str(FIXTURE_ROOT))
        collector.collect(mode="refresh")
        report = build_report(config, ledger)
        assert report["account_id"] == "test-account"
        assert report["surface"] == SURFACE_CHAT
        assert "range" in report
        assert "freshness" in report


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

class TestReporting:
    @pytest.fixture
    def config_and_ledger(self):
        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
            path = f.name
        config = _make_config()
        ledger = Ledger(path)
        yield config, ledger
        ledger.close()
        Path(path).unlink(missing_ok=True)

    def test_report_empty_ledger(self, config_and_ledger):
        config, ledger = config_and_ledger
        report = build_report(config, ledger)
        assert report["account_id"] == "test-account"
        assert report["surface"] == SURFACE_CHAT
        assert report["observed_attempts_by_requested_family"] == {}


# ---------------------------------------------------------------------------
# Time utilities
# ---------------------------------------------------------------------------

class TestTimeUtil:
    def test_parse_iso_duration(self):
        assert parse_iso_duration("PT1H") == timedelta(hours=1)
        assert parse_iso_duration("PT5M") == timedelta(minutes=5)
        assert parse_iso_duration("P14D") == timedelta(days=14)
        assert parse_iso_duration("PT48H") == timedelta(hours=48)

    def test_ensure_utc(self):
        dt = datetime(2026, 9, 5, 12, 0, 0, tzinfo=timezone.utc)
        assert ensure_utc(dt) == dt
        naive = datetime(2026, 9, 5, 12, 0, 0)
        result = ensure_utc(naive)
        assert result.tzinfo == timezone.utc
