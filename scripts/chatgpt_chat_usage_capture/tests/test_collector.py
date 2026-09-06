"""Focused tests for ChatGPT Chat usage collector (D1-752)."""

from __future__ import annotations

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
    sanitize_identity,
    sanitize_mapping,
    observation_projection,
)
from scripts.chatgpt_chat_usage_capture.adapter import ChatGPTHistoryAdapter, FixtureTransport, AdapterError
from scripts.chatgpt_chat_usage_capture.collector import Collector
from scripts.chatgpt_chat_usage_capture.reporting import build_report
from scripts.chatgpt_chat_usage_capture.timeutil import ensure_utc, parse_iso_duration, isoformat_utc
from scripts.chatgpt_chat_usage_capture.models import ConversationSummary, AttemptRecord


FIXTURE_ROOT = Path(__file__).parent / "fixtures"


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
        adapter = ChatGPTHistoryAdapter(transport, expected_identity={"provider_user_id": "user-abc123"})
        identity = adapter.inspect_session()
        assert identity["auth_state"] == "ready"
        assert identity["surface"] == SURFACE_CHAT
        assert identity["provider_user_id"] == "user-abc123"

    def test_adapter_identity_mismatch(self):
        transport = FixtureTransport(FIXTURE_ROOT)
        adapter = ChatGPTHistoryAdapter(transport, expected_identity={"provider_user_id": "wrong-user"})
        identity = adapter.inspect_session()
        assert identity["auth_state"] == "identity_mismatch"

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
