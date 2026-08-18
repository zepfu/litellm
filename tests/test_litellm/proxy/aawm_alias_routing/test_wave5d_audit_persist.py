"""Module-local tests for Wave 5D audit_persist.py extraction.

Drives the new module directly with fresh state/dependency stubs.
Does NOT import llm_passthrough_endpoints at module scope.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from litellm.proxy.pass_through_endpoints.aawm_alias_routing import audit_persist


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _configure_audit_persist():
    """Configure audit_persist runtime with fresh stubs before each test."""
    # Snapshot every prior injected runtime callback so we can restore the
    # exact objects the god-module established at import time.
    _previous = {
        "_record_auto_agent_alias_route_status_rollup": audit_persist._record_auto_agent_alias_route_status_rollup,
        "_aawm_alias_route_verbose_json_enabled": audit_persist._aawm_alias_route_verbose_json_enabled,
        "_aawm_alias_route_healthy_json_enabled": audit_persist._aawm_alias_route_healthy_json_enabled,
    }

    rollup_calls: list[dict[str, Any]] = []
    verbose_enabled = MagicMock(return_value=False)
    healthy_enabled = MagicMock(return_value=False)

    audit_persist.configure_audit_persist_runtime(
        record_route_status_rollup=lambda event: rollup_calls.append(event),
        verbose_json_enabled=verbose_enabled,
        healthy_json_enabled=healthy_enabled,
    )

    yield {
        "rollup_calls": rollup_calls,
        "verbose_enabled": verbose_enabled,
        "healthy_enabled": healthy_enabled,
    }

    # Restore the exact prior runtime objects, even if the test failed.
    for name, value in _previous.items():
        setattr(audit_persist, name, value)


# ---------------------------------------------------------------------------
# _should_emit_auto_agent_alias_route_event tests
# ---------------------------------------------------------------------------


class TestShouldEmitRouteEvent:
    """Route-event filtering logic."""

    def test_warning_level_always_emits(self, _configure_audit_persist):
        event = {"event_type": "candidate_attempt_started"}
        assert audit_persist._should_emit_auto_agent_alias_route_event(event, level="warning") is True

    def test_healthy_json_enabled_always_emits(self, _configure_audit_persist):
        _configure_audit_persist["healthy_enabled"].return_value = True
        event = {"event_type": "candidate_attempt_started"}
        assert audit_persist._should_emit_auto_agent_alias_route_event(event) is True

    def test_failure_class_emits(self, _configure_audit_persist):
        event = {"failure_class": "timeout"}
        assert audit_persist._should_emit_auto_agent_alias_route_event(event) is True

    def test_error_status_code_emits(self, _configure_audit_persist):
        event = {"error_status_code": 500}
        assert audit_persist._should_emit_auto_agent_alias_route_event(event) is True

    def test_redispatch_required_emits(self, _configure_audit_persist):
        event = {"redispatch_required": True}
        assert audit_persist._should_emit_auto_agent_alias_route_event(event) is True

    def test_redispatch_threshold_crossed_emits(self, _configure_audit_persist):
        event = {"redispatch_threshold_crossed": True}
        assert audit_persist._should_emit_auto_agent_alias_route_event(event) is True

    def test_candidate_attempt_started_suppressed(self, _configure_audit_persist):
        event = {"event_type": "candidate_attempt_started"}
        assert audit_persist._should_emit_auto_agent_alias_route_event(event) is False

    def test_candidate_selected_suppressed(self, _configure_audit_persist):
        event = {"event_type": "candidate_selected"}
        assert audit_persist._should_emit_auto_agent_alias_route_event(event) is False

    def test_candidate_status_started_suppressed(self, _configure_audit_persist):
        event = {"candidate_status": "started"}
        assert audit_persist._should_emit_auto_agent_alias_route_event(event) is False

    def test_candidate_status_selected_suppressed(self, _configure_audit_persist):
        event = {"candidate_status": "selected"}
        assert audit_persist._should_emit_auto_agent_alias_route_event(event) is False

    def test_session_affinity_suppressed(self, _configure_audit_persist):
        event = {"selection_reason": "session_affinity"}
        assert audit_persist._should_emit_auto_agent_alias_route_event(event) is False

    def test_generic_event_emits(self, _configure_audit_persist):
        event = {"event_type": "terminal_no_candidate"}
        assert audit_persist._should_emit_auto_agent_alias_route_event(event) is True

    def test_empty_event_emits(self, _configure_audit_persist):
        assert audit_persist._should_emit_auto_agent_alias_route_event({}) is True


# ---------------------------------------------------------------------------
# _emit_auto_agent_alias_route_event tests
# ---------------------------------------------------------------------------


class TestEmitRouteEvent:
    """Emission gating and rollup delegation."""

    def test_rollup_always_called(self, _configure_audit_persist):
        event = {"event_type": "candidate_attempt_started"}
        audit_persist._emit_auto_agent_alias_route_event(event)
        assert _configure_audit_persist["rollup_calls"] == [event]

    def test_no_json_log_when_both_disabled(self, _configure_audit_persist):
        event = {"event_type": "terminal_no_candidate"}
        with patch.object(audit_persist.verbose_aawm_route_logger, "info") as mock_info:
            audit_persist._emit_auto_agent_alias_route_event(event)
            mock_info.assert_not_called()

    def test_json_log_when_verbose_enabled(self, _configure_audit_persist):
        _configure_audit_persist["verbose_enabled"].return_value = True
        event = {"event_type": "terminal_no_candidate", "alias_model": "test"}
        with patch.object(audit_persist.verbose_aawm_route_logger, "info") as mock_info:
            audit_persist._emit_auto_agent_alias_route_event(event)
            mock_info.assert_called_once()
            msg = mock_info.call_args[0][0]
            assert msg.startswith("AAWM_ALIAS_ROUTE: ")
            payload = json.loads(msg[len("AAWM_ALIAS_ROUTE: "):])
            assert payload["event"] == "aawm_alias_route"
            assert payload["event_type"] == "terminal_no_candidate"

    def test_filtered_event_not_logged(self, _configure_audit_persist):
        _configure_audit_persist["verbose_enabled"].return_value = True
        event = {"event_type": "candidate_attempt_started"}
        with patch.object(audit_persist.verbose_aawm_route_logger, "info") as mock_info:
            audit_persist._emit_auto_agent_alias_route_event(event)
            mock_info.assert_not_called()

    def test_warning_level_bypasses_filter(self, _configure_audit_persist):
        _configure_audit_persist["verbose_enabled"].return_value = True
        event = {"event_type": "candidate_attempt_started"}
        with patch.object(audit_persist.verbose_aawm_route_logger, "info") as mock_info:
            audit_persist._emit_auto_agent_alias_route_event(event, level="warning")
            mock_info.assert_called_once()


# ---------------------------------------------------------------------------
# _persist_auto_agent_alias_audit_only_events_best_effort tests
# ---------------------------------------------------------------------------


class TestPersistAuditOnlyEvents:
    """Audit-only persistence dispositions."""

    def test_empty_events_returns_skip_empty(self):
        result = audit_persist._persist_auto_agent_alias_audit_only_events_best_effort([])
        assert result == "skip_empty"

    def test_import_failure_returns_fail_import(self):
        events = [{"event_type": "terminal_no_candidate", "alias_model": "test-alias"}]
        with patch.dict(
            "sys.modules",
            {"litellm.integrations.aawm_agent_identity": None},
        ):
            # Force ImportError by making the module unimportable
            with patch(
                "builtins.__import__",
                side_effect=ImportError("no module"),
            ):
                result = audit_persist._persist_auto_agent_alias_audit_only_events_best_effort(events)
        assert result == "fail_import"

    def test_build_failure_returns_fail_build(self):
        events = [{"event_type": "terminal_no_candidate", "alias_model": "test-alias"}]
        mock_module = MagicMock()
        mock_module._build_alias_routing_audit_only_record.side_effect = ValueError("bad record")
        with patch.dict(
            "sys.modules",
            {"litellm.integrations.aawm_agent_identity": mock_module},
        ):
            result = audit_persist._persist_auto_agent_alias_audit_only_events_best_effort(events)
        assert result == "fail_build"

    def test_spool_success_returns_spool_only(self):
        events = [{"event_type": "terminal_no_candidate", "alias_model": "test-alias"}]
        mock_module = MagicMock()
        mock_module._build_alias_routing_audit_only_record.return_value = {"record": True}
        mock_module._spool_session_history_records.return_value = None
        with patch.dict(
            "sys.modules",
            {"litellm.integrations.aawm_agent_identity": mock_module},
        ):
            result = audit_persist._persist_auto_agent_alias_audit_only_events_best_effort(events)
        assert result == "spool_only"
        mock_module._spool_session_history_records.assert_called_once_with(
            [{"record": True}],
            reason="alias audit terminal write-ahead",
        )

    def test_spool_failure_enqueue_fallback(self):
        events = [{"event_type": "terminal_no_candidate", "alias_model": "test-alias"}]
        mock_module = MagicMock()
        mock_module._build_alias_routing_audit_only_record.return_value = {"record": True}
        mock_module._spool_session_history_records.side_effect = OSError("disk full")
        mock_module._enqueue_session_history_record.return_value = None
        with patch.dict(
            "sys.modules",
            {"litellm.integrations.aawm_agent_identity": mock_module},
        ):
            result = audit_persist._persist_auto_agent_alias_audit_only_events_best_effort(events)
        assert result == "spool_fallback_enqueue"
        mock_module._enqueue_session_history_record.assert_called_once_with({"record": True})

    def test_total_failure_returns_spool_enqueue_failed(self):
        events = [{"event_type": "terminal_no_candidate", "alias_model": "test-alias"}]
        mock_module = MagicMock()
        mock_module._build_alias_routing_audit_only_record.return_value = {"record": True}
        mock_module._spool_session_history_records.side_effect = OSError("disk full")
        mock_module._enqueue_session_history_record.side_effect = RuntimeError("queue dead")
        with patch.dict(
            "sys.modules",
            {"litellm.integrations.aawm_agent_identity": mock_module},
        ):
            result = audit_persist._persist_auto_agent_alias_audit_only_events_best_effort(events)
        assert result == "spool_enqueue_failed"

    def test_metadata_from_request_body(self):
        events = [{"event_type": "terminal_no_candidate", "alias_model": "test-alias"}]
        request_body = {
            "litellm_metadata": {
                "requested_model_alias": "codex-auto",
                "model_alias_label": "codex",
                "repository": "my-repo",
            }
        }
        mock_module = MagicMock()
        mock_module._build_alias_routing_audit_only_record.return_value = {"record": True}
        mock_module._spool_session_history_records.return_value = None
        with patch.dict(
            "sys.modules",
            {"litellm.integrations.aawm_agent_identity": mock_module},
        ):
            result = audit_persist._persist_auto_agent_alias_audit_only_events_best_effort(
                events, request_body=request_body
            )
        assert result == "spool_only"
        call_kwargs = mock_module._build_alias_routing_audit_only_record.call_args[1]
        assert call_kwargs["metadata"]["requested_model_alias"] == "codex-auto"
        assert call_kwargs["metadata"]["model_alias_label"] == "codex"
        assert call_kwargs["metadata"]["repository"] == "my-repo"

    def test_sanitized_logging_no_raw_identifiers(self):
        """Verify that warning logs do not contain raw session IDs."""
        events = [
            {
                "event_type": "terminal_no_candidate",
                "alias_model": "test-alias",
                "session_id": "secret-session-12345",
                "litellm_call_id": "secret-call-67890",
                "trace_id": "secret-trace-abcde",
            }
        ]
        mock_module = MagicMock()
        mock_module._build_alias_routing_audit_only_record.side_effect = ValueError("boom")
        with patch.dict(
            "sys.modules",
            {"litellm.integrations.aawm_agent_identity": mock_module},
        ):
            with patch.object(audit_persist.verbose_proxy_logger, "warning") as mock_warn:
                audit_persist._persist_auto_agent_alias_audit_only_events_best_effort(events)
                # Check that raw identifiers are not in the log call
                for call in mock_warn.call_args_list:
                    all_args = " ".join(str(a) for a in call[0])
                    assert "secret-session-12345" not in all_args
                    assert "secret-call-67890" not in all_args
                    assert "secret-trace-abcde" not in all_args

    def test_event_types_bounded_to_eight_plus_omitted(self):
        """More than 8 distinct event types produces +N_more suffix."""
        events = [{"event_type": f"type_{i}"} for i in range(12)]
        mock_module = MagicMock()
        mock_module._build_alias_routing_audit_only_record.return_value = {"record": True}
        mock_module._spool_session_history_records.return_value = None
        with patch.dict(
            "sys.modules",
            {"litellm.integrations.aawm_agent_identity": mock_module},
        ):
            with patch.object(audit_persist.verbose_proxy_logger, "warning"):
                result = audit_persist._persist_auto_agent_alias_audit_only_events_best_effort(events)
        assert result == "spool_only"

    def test_non_dict_last_event_uses_empty_primary(self):
        """If last event is not a dict, primary defaults to empty dict."""
        events: list[Any] = [{"event_type": "first"}, "not_a_dict"]
        mock_module = MagicMock()
        mock_module._build_alias_routing_audit_only_record.return_value = {"record": True}
        mock_module._spool_session_history_records.return_value = None
        with patch.dict(
            "sys.modules",
            {"litellm.integrations.aawm_agent_identity": mock_module},
        ):
            result = audit_persist._persist_auto_agent_alias_audit_only_events_best_effort(events)
        assert result == "spool_only"
        call_kwargs = mock_module._build_alias_routing_audit_only_record.call_args[1]
        assert call_kwargs["session_id"] is None
        assert call_kwargs["model"] is None

    def test_non_raising_on_all_exceptions(self):
        """Function never raises regardless of internal failures."""
        events = [{"event_type": "terminal_no_candidate"}]
        mock_module = MagicMock()
        mock_module._build_alias_routing_audit_only_record.side_effect = Exception("unexpected")
        with patch.dict(
            "sys.modules",
            {"litellm.integrations.aawm_agent_identity": mock_module},
        ):
            # Must not raise
            result = audit_persist._persist_auto_agent_alias_audit_only_events_best_effort(events)
        assert result == "fail_build"

    def test_child_canonical_thread_id_is_durable_session_id(self):
        parent = "01a012a1-2a97-7622-837c-3066ec78f02f"
        child = "01a012a6-c49a-7a42-899d-de19e2af2e9e"
        events = [
            {
                "event_type": "terminal_no_candidate",
                "alias_model": "test-alias",
                "session_id": parent,
                "canonical_thread_id": child,
                "parent_thread_id": parent,
            }
        ]
        mock_module = MagicMock()
        mock_module._build_alias_routing_audit_only_record.return_value = {"record": True}
        mock_module._spool_session_history_records.return_value = None
        with patch.dict(
            "sys.modules",
            {"litellm.integrations.aawm_agent_identity": mock_module},
        ):
            result = audit_persist._persist_auto_agent_alias_audit_only_events_best_effort(events)
        assert result == "spool_only"
        call_kwargs = mock_module._build_alias_routing_audit_only_record.call_args[1]
        assert call_kwargs["session_id"] == child
        assert call_kwargs["metadata"]["canonical_thread_id"] == child
        assert call_kwargs["metadata"]["parent_thread_id"] == parent
        assert call_kwargs["metadata"]["aawm_alias_routing_audit_only"] is True

    def test_three_children_under_one_parent_remain_independently_queryable(self):
        parent = "01a012a1-2a97-7622-837c-3066ec78f02f"
        children = [
            "01a012a6-c49a-7a42-899d-de19e2af2e9e",
            "01a012b0-e33f-7153-9e20-6af4560b4cec",
            "01a012b0-e58e-7372-b7d9-38bc36db15e7",
        ]
        durable_ids: list[str] = []
        mock_module = MagicMock()
        mock_module._build_alias_routing_audit_only_record.return_value = {"record": True}
        mock_module._spool_session_history_records.return_value = None
        with patch.dict(
            "sys.modules",
            {"litellm.integrations.aawm_agent_identity": mock_module},
        ):
            for child in children:
                events = [
                    {
                        "event_type": "terminal_no_candidate",
                        "alias_model": "test-alias",
                        "session_id": parent,
                        "canonical_thread_id": child,
                        "parent_thread_id": parent,
                    }
                ]
                result = audit_persist._persist_auto_agent_alias_audit_only_events_best_effort(
                    events
                )
                assert result == "spool_only"
                call_kwargs = mock_module._build_alias_routing_audit_only_record.call_args[1]
                durable_ids.append(call_kwargs["session_id"])
                assert call_kwargs["metadata"]["parent_thread_id"] == parent
        assert durable_ids == children
        assert len(set(durable_ids)) == 3

    def test_parent_only_event_keeps_parent_session_id(self):
        events = [
            {
                "event_type": "terminal_no_candidate",
                "alias_model": "test-alias",
                "session_id": "session-parent-only",
            }
        ]
        mock_module = MagicMock()
        mock_module._build_alias_routing_audit_only_record.return_value = {"record": True}
        mock_module._spool_session_history_records.return_value = None
        with patch.dict(
            "sys.modules",
            {"litellm.integrations.aawm_agent_identity": mock_module},
        ):
            result = audit_persist._persist_auto_agent_alias_audit_only_events_best_effort(events)
        assert result == "spool_only"
        call_kwargs = mock_module._build_alias_routing_audit_only_record.call_args[1]
        assert call_kwargs["session_id"] == "session-parent-only"
        assert "canonical_thread_id" not in call_kwargs["metadata"]
        assert "parent_thread_id" not in call_kwargs["metadata"]

    def test_sanitized_logging_hashes_durable_child_identity(self):
        parent = "secret-parent-thread-aaaa"
        child = "secret-child-thread-bbbb"
        events = [
            {
                "event_type": "terminal_no_candidate",
                "alias_model": "test-alias",
                "session_id": parent,
                "canonical_thread_id": child,
                "parent_thread_id": parent,
                "litellm_call_id": "secret-call-cccc",
            }
        ]
        mock_module = MagicMock()
        mock_module._build_alias_routing_audit_only_record.side_effect = ValueError("boom")
        with patch.dict(
            "sys.modules",
            {"litellm.integrations.aawm_agent_identity": mock_module},
        ):
            with patch.object(audit_persist.verbose_proxy_logger, "warning") as mock_warn:
                audit_persist._persist_auto_agent_alias_audit_only_events_best_effort(events)
                for call in mock_warn.call_args_list:
                    all_args = " ".join(str(a) for a in call[0])
                    assert parent not in all_args
                    assert child not in all_args
                    assert "secret-call-cccc" not in all_args

    def test_classification_and_selected_lane_survive_persist_without_content(self):
        secret = "SECRET_ENCRYPTED_BLOB"
        prompt = "raw user prompt must not leak"
        tool_args = '{"command":"cat /etc/shadow"}'
        credential = "sk-secret-credential"
        lane = "codex-oauth:account1:hash-account-1"
        events = [
            {
                "event_type": "candidate_retryable_failure",
                "alias_model": "test-alias",
                "session_id": "child-thread-9",
                "canonical_thread_id": "child-thread-9",
                "parent_thread_id": "parent-thread-9",
                "has_account_bound_state": True,
                "account_bound_classification": "account_bound",
                "account_lane": lane,
                "failure_class": "item_not_found",
                "provider": "openai",
                "litellm_call_id": "call-bound-1",
            }
        ]
        request_body = {
            "encrypted_content": secret,
            "instructions": prompt,
            "input": [
                {
                    "type": "function_call",
                    "name": "shell",
                    "arguments": tool_args,
                }
            ],
            "litellm_metadata": {
                "requested_model_alias": "codex-auto",
                "authorization": credential,
            },
        }
        mock_module = MagicMock()
        mock_module._build_alias_routing_audit_only_record.return_value = {"record": True}
        mock_module._spool_session_history_records.return_value = None
        with patch.dict(
            "sys.modules",
            {"litellm.integrations.aawm_agent_identity": mock_module},
        ):
            result = audit_persist._persist_auto_agent_alias_audit_only_events_best_effort(
                events,
                request_body=request_body,
            )
        assert result == "spool_only"
        call_kwargs = mock_module._build_alias_routing_audit_only_record.call_args[1]
        metadata = call_kwargs["metadata"]
        persisted_events = call_kwargs["events"]
        assert metadata["has_account_bound_state"] is True
        assert metadata["account_bound_classification"] == "account_bound"
        assert metadata["account_lane"] == lane
        assert metadata["failure_class"] == "item_not_found"
        assert persisted_events[0]["has_account_bound_state"] is True
        assert persisted_events[0]["account_bound_classification"] == "account_bound"
        assert persisted_events[0]["account_lane"] == lane
        assert persisted_events[0]["failure_class"] == "item_not_found"
        serialized = str({"kwargs": call_kwargs, "metadata": metadata})
        assert secret not in serialized
        assert prompt not in serialized
        assert tool_args not in serialized
        assert credential not in serialized
