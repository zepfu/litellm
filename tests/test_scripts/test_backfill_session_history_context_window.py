import argparse
import json
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import scripts.backfill_session_history as backfill_session_history
from litellm.integrations import aawm_agent_identity


def test_backfill_langfuse_record_classifies_suffix_1m_metadata() -> None:
    trace = {"id": "trace-1", "sessionId": "sess-1", "input": "", "output": ""}
    observation = {
        "id": "obs-1",
        "type": "GENERATION",
        "model": "claude-opus-4-7",
        "startTime": "2026-01-01T00:00:00Z",
        "endTime": "2026-01-01T00:00:01Z",
        "metadata": {
            "session_id": "sess-1",
            "requested_model_alias": "claude-opus-4-7[1m]",
            "custom_llm_provider": "anthropic",
            "passthrough_route_family": "anthropic_messages",
        },
        "usageDetails": {"input": 1, "output": 1, "total": 2},
    }

    record = aawm_agent_identity._build_session_history_record_from_langfuse_trace_observation(
        trace,
        observation,
        backfill_run_id="run-1",
    )
    assert record is not None
    meta = record["metadata"]
    assert meta["anthropic_context_window_mode"] == "extended_1m"
    assert meta["anthropic_context_window_source"] == "model_suffix_1m"
    assert meta["anthropic_context_window_classification"] == "classified"


def test_backfill_enrich_marks_unavailable_without_evidence() -> None:
    record = {
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "inbound_model_alias": "claude-sonnet-4-6",
        "metadata": {
            "custom_llm_provider": "anthropic",
            "passthrough_route_family": "anthropic_messages",
        },
    }
    aawm_agent_identity._enrich_backfill_anthropic_context_window_metadata(record)
    meta = record["metadata"]
    assert meta["anthropic_context_window_mode"] == "unknown"
    assert meta["anthropic_context_window_requested_tokens"] is None
    assert meta["anthropic_context_window_source"] == "unavailable"
    assert meta["anthropic_context_window_classification"] == "unavailable"


def test_resolve_repair_modes_anthropic_only_skips_tenant_and_cost() -> None:
    args = argparse.Namespace(
        repair_costs=False,
        repair_tenant_ids=False,
        repair_anthropic_context_window=True,
    )
    modes = backfill_session_history._resolve_session_history_repair_modes(args)
    assert modes == {
        "repair_costs": False,
        "repair_tenant_ids": False,
        "repair_anthropic_context_window": True,
    }


def test_resolve_repair_modes_default_still_repairs_tenant_and_cost() -> None:
    args = argparse.Namespace(
        repair_costs=False,
        repair_tenant_ids=False,
        repair_anthropic_context_window=False,
    )
    modes = backfill_session_history._resolve_session_history_repair_modes(args)
    assert modes["repair_costs"] is True
    assert modes["repair_tenant_ids"] is True
    assert modes["repair_anthropic_context_window"] is False


def test_repair_selection_extended_1m_from_retained_suffix_metadata() -> None:
    metadata = {
        "custom_llm_provider": "anthropic",
        "requested_model_alias": "claude-opus-4-7[1m]",
    }
    row = {
        "provider": "anthropic",
        "model": "claude-opus-4-7",
        "inbound_model_alias": "claude-opus-4-7",
        "metadata": metadata,
    }
    assert backfill_session_history._session_history_row_needs_anthropic_context_window_metadata_repair(
        row, dict(metadata)
    )
    record = {**row, "metadata": dict(metadata)}
    aawm_agent_identity._enrich_backfill_anthropic_context_window_metadata(record)
    assert record["metadata"]["anthropic_context_window_mode"] == "extended_1m"
    assert record["metadata"]["anthropic_context_window_source"] == "model_suffix_1m"


def test_repair_selection_unavailable_on_anthropic_row_without_evidence() -> None:
    metadata = {
        "custom_llm_provider": "anthropic",
        "passthrough_route_family": "anthropic_messages",
    }
    row = {
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "metadata": metadata,
    }
    assert backfill_session_history._session_history_row_needs_anthropic_context_window_metadata_repair(
        row, dict(metadata)
    )


def test_repair_selection_skips_non_anthropic_rows() -> None:
    metadata = {"custom_llm_provider": "openai"}
    row = {"provider": "openai", "model": "gpt-4.1", "metadata": metadata}
    assert not backfill_session_history._session_history_row_needs_anthropic_context_window_metadata_repair(
        row, metadata
    )


def test_parser_accepts_repair_anthropic_context_window_flag() -> None:
    parser = backfill_session_history._build_arg_parser()
    args = parser.parse_args(
        ["--repair-session-history", "--repair-anthropic-context-window"]
    )
    assert args.repair_session_history is True
    assert args.repair_anthropic_context_window is True


@pytest.mark.asyncio
async def test_run_session_history_repair_dry_run_counts_anthropic_updates() -> None:
    row = {
        "litellm_call_id": "call-1",
        "tenant_id": None,
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "inbound_model_alias": "claude-sonnet-4-6",
        "input_tokens": 10,
        "output_tokens": 5,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "response_cost_usd": 0.01,
        "repository": None,
        "metadata": {
            "custom_llm_provider": "anthropic",
            "passthrough_route_family": "anthropic_messages",
        },
    }

    async def fake_fetch(_query: str, *_args: Any) -> List[Dict[str, Any]]:
        return [row]

    mock_connection = MagicMock()
    mock_connection.fetch = AsyncMock(side_effect=fake_fetch)
    mock_connection.execute = AsyncMock()

    mock_pool = MagicMock()
    mock_acquire = MagicMock()
    mock_acquire.__aenter__ = AsyncMock(return_value=mock_connection)
    mock_acquire.__aexit__ = AsyncMock(return_value=None)
    mock_pool.acquire.return_value = mock_acquire

    args = argparse.Namespace(
        repair_gemini_control_plane=None,
        repair_costs=False,
        repair_tenant_ids=False,
        repair_anthropic_context_window=True,
        request_id=None,
        trace_id=None,
        session_id=None,
        provider=None,
        model=None,
        from_start_time=None,
        to_start_time=None,
        limit=None,
        batch_size=50,
        apply=False,
    )

    with patch.object(
        backfill_session_history,
        "_get_session_history_pool",
        AsyncMock(return_value=mock_pool),
    ), patch.object(
        backfill_session_history,
        "_ensure_session_history_schema_with_pool",
        AsyncMock(),
    ):
        result = await backfill_session_history._run_session_history_repair(args)

    assert result["stats"]["scanned_rows"] == 1
    assert result["stats"]["rows_with_updates"] == 1
    assert result["stats"]["anthropic_context_window_updates"] == 1
    assert result["stats"]["tenant_updates"] == 0
    assert result["stats"]["cost_updates"] == 0
    mock_connection.execute.assert_not_called()
