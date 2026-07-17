"""B5: session_history retry exhaustion must not loop forever; mid-batch sentinel drains."""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import litellm.integrations.aawm_agent_identity as identity


def test_flush_with_retry_returns_after_max_retries_when_exhaustion_cannot_spool() -> None:
    """When flush always fails and exhaustion returns False, retry loop must terminate."""
    batch: List[Dict[str, Any]] = [{"litellm_call_id": "call-b5-1"}]
    flush_calls = {"n": 0}

    def _always_fail_flush(*_args: Any, **kwargs: Any) -> bool:
        flush_calls["n"] += 1
        cb = kwargs.get("failure_callback")
        if cb is not None:
            cb(RuntimeError("db down"))
        return False

    with patch.object(
        identity, "_flush_session_history_batch", side_effect=_always_fail_flush
    ), patch.object(
        identity, "_get_session_history_failed_flush_max_retries", return_value=2
    ), patch.object(
        identity, "_get_session_history_failed_flush_retry_seconds", return_value=0.0
    ), patch.object(
        identity, "_handle_session_history_retry_exhaustion", return_value=False
    ), patch.object(
        identity,
        "_prepare_session_history_retry_after_failure",
        return_value=(False, None, 0, None),
    ), patch.object(identity, "_log_session_history_retry"), patch.object(
        identity, "time"
    ) as mock_time:
        mock_time.sleep = MagicMock()
        identity._flush_session_history_batch_with_retry(
            batch, retry_message="unit-test flush"
        )

    # Finite attempts only — no infinite loop when exhaustion cannot spool.
    assert flush_calls["n"] >= 3
    assert flush_calls["n"] <= 10
    mock_time.sleep.assert_called()


def test_flush_with_retry_stops_when_exhaustion_spools_successfully() -> None:
    batch = [{"litellm_call_id": "call-b5-2"}]
    flush_calls = {"n": 0}

    def _always_fail(*_a: Any, **kwargs: Any) -> bool:
        flush_calls["n"] += 1
        cb = kwargs.get("failure_callback")
        if cb is not None:
            cb(RuntimeError("transient"))
        return False

    with patch.object(
        identity, "_flush_session_history_batch", side_effect=_always_fail
    ), patch.object(
        identity, "_get_session_history_failed_flush_max_retries", return_value=0
    ), patch.object(
        identity, "_get_session_history_failed_flush_retry_seconds", return_value=0.0
    ), patch.object(
        identity, "_handle_session_history_retry_exhaustion", return_value=True
    ), patch.object(identity, "_log_session_history_retry"), patch.object(
        identity, "time"
    ) as mock_time:
        mock_time.sleep = MagicMock()
        identity._flush_session_history_batch_with_retry(batch)

    assert flush_calls["n"] == 1


def test_worker_main_drains_queue_when_sentinel_arrives_mid_batch() -> None:
    """Mid-batch None sentinel flushes partial batch and drains leftovers."""
    while True:
        try:
            identity._aawm_session_history_queue.get_nowait()
        except Exception:
            break

    flushed: List[List[Dict[str, Any]]] = []

    def _capture_flush(batch: List[Dict[str, Any]], **_kwargs: Any) -> None:
        flushed.append(list(batch))

    with patch.object(
        identity, "_get_session_history_flush_interval_seconds", return_value=0.05
    ), patch.object(
        identity, "_get_session_history_batch_size", return_value=10
    ), patch.object(
        identity, "_flush_session_history_batch_with_retry", side_effect=_capture_flush
    ), patch.object(
        identity,
        "_close_aawm_session_history_pools_for_current_loop",
        new=AsyncMock(),
    ):
        identity._aawm_session_history_queue.put({"id": "a"})
        identity._aawm_session_history_queue.put({"id": "b"})
        identity._aawm_session_history_queue.put(None)  # mid-batch sentinel
        identity._aawm_session_history_queue.put({"id": "c"})  # behind sentinel

        identity._session_history_worker_main()

    flat = [r.get("id") for batch in flushed for r in batch]
    assert "a" in flat
    assert "b" in flat
    assert "c" in flat
