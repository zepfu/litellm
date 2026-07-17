"""Focused RR-058 regressions for non-streaming pass-through success_handler."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy.aawm_route_logging import (
    clear_aawm_route_rollups,
    flush_aawm_route_rollups,
)
from litellm.proxy.pass_through_endpoints.success_handler import (
    PassThroughEndpointLogging,
)


def _base_logging_obj(
    *,
    sync_callbacks: Optional[List[Any]] = None,
    async_callbacks: Optional[List[Any]] = None,
    model_call_details: Optional[dict] = None,
    call_type: str = "pass_through_endpoint",
) -> MagicMock:
    logging_obj = MagicMock()
    logging_obj.call_type = call_type
    logging_obj.dynamic_success_callbacks = None
    logging_obj.dynamic_async_success_callbacks = None
    logging_obj.model_call_details = model_call_details if model_call_details is not None else {}
    logging_obj.get_combined_callback_list.side_effect = [
        list(sync_callbacks or []),
        list(async_callbacks or []),
    ]
    return logging_obj


@pytest.mark.asyncio
async def test_async_callback_exception_isolates_later_callbacks() -> None:
    handler = PassThroughEndpointLogging()
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=1)
    events: List[str] = []

    class RaisingAsyncCallback:
        async def async_logging_hook(self, kwargs, result, call_type):
            events.append("raising_hook")
            raise RuntimeError("boom-async-hook")

        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            events.append("raising_success")

    class HealthyAsyncCallback:
        def __init__(self) -> None:
            self.success_kwargs: Optional[dict] = None
            self.calls = 0

        async def async_logging_hook(self, kwargs, result, call_type):
            events.append("healthy_hook")
            next_kwargs = dict(kwargs)
            next_kwargs["healthy_hook"] = True
            return next_kwargs, result

        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            events.append("healthy_success")
            self.calls += 1
            self.success_kwargs = dict(kwargs)

    raising = RaisingAsyncCallback()
    healthy = HealthyAsyncCallback()
    logging_obj = _base_logging_obj(async_callbacks=[raising, healthy])
    logging_obj._build_standard_logging_payload = MagicMock(
        return_value={
            "id": "call-1",
            "response_cost": 0.01,
            "messages": [{"role": "user", "content": "hi"}],
            "response": {"text": "ok"},
            "metadata": {},
        }
    )

    with patch(
        "litellm.proxy.pass_through_endpoints.success_handler.thread_pool_executor.submit"
    ):
        await handler._handle_logging(
            logging_obj=logging_obj,
            standard_logging_response_object={"ok": True},
            result="",
            start_time=start_time,
            end_time=end_time,
            cache_hit=False,
            response_cost=0.01,
            model="claude-sonnet-4",
            litellm_params={"metadata": {}},
        )

    # Hook failure is isolated; the same callback may still attempt success logging,
    # and later callbacks must continue.
    assert events == [
        "raising_hook",
        "raising_success",
        "healthy_hook",
        "healthy_success",
    ]
    assert healthy.calls == 1
    assert healthy.success_kwargs is not None
    assert healthy.success_kwargs.get("healthy_hook") is True
    assert isinstance(healthy.success_kwargs.get("standard_logging_object"), dict)


@pytest.mark.asyncio
async def test_async_log_success_event_exception_isolates_later_callbacks() -> None:
    handler = PassThroughEndpointLogging()
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=1)
    success_calls: List[str] = []

    class RaisingSuccessCallback:
        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            success_calls.append("raising")
            raise ValueError("boom-success")

    class HealthySuccessCallback:
        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            success_calls.append("healthy")

    logging_obj = _base_logging_obj(
        async_callbacks=[RaisingSuccessCallback(), HealthySuccessCallback()]
    )
    logging_obj._build_standard_logging_payload = MagicMock(
        return_value={"id": "call-2", "response_cost": 0.0, "metadata": {}}
    )

    await handler._handle_logging(
        logging_obj=logging_obj,
        standard_logging_response_object={"ok": True},
        result="",
        start_time=start_time,
        end_time=end_time,
        cache_hit=False,
    )

    assert success_calls == ["raising", "healthy"]


@pytest.mark.asyncio
async def test_pass_through_async_success_handler_continues_route_rollup_after_async_failure(
    monkeypatch,
) -> None:
    clear_aawm_route_rollups()
    monkeypatch.setenv("AAWM_ROUTE_ROLLUP_INTERVAL_SECONDS", "60")

    class RaisingAsyncCallback:
        async def async_logging_hook(self, kwargs, result, call_type):
            raise RuntimeError("rollup-must-continue")

        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            return None

    kwargs = {
        "litellm_params": {
            "metadata": {
                "aawm_route_rollup_context": {
                    "group_header_label": "litellm@Codex[rr058]",
                    "incoming_endpoint": "/openai_passthrough/responses",
                    "outgoing_target": "api.openai.com/v1/responses",
                    "model_label": "gpt-test(aawm)",
                }
            }
        }
    }
    handler = PassThroughEndpointLogging()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = '{"success": true}'
    mock_response.headers = {}
    mock_response.request = MagicMock(
        method="POST", url="https://api.openai.com/v1/responses"
    )

    logging_obj = _base_logging_obj(async_callbacks=[RaisingAsyncCallback()])
    logging_obj._build_standard_logging_payload = MagicMock(
        return_value={"id": "rollup-call", "response_cost": 0.0, "metadata": {}}
    )

    with patch.object(
        handler,
        "normalize_llm_passthrough_logging_payload",
        return_value={
            "standard_logging_response_object": {"success": True},
            "kwargs": kwargs,
        },
    ), patch(
        "litellm.proxy.pass_through_endpoints.success_handler.thread_pool_executor.submit"
    ):
        await handler.pass_through_async_success_handler(
            httpx_response=mock_response,
            response_body={"success": True},
            logging_obj=logging_obj,
            url_route="https://api.openai.com/v1/responses",
            result="",
            start_time=datetime.now(),
            end_time=datetime.now(),
            cache_hit=False,
            request_body={"model": "gpt-test"},
            passthrough_logging_payload=MagicMock(),
            custom_llm_provider="openai",
            **kwargs,
        )

    flushed = flush_aawm_route_rollups(force=True)
    rendered = "\n".join(flushed)
    assert len(flushed) == 2
    assert "litellm@Codex[rr058] /openai_passthrough/responses" in rendered
    assert (
        kwargs["litellm_params"]["metadata"]["aawm_route_rollup_turn_recorded"] is True
    )
    clear_aawm_route_rollups()


@pytest.mark.asyncio
async def test_handle_logging_builds_standard_logging_object_once() -> None:
    handler = PassThroughEndpointLogging()
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=1)
    captured: Dict[str, Any] = {}

    class CaptureAsyncCallback:
        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            captured["kwargs"] = dict(kwargs)
            captured["response_obj"] = response_obj

    payload = {
        "id": "std-1",
        "response_cost": 0.123,
        "messages": [{"role": "user", "content": "secret-prompt"}],
        "response": {"choices": [{"message": {"content": "secret-answer"}}]},
        "metadata": {"usage_object": {"prompt_tokens": 3, "completion_tokens": 5}},
        "model": "claude-sonnet-4",
        "custom_llm_provider": "anthropic",
    }

    logging_obj = _base_logging_obj(
        async_callbacks=[CaptureAsyncCallback()],
        model_call_details={
            "model": "claude-sonnet-4",
            "custom_llm_provider": "anthropic",
            "response_cost": 0.123,
            "litellm_params": {"metadata": {}},
        },
    )
    logging_obj._build_standard_logging_payload = MagicMock(return_value=payload)

    await handler._handle_logging(
        logging_obj=logging_obj,
        standard_logging_response_object={"ok": True, "id": "resp"},
        result="",
        start_time=start_time,
        end_time=end_time,
        cache_hit=False,
        response_cost=0.123,
        model="claude-sonnet-4",
        custom_llm_provider="anthropic",
        litellm_params={"metadata": {}},
    )

    logging_obj._build_standard_logging_payload.assert_called_once()
    assert isinstance(captured.get("kwargs"), dict)
    standard_logging_object = captured["kwargs"].get("standard_logging_object")
    assert standard_logging_object is payload
    assert standard_logging_object["response_cost"] == 0.123
    assert standard_logging_object["metadata"]["usage_object"]["prompt_tokens"] == 3
    assert logging_obj.model_call_details["standard_logging_object"] is payload
    # Only one callback invocation for the single registered async success logger.
    assert logging_obj.get_combined_callback_list.call_count == 2


@pytest.mark.asyncio
async def test_handle_logging_reuses_existing_standard_logging_object_without_rebuild() -> None:
    handler = PassThroughEndpointLogging()
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=1)
    existing = {
        "id": "already-built",
        "response_cost": 0.5,
        "metadata": {},
        "messages": [{"role": "user", "content": "x"}],
        "response": {"text": "y"},
    }
    captured: Dict[str, Any] = {}

    class CaptureAsyncCallback:
        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            captured["kwargs"] = dict(kwargs)

    logging_obj = _base_logging_obj(async_callbacks=[CaptureAsyncCallback()])
    logging_obj._build_standard_logging_payload = MagicMock(
        side_effect=AssertionError("must not rebuild existing standard_logging_object")
    )

    await handler._handle_logging(
        logging_obj=logging_obj,
        standard_logging_response_object={"ok": True},
        result="",
        start_time=start_time,
        end_time=end_time,
        cache_hit=False,
        standard_logging_object=existing,
        response_cost=0.5,
    )

    logging_obj._build_standard_logging_payload.assert_not_called()
    assert captured["kwargs"]["standard_logging_object"] is existing


@pytest.mark.asyncio
async def test_handle_logging_does_not_invoke_native_async_success_handler() -> None:
    handler = PassThroughEndpointLogging()
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=1)
    call_count = {"n": 0}

    class CountingCallback:
        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            call_count["n"] += 1

    logging_obj = _base_logging_obj(async_callbacks=[CountingCallback()])
    logging_obj.async_success_handler = AsyncMock()
    logging_obj.success_handler = MagicMock()
    logging_obj._build_standard_logging_payload = MagicMock(
        return_value={"id": "no-double", "response_cost": 0.0, "metadata": {}}
    )

    await handler._handle_logging(
        logging_obj=logging_obj,
        standard_logging_response_object={"ok": True},
        result="",
        start_time=start_time,
        end_time=end_time,
        cache_hit=False,
    )

    assert call_count["n"] == 1
    logging_obj.async_success_handler.assert_not_called()
    logging_obj.success_handler.assert_not_called()


@pytest.mark.asyncio
async def test_handle_logging_preserves_redaction_before_callbacks(monkeypatch) -> None:
    """RR-058 #1 already landed; keep redaction ordering intact with new payload work."""
    handler = PassThroughEndpointLogging()
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=1)
    seen_results: List[Any] = []

    class CaptureAsyncCallback:
        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            seen_results.append(response_obj)

    logging_obj = _base_logging_obj(
        async_callbacks=[CaptureAsyncCallback()],
        model_call_details={
            "messages": [{"role": "user", "content": "secret"}],
            "litellm_params": {"metadata": {}},
            "standard_callback_dynamic_params": {},
        },
    )
    logging_obj._build_standard_logging_payload = MagicMock(
        return_value={
            "id": "redact",
            "messages": [{"role": "user", "content": "secret"}],
            "response": {"text": "secret-out"},
            "metadata": {},
        }
    )

    monkeypatch.setattr(
        "litellm.proxy.pass_through_endpoints.success_handler.redact_message_input_output_from_logging",
        lambda model_call_details, result: {"text": "redacted-by-litellm"},
    )

    await handler._handle_logging(
        logging_obj=logging_obj,
        standard_logging_response_object={"text": "secret-out"},
        result="",
        start_time=start_time,
        end_time=end_time,
        cache_hit=False,
    )

    assert seen_results == [{"text": "redacted-by-litellm"}]
    # Payload build still happens once from the pre-callback path.
    logging_obj._build_standard_logging_payload.assert_called_once()
