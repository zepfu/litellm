from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from litellm.integrations.aawm_session_history.cohere_accepted_calls import (
    CohereAcceptedCallState,
)
from litellm.proxy.pass_through_endpoints.success_handler import (
    PassThroughEndpointLogging,
)

_COHERE_URL = "https://api.cohere.com/v2/chat"
_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_ROUTE_FAMILY = "codex_cohere_chat_completions_adapter"
_OTHER_ROUTE_FAMILY = "codex_openai_chat_completions_adapter"
_END_TIME = datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc)


def _state(
    *,
    counted: bool = True,
    monthly_used: int = 1,
    monthly_remaining: int = 999,
    monthly_limit: int = 1000,
    rpm_used: int = 1,
    rpm_remaining: int | None = 19,
    rpm_limit: int | None = 20,
) -> CohereAcceptedCallState:
    return CohereAcceptedCallState(
        counted=counted,
        monthly_used=monthly_used,
        monthly_remaining=monthly_remaining,
        monthly_limit=monthly_limit,
        rpm_used=rpm_used,
        rpm_remaining=rpm_remaining,
        rpm_limit=rpm_limit,
        month_start=datetime(2026, 8, 1, tzinfo=timezone.utc),
        month_end=datetime(2026, 9, 1, tzinfo=timezone.utc),
    )


def _logging_obj(call_id: str) -> MagicMock:
    logging_obj = MagicMock()
    logging_obj.call_type = "pass_through_endpoint"
    logging_obj.model_call_details = {"litellm_call_id": call_id}
    logging_obj.dynamic_success_callbacks = []
    logging_obj.dynamic_async_success_callbacks = []
    logging_obj.get_combined_callback_list.return_value = []
    return logging_obj


async def _terminal_success(
    *,
    call_id: str = "call-cohere-1",
    url: str = _COHERE_URL,
    provider: str = "cohere",
    route_family: str = _ROUTE_FAMILY,
    status_code: int = 200,
    result: str = "complete",
    request_body: dict | None = None,
    recorder: AsyncMock,
) -> dict:
    handler = PassThroughEndpointLogging()
    callback_kwargs = {
        "litellm_call_id": call_id,
        "passthrough_route_family": route_family,
        "litellm_params": {"metadata": {"session_id": "session-1", "trace_id": "trace-1"}},
    }
    logging_obj = _logging_obj(call_id)
    response = httpx.Response(
        status_code,
        content=b"{}",
        request=httpx.Request("POST", url),
    )
    with patch(
        "litellm.proxy.pass_through_endpoints.success_handler.record_aawm_route_rollup_turn"
    ), patch.object(
        handler,
        "normalize_llm_passthrough_logging_payload",
        return_value={
            "standard_logging_response_object": {},
            "kwargs": callback_kwargs,
        },
    ), patch(
        "litellm.integrations.aawm_session_history.cohere_accepted_calls.record_cohere_accepted_call",
        recorder,
    ):
        await handler.pass_through_async_success_handler(
            httpx_response=response,
            response_body={},
            logging_obj=logging_obj,
            url_route=url,
            result=result,
            start_time=_END_TIME,
            end_time=_END_TIME,
            cache_hit=False,
            request_body=request_body or {"model": "cohere/command-a"},
            passthrough_logging_payload={},
            custom_llm_provider=provider,
            **callback_kwargs,
        )
    return callback_kwargs


@pytest.mark.asyncio
async def test_direct_non_stream_success_counts_once_and_publishes_normalized_observations():
    recorder = AsyncMock(return_value=_state())

    callback_kwargs = await _terminal_success(recorder=recorder)

    recorder.assert_awaited_once_with(
        litellm_call_id="call-cohere-1",
        accepted_at=_END_TIME,
        model="cohere/command-a",
        session_id="session-1",
        trace_id="trace-1",
        source=_ROUTE_FAMILY,
    )
    observations = callback_kwargs["rate_limit_observations"]
    assert len(observations) == 2
    for observation in observations:
        assert set(observation) == {
            "provider",
            "model",
            "quota_key",
            "quota_type",
            "limit_scope",
            "quota_period",
            "window_minutes",
            "remaining_pct",
            "observed_at",
            "expected_reset_at",
            "status",
            "exhausted",
            "source",
        }
        assert observation["provider"] == "cohere"
        assert observation["model"] == "cohere/command-a"
        assert observation["source"] == "locally_counted"
        assert observation["exhausted"] is False
    assert {observation["quota_type"] for observation in observations} == {
        "monthly",
        "rpm",
    }


@pytest.mark.asyncio
async def test_stream_partial_and_chunk_do_not_count_but_terminal_complete_counts_once():
    recorder = AsyncMock(return_value=_state())

    await _terminal_success(result="partial", recorder=recorder)
    await _terminal_success(result="chunk", recorder=recorder)
    recorder.assert_not_awaited()

    await _terminal_success(result="complete", recorder=recorder)

    recorder.assert_awaited_once()


@pytest.mark.asyncio
async def test_nonterminal_result_does_not_count():
    recorder = AsyncMock(return_value=_state())

    await _terminal_success(result="partial", recorder=recorder)

    recorder.assert_not_awaited()


@pytest.mark.asyncio
async def test_non_terminal_or_failed_responses_do_not_count():
    for status_code in (206, 400, 422, 500, 504):
        recorder = AsyncMock(return_value=_state())
        await _terminal_success(status_code=status_code, recorder=recorder)
        recorder.assert_not_awaited()


@pytest.mark.asyncio
async def test_replay_reuses_the_same_stable_call_id_for_persistence_deduplication():
    recorder = AsyncMock(side_effect=[_state(counted=True), _state(counted=False)])

    await _terminal_success(recorder=recorder)
    await _terminal_success(recorder=recorder)

    assert [call.kwargs["litellm_call_id"] for call in recorder.await_args_list] == [
        "call-cohere-1",
        "call-cohere-1",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("url", "provider", "route_family"),
    [
        (_OPENROUTER_URL, "openrouter", _ROUTE_FAMILY),
        (_COHERE_URL, "cohere", _OTHER_ROUTE_FAMILY),
    ],
)
async def test_non_direct_route_is_excluded(url: str, provider: str, route_family: str):
    recorder = AsyncMock(return_value=_state())

    await _terminal_success(
        url=url,
        provider=provider,
        route_family=route_family,
        recorder=recorder,
    )

    recorder.assert_not_awaited()


@pytest.mark.asyncio
async def test_cohere_sibling_path_does_not_persist():
    recorder = AsyncMock(return_value=_state())

    await _terminal_success(
        url="https://api.cohere.com/v2/chat/sibling",
        recorder=recorder,
    )

    recorder.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_call_id_fails_closed_and_logs_no_secret(caplog: pytest.LogCaptureFixture):
    recorder = AsyncMock(return_value=_state())

    await _terminal_success(call_id="  ", recorder=recorder)

    recorder.assert_not_awaited()
    assert "call-cohere-1" not in caplog.text
    assert "secret" not in caplog.text.lower()


@pytest.mark.asyncio
async def test_observations_use_state_limits_and_remaining_values():
    recorder = AsyncMock(
        return_value=_state(
            monthly_used=3,
            monthly_remaining=1197,
            monthly_limit=1200,
            rpm_used=3,
            rpm_remaining=4,
            rpm_limit=7,
        )
    )

    callback_kwargs = await _terminal_success(recorder=recorder)

    observations = {
        observation["quota_type"]: observation
        for observation in callback_kwargs["rate_limit_observations"]
    }
    assert observations["monthly"]["remaining_pct"] == pytest.approx(99.75)
    assert observations["monthly"]["exhausted"] is False
    assert observations["rpm"]["remaining_pct"] == pytest.approx(57.142857)
    assert observations["rpm"]["exhausted"] is False


@pytest.mark.asyncio
async def test_unknown_rpm_state_remains_unknown():
    recorder = AsyncMock(
        return_value=_state(rpm_remaining=None, rpm_limit=None)
    )

    callback_kwargs = await _terminal_success(recorder=recorder)

    observations = {
        observation["quota_type"]: observation
        for observation in callback_kwargs["rate_limit_observations"]
    }
    assert observations["rpm"]["remaining_pct"] is None
    assert observations["rpm"]["exhausted"] is None
    assert observations["rpm"]["status"] == "unknown"


@pytest.mark.asyncio
async def test_persistence_failure_redacts_exception_details(caplog: pytest.LogCaptureFixture):
    secret = "cohere-secret-value"
    recorder = AsyncMock(side_effect=RuntimeError(secret))

    await _terminal_success(recorder=recorder)

    assert secret not in caplog.text
