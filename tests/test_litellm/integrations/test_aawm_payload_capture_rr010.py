"""RR-010: AAWM payload capture async hook offloads dump work."""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import AsyncMock, patch

import pytest

from litellm.integrations.aawm_payload_capture import AawmPayloadCapture


@pytest.mark.asyncio
async def test_async_logging_hook_offloads_dump_capture_once() -> None:
    logger = AawmPayloadCapture()
    kwargs: Dict[str, Any] = {"model": "test-model", "messages": []}
    result = {"ok": True}

    with (
        patch(
            "litellm.integrations.aawm_payload_capture._dump_capture"
        ) as dump_mock,
        patch(
            "litellm.integrations.aawm_payload_capture.asyncio.to_thread",
            new_callable=AsyncMock,
        ) as to_thread_mock,
    ):
        to_thread_mock.return_value = None
        out_kwargs, out_result = await logger.async_logging_hook(
            kwargs, result, "completion"
        )

    to_thread_mock.assert_awaited_once_with(dump_mock, kwargs)
    dump_mock.assert_not_called()
    assert out_kwargs is kwargs
    assert out_result is result


def test_sync_logging_hook_calls_dump_capture_directly() -> None:
    logger = AawmPayloadCapture()
    kwargs: Dict[str, Any] = {"model": "test-model", "messages": []}
    result = {"ok": True}

    with patch(
        "litellm.integrations.aawm_payload_capture._dump_capture"
    ) as dump_mock:
        out_kwargs, out_result = logger.logging_hook(
            kwargs, result, "completion"
        )

    dump_mock.assert_called_once_with(kwargs)
    assert out_kwargs is kwargs
    assert out_result is result
