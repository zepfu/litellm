"""Focused closure evidence for RR-030."""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from litellm.llms.anthropic.experimental_pass_through.adapters.observability import (
    derive_prompt_cache_key,
)
from litellm.llms.anthropic.experimental_pass_through.responses_adapters.handler import (
    LiteLLMMessagesToResponsesAPIHandler,
    _build_responses_kwargs,
)


def _cache_body(turn_text: str) -> Dict[str, Any]:
    return {
        "system": [
            {
                "type": "text",
                "text": "stable system",
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "tools": [
            {
                "name": "bash",
                "description": "run shell",
                "input_schema": {"type": "object", "properties": {}},
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": turn_text,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ],
    }


def test_handler_uses_stable_shared_prompt_cache_key() -> None:
    first = _cache_body("first turn")
    second = _cache_body("different second turn")

    def build(body: Dict[str, Any]) -> Dict[str, Any]:
        return _build_responses_kwargs(
            max_tokens=128,
            messages=body["messages"],
            model="gpt-4o",
            system=body["system"],
            tools=body["tools"],
        )

    first_kwargs = build(first)
    second_kwargs = build(second)
    assert first_kwargs["prompt_cache_key"] == second_kwargs["prompt_cache_key"]
    assert first_kwargs["prompt_cache_key"] == derive_prompt_cache_key(first)


def test_handler_does_not_reimplement_cache_key_or_sse_encoding() -> None:
    import litellm.llms.anthropic.experimental_pass_through.responses_adapters.handler as handler

    source = open(handler.__file__, encoding="utf-8").read()
    assert "hashlib" not in source
    assert "sha256" not in source
    assert "event: " not in source


@pytest.mark.asyncio
async def test_async_stream_delegates_to_shared_responses_wrapper() -> None:
    sentinel = MagicMock()

    async def fake_aresponses(**kwargs: Any) -> object:
        assert kwargs["stream"] is True
        return object()

    with patch("litellm.aresponses", new=fake_aresponses), patch(
        "litellm.llms.anthropic.experimental_pass_through.responses_adapters.handler.AnthropicResponsesStreamWrapper"
    ) as wrapper_type:
        wrapper_type.return_value.async_anthropic_sse_wrapper.return_value = sentinel
        result = await LiteLLMMessagesToResponsesAPIHandler.async_anthropic_messages_handler(
            max_tokens=16,
            messages=[{"role": "user", "content": "hi"}],
            model="gpt-4o",
            stream=True,
        )

    assert result is sentinel
    wrapper_type.return_value.async_anthropic_sse_wrapper.assert_called_once_with()


def test_sync_stream_delegates_to_shared_responses_wrapper() -> None:
    sentinel = MagicMock()
    with patch("litellm.responses", return_value=object()), patch(
        "litellm.llms.anthropic.experimental_pass_through.responses_adapters.handler.AnthropicResponsesStreamWrapper"
    ) as wrapper_type:
        wrapper_type.return_value.async_anthropic_sse_wrapper.return_value = sentinel
        result = LiteLLMMessagesToResponsesAPIHandler.anthropic_messages_handler(
            max_tokens=16,
            messages=[{"role": "user", "content": "hi"}],
            model="gpt-4o",
            stream=True,
        )

    assert result is sentinel
    wrapper_type.return_value.async_anthropic_sse_wrapper.assert_called_once_with()
