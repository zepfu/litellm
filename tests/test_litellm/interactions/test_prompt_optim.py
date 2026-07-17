"""Regression tests for prompt_optim placeholder API logging safety."""

import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.abspath("../../.."))

from litellm._logging import verbose_logger
from litellm.interactions.prompt_optim import prompt_optim


def test_prompt_optim_returns_messages_unchanged():
    """Placeholder prompt_optim must preserve input messages."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "secret user prompt content"},
    ]

    result = asyncio.run(prompt_optim(model="gpt-4o-mini", messages=messages))

    assert result is messages
    assert result == messages


def test_prompt_optim_debug_log_omits_message_contents():
    """
    RR-016: debug logs must not emit full prompt/message contents.

    Uses a custom handler instead of caplog for pytest-xdist reliability.
    """
    secret = "TOP_SECRET_PROMPT_CONTENT_SHOULD_NEVER_APPEAR"

    class LogRecordHandler(logging.Handler):
        def __init__(self):
            super().__init__()
            self.messages = []

        def emit(self, record):
            self.messages.append(record.getMessage())

    handler = LogRecordHandler()
    handler.setLevel(logging.DEBUG)
    original_level = verbose_logger.level
    verbose_logger.setLevel(logging.DEBUG)
    verbose_logger.addHandler(handler)

    try:
        messages = [
            {"role": "system", "content": secret},
            {"role": "user", "content": secret},
        ]
        result = asyncio.run(
            prompt_optim(model="gpt-4o-mini", messages=messages, unused_kwarg=1)
        )
        assert result is messages

        combined = "\n".join(handler.messages)
        assert "litellm.prompt_optim" in combined
        assert "message_count=2" in combined
        assert "roles=" in combined
        assert "system" in combined
        assert "user" in combined
        assert secret not in combined
        assert "content=" not in combined
        assert "messages=[" not in combined
    finally:
        verbose_logger.removeHandler(handler)
        verbose_logger.setLevel(original_level)
