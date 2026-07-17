"""RR-026: prompt_cache_key must be stable across user-turn changes."""

from __future__ import annotations

from litellm.llms.anthropic.experimental_pass_through.adapters.observability import (
    derive_prompt_cache_key,
)


def _body(user_text: str) -> dict:
    return {
        "system": [
            {
                "type": "text",
                "text": "You are helpful.",
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "tools": [
            {
                "name": "Bash",
                "description": "run shell",
                "input_schema": {"type": "object"},
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_text,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ],
    }


def test_prompt_cache_key_stable_when_user_message_changes() -> None:
    k1 = derive_prompt_cache_key(_body("first turn"))
    k2 = derive_prompt_cache_key(_body("second turn completely different"))
    assert k1 is not None and k2 is not None
    assert k1 == k2


def test_prompt_cache_key_changes_when_system_changes() -> None:
    a = _body("same user")
    b = _body("same user")
    b["system"][0]["text"] = "Different system prompt"
    assert derive_prompt_cache_key(a) != derive_prompt_cache_key(b)


def test_prompt_cache_key_none_for_empty_payload() -> None:
    assert derive_prompt_cache_key({}) is None
    assert derive_prompt_cache_key({"messages": []}) is None
