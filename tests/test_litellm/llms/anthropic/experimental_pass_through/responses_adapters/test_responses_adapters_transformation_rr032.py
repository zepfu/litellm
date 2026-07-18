"""RR-032: responses_adapters transformation residuals.

#1 High/operational — prompt_cache_key stability / correct bound path
#2 Medium/maintainability — this module is request/response transform only;
   SSE emitter consolidation is owned by streaming_iterator trees, not here.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Any, Dict

from litellm.llms.anthropic.experimental_pass_through.adapters import (
    streaming_iterator as shared_sse_emitter,
)
from litellm.llms.anthropic.experimental_pass_through.adapters.observability import (
    derive_prompt_cache_key,
)
from litellm.llms.anthropic.experimental_pass_through.responses_adapters import (
    streaming_iterator as responses_stream,
)
from litellm.llms.anthropic.experimental_pass_through.responses_adapters import (
    transformation as transformation_mod,
)
from litellm.llms.anthropic.experimental_pass_through.responses_adapters.transformation import (
    LiteLLMAnthropicToResponsesAPIAdapter,
    _bound_prompt_cache_key,
    _resolve_responses_prompt_cache_key,
)
from litellm.types.llms.anthropic import AnthropicMessagesRequest


def _req(**overrides: Any) -> AnthropicMessagesRequest:
    base: Dict[str, Any] = {
        "model": "openai.gpt-5.1-codex",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 1024,
    }
    base.update(overrides)
    return AnthropicMessagesRequest(**base)


_ADAPTER = LiteLLMAnthropicToResponsesAPIAdapter()


def test_prompt_cache_key_stable_across_user_turns_with_system_tools() -> None:
    """RR-032 #1: moving message cache_control must not churn the key."""
    system = [
        {
            "type": "text",
            "text": "You are helpful.",
            "cache_control": {"type": "ephemeral"},
        }
    ]
    tools = [
        {
            "name": "Bash",
            "description": "run shell",
            "input_schema": {"type": "object", "properties": {}},
            "cache_control": {"type": "ephemeral"},
        }
    ]

    def body(user_text: str) -> AnthropicMessagesRequest:
        return _req(
            system=system,
            tools=tools,
            messages=[
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
        )

    k1 = _ADAPTER.translate_request(body("first turn"), custom_llm_provider="openai")
    k2 = _ADAPTER.translate_request(
        body("second turn completely different"), custom_llm_provider="openai"
    )
    assert k1["prompt_cache_key"] == k2["prompt_cache_key"]
    assert k1["prompt_cache_key"].startswith("anthropic-cache-")
    assert len(k1["prompt_cache_key"]) <= 64
    assert k1["litellm_metadata"]["openai_prompt_cache_key_present"] is True
    assert k1["litellm_metadata"]["anthropic_adapter_cache_control_present"] is True
    # Shared helper is the source of the stable digest (system/tools roots only).
    shared = derive_prompt_cache_key(
        {
            "system": system,
            "tools": tools,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "ignored-for-key",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ],
        }
    )
    assert k1["prompt_cache_key"] == shared


def test_message_only_cache_control_omits_volatile_prompt_cache_key() -> None:
    """Prefer omit over per-turn key when system/tools surface is absent."""
    req = _req(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "cache me",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ]
    )
    kwargs = _ADAPTER.translate_request(req, custom_llm_provider="openai")
    assert "prompt_cache_key" not in kwargs
    meta = kwargs["litellm_metadata"]
    assert meta["anthropic_adapter_cache_control_present"] is True
    assert meta["openai_prompt_cache_key_present"] is False
    assert (
        meta["openai_prompt_cache_key_omitted_reason"]
        == "no_stable_system_or_tools_cache_surface"
    )


def test_explicit_prompt_cache_key_is_bounded_not_rederived_as_system_tools() -> None:
    """Long explicit keys must be hashed, not dropped via system/tools roots."""
    long_key = "k" * 120
    req = _req(
        system=[
            {"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}
        ],
        messages=[{"role": "user", "content": "hi"}],
    )
    req_dict = dict(req)
    req_dict["prompt_cache_key"] = long_key
    resolved = _resolve_responses_prompt_cache_key(req_dict)
    assert resolved is not None
    assert len(resolved) <= 64
    expected = "anthropic-cache-" + hashlib.sha256(long_key.encode()).hexdigest()[:40]
    assert resolved == expected[:64]
    assert resolved == _bound_prompt_cache_key(long_key)
    # Must NOT equal re-deriving via system/tools only (old buggy path).
    buggy = derive_prompt_cache_key({"prompt_cache_key": long_key})
    assert buggy is None


def test_explicit_prompt_cache_key_translates_without_cache_control() -> None:
    long_key = "explicit-" + ("x" * 100)
    request = dict(_req())
    request["prompt_cache_key"] = long_key

    translated = _ADAPTER.translate_request(
        request,
        custom_llm_provider="openai",
    )

    assert translated["prompt_cache_key"] == _bound_prompt_cache_key(long_key)
    assert translated["litellm_metadata"]["openai_prompt_cache_key_present"] is True
    assert (
        translated["litellm_metadata"]["anthropic_adapter_cache_control_present"]
        is False
    )


def test_bound_prompt_cache_key_preserves_short_keys() -> None:
    assert _bound_prompt_cache_key("short-key") == "short-key"


def test_shared_tool_use_id_sanitizer_is_used() -> None:
    """Reuse shared adapters sanitizers rather than local copies."""
    block = _ADAPTER._responses_function_call_to_anthropic_tool_use(
        upstream_tool_name="Bash",
        arguments='{"command":"ls"}',
        call_id="toolu_01ABC",
        fallback_id=None,
        use_codex_native_tools=False,
    )
    assert block["type"] == "tool_use"
    assert block["id"] == "toolu_01ABC"


def test_module_is_not_sse_emitter_surface() -> None:
    """RR-032 #2 file responsibility: no Anthropic SSE event emission here."""
    source = Path(transformation_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    # No SSE wrapper / streaming-emitter APIs in this transform module.
    assert "async_anthropic_sse_wrapper" not in names
    assert "anthropic_sse_wrapper" not in names
    assert "emit_content_block_delta" not in names
    assert "emit_message_start" not in names
    assert "encode_anthropic_sse_chunk" not in names
    assert "content_block_delta" not in names
    # Ownership docs + shared helper imports remain.
    assert "streaming_iterator" in source
    assert "not an SSE emitter" in source
    assert "derive_prompt_cache_key" in source
    assert "sanitize_anthropic_tool_use_input" in source
    assert "sanitize_anthropic_tool_use_id" in source


def test_sse_emitter_shared_by_streaming_iterators_not_this_module() -> None:
    """RR-032 #2: emitter consolidation lives on streaming trees; transform reuses sanitizers only."""
    # Shared helpers exported from Chat Completions streaming_iterator.
    for name in (
        "emit_content_block_delta",
        "emit_content_block_start",
        "emit_content_block_stop",
        "emit_message_start",
        "emit_message_delta",
        "emit_message_stop",
        "encode_anthropic_sse_chunk",
    ):
        assert hasattr(shared_sse_emitter, name)
        assert getattr(responses_stream, name) is getattr(shared_sse_emitter, name)

    # Transformation must not re-export or reimplement those helpers.
    for name in (
        "emit_content_block_delta",
        "emit_message_start",
        "encode_anthropic_sse_chunk",
    ):
        assert not hasattr(transformation_mod, name)

    # Responses streaming_iterator imports emitter from adapters; transformation does not.
    stream_src = Path(responses_stream.__file__).read_text(encoding="utf-8")
    assert "emit_content_block_delta" in stream_src
    assert "adapters.streaming_iterator" in stream_src
    transform_src = Path(transformation_mod.__file__).read_text(encoding="utf-8")
    assert "emit_content_block_delta" not in transform_src
    assert "encode_anthropic_sse_chunk" not in transform_src
