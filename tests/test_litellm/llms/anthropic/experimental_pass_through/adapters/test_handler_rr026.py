"""RR-026: adapters/handler.py ownership and orchestration invariants."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.llms.anthropic.experimental_pass_through.adapters import handler as handler_mod
from litellm.llms.anthropic.experimental_pass_through.adapters.handler import (
    LiteLLMMessagesToCompletionTransformationHandler as Handler,
)
from litellm.llms.anthropic.experimental_pass_through.adapters.observability import (
    derive_prompt_cache_key,
)


HANDLER_PATH = Path(handler_mod.__file__).resolve()
ADAPTERS_DIR = HANDLER_PATH.parent
RESPONSES_HANDLER = (
    ADAPTERS_DIR.parent / "responses_adapters" / "handler.py"
).resolve()


def test_rr026_handler_does_not_implement_prompt_cache_key_or_sse_emitter() -> None:
    """Issue #1/#2 file-responsibility: this handler is orchestration-only."""
    source = HANDLER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    # Executable body only — module docstring may name deferred owners.
    # Strip leading module docstring node text for code-only checks.
    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(getattr(tree.body[0], "value", None), ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        # Check executable definitions/calls via AST names.
        defined_names = {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
        called_names = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "derive_prompt_cache_key" not in defined_names
        assert "derive_prompt_cache_key" not in called_names
        assert "_collect_cache_control_material" not in defined_names
        assert "_collect_cache_control_material" not in called_names
        assert "content_block_delta" not in defined_names
    else:
        assert False, "expected module docstring"

    # Streaming is delegated to the adapter, not reimplemented here.
    assert "translate_completion_output_params_streaming" in source
    assert "ANTHROPIC_ADAPTER" in source
    # No Anthropic SSE event construction APIs in executable code paths.
    assert "content_block_start" not in source.split('"""', 2)[-1]
    assert "content_block_delta" not in source.split('"""', 2)[-1]


def test_rr026_prompt_cache_key_owned_by_observability_system_tools_only() -> None:
    """Issue #1 residual for this file: hashing lives in observability, stable roots."""
    body_turn_1 = {
        "system": [
            {
                "type": "text",
                "text": "stable system",
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "tools": [
            {
                "name": "Bash",
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
                        "text": "turn one",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ],
    }
    body_turn_2 = {
        **body_turn_1,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "turn two completely different",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ],
    }
    k1 = derive_prompt_cache_key(body_turn_1)
    k2 = derive_prompt_cache_key(body_turn_2)
    assert k1 is not None and k2 is not None
    assert k1 == k2


def test_rr026_parallel_handler_trees_are_distinct_orchestration_surfaces() -> None:
    """Issue #2 not_valid for this file: handlers are thin; SSE lives in iterators."""
    chat_source = HANDLER_PATH.read_text(encoding="utf-8")
    assert RESPONSES_HANDLER.is_file()
    responses_source = RESPONSES_HANDLER.read_text(encoding="utf-8")

    # Chat Completions path
    assert "litellm.completion" in chat_source or "litellm.acompletion" in chat_source
    assert "AnthropicAdapter" in chat_source

    # Responses path
    assert "litellm.responses" in responses_source or "litellm.aresponses" in responses_source
    assert "AnthropicResponsesStreamWrapper" in responses_source

    # Neither handler embeds Anthropic SSE event construction.
    for label, source in ("chat", chat_source), ("responses", responses_source):
        assert "content_block_start" not in source, label
        assert "content_block_delta" not in source, label
        assert "message_delta" not in source, label


def test_rr026_sync_and_async_share_transform_helper() -> None:
    """Local maintainability: sync/async must not re-fork translation policy."""
    source = HANDLER_PATH.read_text(encoding="utf-8")
    assert "def _transform_completion_response" in source
    # Both entrypoints call the shared transform path.
    assert source.count("_transform_completion_response(") >= 3


def test_rr026_prepare_completion_kwargs_sets_stream_options() -> None:
    kwargs, mapping = Handler._prepare_completion_kwargs(
        max_tokens=16,
        messages=[{"role": "user", "content": "hi"}],
        model="gpt-4o-mini",
        stream=True,
        extra_kwargs={"custom_llm_provider": "openai"},
    )
    assert kwargs["stream"] is True
    assert kwargs.get("stream_options") == {"include_usage": True}
    assert isinstance(mapping, dict)


def test_rr026_transform_streaming_delegates_to_adapter() -> None:
    fake_stream = object()
    with patch.object(
        handler_mod.ANTHROPIC_ADAPTER,
        "translate_completion_output_params_streaming",
        return_value=fake_stream,
    ) as stream_mock:
        out = Handler._transform_completion_response(
            completion_response=MagicMock(),
            model="gpt-4o-mini",
            stream=True,
            tool_name_mapping={},
        )
    stream_mock.assert_called_once()
    assert out is fake_stream


def test_rr026_transform_non_stream_delegates_to_adapter() -> None:
    fake_response = {"id": "msg_test", "type": "message", "role": "assistant"}
    with patch.object(
        handler_mod.ANTHROPIC_ADAPTER,
        "translate_completion_output_params",
        return_value=fake_response,
    ) as nonstream_mock:
        out = Handler._transform_completion_response(
            completion_response=MagicMock(),
            model="gpt-4o-mini",
            stream=False,
            tool_name_mapping={"t": "tool"},
        )
    nonstream_mock.assert_called_once()
    assert out == fake_response


def test_rr026_transform_streaming_failure_raises() -> None:
    with patch.object(
        handler_mod.ANTHROPIC_ADAPTER,
        "translate_completion_output_params_streaming",
        return_value=None,
    ):
        with pytest.raises(ValueError, match="Failed to transform streaming response"):
            Handler._transform_completion_response(
                completion_response=MagicMock(),
                model="gpt-4o-mini",
                stream=True,
                tool_name_mapping={},
            )


def test_rr026_sync_handler_uses_shared_transform() -> None:
    prepared = ({"model": "gpt-4o-mini"}, {})
    with (
        patch.object(
            Handler,
            "_prepare_completion_kwargs",
            return_value=prepared,
        ),
        patch.object(handler_mod.litellm, "completion", return_value=MagicMock()) as comp,
        patch.object(
            Handler,
            "_transform_completion_response",
            return_value={"ok": True},
        ) as transform,
    ):
        result = Handler.anthropic_messages_handler(
            max_tokens=8,
            messages=[{"role": "user", "content": "x"}],
            model="gpt-4o-mini",
            stream=False,
        )
    comp.assert_called_once()
    transform.assert_called_once()
    assert result == {"ok": True}


@pytest.mark.asyncio
async def test_rr026_async_handler_uses_shared_transform() -> None:
    prepared = ({"model": "gpt-4o-mini"}, {})
    with (
        patch.object(
            Handler,
            "_prepare_completion_kwargs",
            return_value=prepared,
        ),
        patch.object(
            handler_mod.litellm,
            "acompletion",
            new=AsyncMock(return_value=MagicMock()),
        ) as acomp,
        patch.object(
            Handler,
            "_transform_completion_response",
            return_value={"ok": True},
        ) as transform,
    ):
        result = await Handler.async_anthropic_messages_handler(
            max_tokens=8,
            messages=[{"role": "user", "content": "x"}],
            model="gpt-4o-mini",
            stream=False,
        )
    acomp.assert_awaited_once()
    transform.assert_called_once()
    assert result == {"ok": True}


def test_rr026_module_documents_sse_ownership_boundary() -> None:
    doc = inspect.getdoc(handler_mod) or ""
    assert "streaming_iterator" in doc
    assert "prompt_cache_key" in doc or "derive_prompt_cache_key" in doc
    assert "responses_adapters" in doc
