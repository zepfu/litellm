"""RR-054 #9 wrapper parity behavioral tests across nine Anthropic adapters.

Covers shared-driver parity for:
- retryable upstream status codes (Responses pass-through drivers)
- stream vs non-stream finalization
- empty-success handling (config + builders)
- parallel-instruction policy routing
- forced bash tool-choice (responses + completion paths)

No production edits: these tests document current behavior and surface drift.
"""

from __future__ import annotations

import inspect
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, Request, Response
from starlette.responses import StreamingResponse

from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import adapter_config

# ---------------------------------------------------------------------------
# Canonical nine adapters (RR-054 #9)
# ---------------------------------------------------------------------------

RESPONSES_ADAPTERS: list[tuple[str, adapter_config.AnthropicResponsesAdapterConfig]] = [
    ("openai_responses", adapter_config.OPENAI_RESPONSES),
    ("xai_oauth_responses", adapter_config.XAI_OAUTH_RESPONSES),
    ("grok_native_responses", adapter_config.GROK_NATIVE_RESPONSES),
    ("openrouter_responses", adapter_config.OPENROUTER_RESPONSES),
    ("opencode_zen_responses", adapter_config.OPENCODE_ZEN_RESPONSES),
]

COMPLETION_ADAPTERS: list[
    tuple[str, adapter_config.AnthropicCompletionAdapterConfig]
] = [
    ("xai_oauth_completion", adapter_config.XAI_OAUTH_COMPLETION),
    ("nvidia_completion", adapter_config.NVIDIA_COMPLETION),
    ("openrouter_completion", adapter_config.OPENROUTER_COMPLETION),
    ("opencode_zen_completion", adapter_config.OPENCODE_ZEN_COMPLETION),
]

NINE_ADAPTER_LABELS = [name for name, _ in RESPONSES_ADAPTERS] + [
    name for name, _ in COMPLETION_ADAPTERS
]

DEFAULT_RETRYABLE = list(lpe._AAWM_ALIAS_CANDIDATE_RETRYABLE_UPSTREAM_STATUS_CODES_DEFAULT)
PROBE_RETRYABLE = list(lpe._AAWM_ALIAS_CANDIDATE_RETRYABLE_UPSTREAM_STATUS_CODES)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _minimal_request(path: str = "/v1/messages") -> Request:
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("utf-8"),
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 12345),
        "server": ("test", 80),
    }
    return Request(scope)


def _empty_success_responses_body(*, model: str = "test-model") -> dict[str, Any]:
    return {
        "id": "resp_empty",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": model,
        "output": [],
        "usage": {
            "input_tokens": 1,
            "output_tokens": 0,
            "total_tokens": 1,
        },
    }


def _non_empty_responses_body(*, model: str = "test-model") -> dict[str, Any]:
    return {
        "id": "resp_ok",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": model,
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": "hello from adapter",
                        "annotations": [],
                    }
                ],
            }
        ],
        "usage": {
            "input_tokens": 3,
            "output_tokens": 2,
            "total_tokens": 5,
        },
    }


def _parallel_policy_body() -> dict[str, Any]:
    return {
        "parallel_tool_calls": True,
        "instructions": "Original system prompt that must be preserved.",
        "tools": [
            {"type": "function", "name": "Bash", "parameters": {"type": "object"}},
            {
                "type": "function",
                "name": "Read",
                "parameters": {"type": "object"},
            },
        ],
        "input": [],
        "model": "test-model",
    }


def _forced_bash_prepared_body() -> dict[str, Any]:
    return {
        "model": "test-model",
        "messages": [
            {
                "role": "user",
                "content": "Please use the bash tool to list files.",
            }
        ],
        "tools": [
            {
                "name": "Bash",
                "description": "Run shell",
                "input_schema": {"type": "object", "properties": {}},
            }
        ],
    }


def _forced_bash_translated_body() -> dict[str, Any]:
    return {
        "model": "test-model",
        "instructions": "keep me",
        "tools": [
            {
                "type": "function",
                "name": "Bash",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
        "input": [],
    }


# ---------------------------------------------------------------------------
# Inventory: nine adapters present and wired into shared drivers
# ---------------------------------------------------------------------------


def test_rr054_issue9_nine_adapter_inventory() -> None:
    assert len(NINE_ADAPTER_LABELS) == 9
    assert len(RESPONSES_ADAPTERS) == 5
    assert len(COMPLETION_ADAPTERS) == 4

    responses_handlers = {
        "openai_responses": lpe._handle_anthropic_openai_responses_adapter_route,
        "xai_oauth_responses": lpe._handle_anthropic_xai_oauth_responses_adapter_route,
        "grok_native_responses": lpe._handle_anthropic_grok_native_oauth_responses_adapter_route,
        "openrouter_responses": lpe._handle_anthropic_openrouter_responses_adapter_route,
        "opencode_zen_responses": lpe._handle_anthropic_opencode_zen_responses_adapter_route,
    }
    completion_handlers = {
        "xai_oauth_completion": lpe._handle_anthropic_xai_oauth_completion_adapter_route,
        "nvidia_completion": lpe._handle_anthropic_nvidia_completion_adapter_route,
        "openrouter_completion": lpe._handle_anthropic_openrouter_completion_adapter_route,
        "opencode_zen_completion": lpe._handle_anthropic_opencode_zen_completion_adapter_route,
    }
    assert set(responses_handlers) | set(completion_handlers) == set(NINE_ADAPTER_LABELS)

    for name, handler in responses_handlers.items():
        source = inspect.getsource(handler)
        assert "_aawm_adapter_driver.run_responses_adapter_route" in source, name
        assert len(source.splitlines()) <= 25, name

    for name, handler in completion_handlers.items():
        source = inspect.getsource(handler)
        assert "_aawm_adapter_driver.run_completion_adapter_route" in source, name
        assert len(source.splitlines()) <= 25, name


# ---------------------------------------------------------------------------
# Retryable statuses (Responses shared pass-through driver)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("adapter_name,config", RESPONSES_ADAPTERS, ids=[n for n, _ in RESPONSES_ADAPTERS])
@pytest.mark.asyncio
async def test_rr054_issue9_responses_retryable_statuses_default_vs_probe(
    adapter_name: str,
    config: adapter_config.AnthropicResponsesAdapterConfig,
) -> None:
    """All five Responses adapters share the same retry-code selection policy."""
    request = _minimal_request()
    user = MagicMock()
    translated = {"model": "m", "stream": False, "input": []}
    target = "https://example.test/v1/responses"
    headers = {"Authorization": "Bearer t"}
    upstream = Response(content=json.dumps(_non_empty_responses_body()), media_type="application/json")

    transport = AsyncMock(return_value=upstream)
    finalize = AsyncMock(return_value=Response(content=b'{"ok":true}'))

    with patch.object(
        lpe, "_finalize_anthropic_responses_adapter_from_config", finalize
    ):
        await lpe._perform_anthropic_responses_adapter_pass_through(
            config=config,
            request=request,
            user_api_key_dict=user,
            translated_request_body=translated,
            adapter_model="m",
            target_url=target,
            custom_headers=headers,
            client_requested_stream=False,
            use_alias_candidate_probe=False,
            pass_through_fn=transport,
        )
        await lpe._perform_anthropic_responses_adapter_pass_through(
            config=config,
            request=request,
            user_api_key_dict=user,
            translated_request_body=translated,
            adapter_model="m",
            target_url=target,
            custom_headers=headers,
            client_requested_stream=False,
            use_alias_candidate_probe=True,
            pass_through_fn=transport,
        )

    assert transport.await_count == 2
    default_kwargs = transport.await_args_list[0].kwargs
    probe_kwargs = transport.await_args_list[1].kwargs
    assert default_kwargs["retryable_upstream_status_codes"] == DEFAULT_RETRYABLE, (
        f"{adapter_name}: default retryable statuses diverged"
    )
    assert probe_kwargs["retryable_upstream_status_codes"] == PROBE_RETRYABLE, (
        f"{adapter_name}: probe retryable statuses diverged"
    )
    assert default_kwargs["caller_managed_hidden_retry"] is False
    assert probe_kwargs["caller_managed_hidden_retry"] is True
    # Default always includes 429 + common 5xx; probe set excludes 429 by design.
    assert 429 in default_kwargs["retryable_upstream_status_codes"]
    assert 429 not in probe_kwargs["retryable_upstream_status_codes"]
    for code in (500, 502, 503, 504):
        assert code in default_kwargs["retryable_upstream_status_codes"]
        assert code in probe_kwargs["retryable_upstream_status_codes"]


@pytest.mark.asyncio
async def test_rr054_issue9_responses_retryable_statuses_explicit_override() -> None:
    request = _minimal_request()
    transport = AsyncMock(return_value=Response(content=b"{}"))
    with patch.object(
        lpe,
        "_finalize_anthropic_responses_adapter_from_config",
        new=AsyncMock(return_value=Response(content=b'{"ok":true}')),
    ):
        await lpe._perform_anthropic_responses_adapter_pass_through(
            config=adapter_config.OPENAI_RESPONSES,
            request=request,
            user_api_key_dict=MagicMock(),
            translated_request_body={"model": "m", "stream": False},
            adapter_model="m",
            target_url="https://example.test/v1/responses",
            custom_headers={},
            client_requested_stream=False,
            use_alias_candidate_probe=True,
            retryable_upstream_status_codes=[503],
            pass_through_fn=transport,
        )
    assert transport.await_args.kwargs["retryable_upstream_status_codes"] == [503]


# ---------------------------------------------------------------------------
# Stream / non-stream finalization parity
# ---------------------------------------------------------------------------


def test_rr054_issue9_finalize_uses_canonical_malformed_intake_helper() -> None:
    source = inspect.getsource(
        lpe._aawm_responses_finalize.finalize_anthropic_responses_adapter_upstream_response
    )
    runtime_getter = lpe._aawm_responses_finalize._get_runtime
    runtime = (
        runtime_getter()
        if callable(runtime_getter)
        and not hasattr(runtime_getter, "build_malformed_context")
        else runtime_getter
    )
    runtime_source = inspect.getsource(runtime.build_malformed_context)
    assert "build_malformed_context" in source
    assert "_build_malformed_intake_context_for_anthropic_responses_adapter" in (
        runtime_source
    )
    assert "_build_malformed_inplace_context_for_anthropic_responses_adapter" not in (
        runtime_source
    )
    assert hasattr(
        lpe, "_build_malformed_intake_context_for_anthropic_responses_adapter"
    )


@pytest.mark.parametrize("adapter_name,config", RESPONSES_ADAPTERS, ids=[n for n, _ in RESPONSES_ADAPTERS])
@pytest.mark.asyncio
async def test_rr054_issue9_responses_stream_vs_nonstream_finalization(
    adapter_name: str,
    config: adapter_config.AnthropicResponsesAdapterConfig,
) -> None:
    request = _minimal_request()
    body = _non_empty_responses_body(model="m")
    nonstream_upstream = Response(
        content=json.dumps(body),
        media_type="application/json",
        status_code=200,
        headers={"x-upstream": "yes", "content-length": "999"},
    )

    async def _sse_chunks():
        yield b'data: {"type":"response.completed","response":'
        yield json.dumps(body).encode("utf-8")
        yield b"}\n\n"

    stream_upstream = StreamingResponse(
        _sse_chunks(),
        media_type="text/event-stream",
        status_code=200,
        headers={"x-upstream": "stream"},
    )

    canonical_name = "_build_malformed_intake_context_for_anthropic_responses_adapter"
    real_helper = lpe._build_malformed_intake_context_for_anthropic_responses_adapter

    with patch.object(lpe, canonical_name, real_helper):
        # Non-stream branch: Response -> anthropic JSON body
        with patch.object(
            lpe,
            "_build_anthropic_response_from_responses_response",
            return_value=Response(
                content=b'{"type":"message"}', media_type="application/json"
            ),
        ) as build_nonstream:
            nonstream = await lpe._finalize_anthropic_responses_adapter_from_config(
                config=config,
                upstream_response=nonstream_upstream,
                request=request,
                translated_request_body={"model": "m", "stream": False},
                adapter_model="m",
                target_url="https://example.test/v1/responses",
                client_requested_stream=False,
                use_alias_candidate_probe=False,
            )
        assert build_nonstream.called
        assert nonstream.headers.get("x-upstream") == "yes"
        # content-length must not be copied from upstream
        assert nonstream.headers.get("content-length") != "999"
        assert nonstream.status_code == 200

        # Stream branch with client_requested_stream=True keeps streaming wrapper
        with patch.object(
            lpe,
            "_validate_alias_candidate_responses_stream_if_needed",
            new=AsyncMock(side_effect=lambda resp, **kwargs: resp),
        ), patch.object(
            lpe,
            "_build_anthropic_streaming_response_from_responses_stream",
            return_value=StreamingResponse(
                _sse_chunks(), media_type="text/event-stream", status_code=200
            ),
        ) as build_stream:
            streamed = await lpe._finalize_anthropic_responses_adapter_from_config(
                config=config,
                upstream_response=stream_upstream,
                request=request,
                translated_request_body={"model": "m", "stream": True},
                adapter_model="m",
                target_url="https://example.test/v1/responses",
                client_requested_stream=True,
                use_alias_candidate_probe=False,
            )
        assert build_stream.called
        assert isinstance(streamed, StreamingResponse)

        # Stream upstream + client_requested_stream=False collapses to non-stream
        collected_body = _non_empty_responses_body(model="m")
        with patch.object(
            lpe,
            "_validate_alias_candidate_responses_stream_if_needed",
            new=AsyncMock(side_effect=lambda resp, **kwargs: resp),
        ), patch.object(
            lpe,
            "_collect_responses_response_from_stream",
            new=AsyncMock(return_value=collected_body),
        ), patch.object(
            lpe,
            "_build_anthropic_response_from_responses_response",
            return_value=Response(
                content=b'{"type":"message"}', media_type="application/json"
            ),
        ) as build_collapsed:
            collapsed = await lpe._finalize_anthropic_responses_adapter_from_config(
                config=config,
                upstream_response=StreamingResponse(
                    _sse_chunks(), media_type="text/event-stream", status_code=201
                ),
                request=request,
                translated_request_body={"model": "m", "stream": False},
                adapter_model="m",
                target_url="https://example.test/v1/responses",
                client_requested_stream=False,
                use_alias_candidate_probe=False,
            )
        assert build_collapsed.called
        assert isinstance(collapsed, Response)
        assert not isinstance(collapsed, StreamingResponse)
        assert collapsed.status_code == 201


@pytest.mark.parametrize("adapter_name,config", COMPLETION_ADAPTERS, ids=[n for n, _ in COMPLETION_ADAPTERS])
@pytest.mark.asyncio
async def test_rr054_issue9_completion_stream_vs_nonstream_finalization(
    adapter_name: str,
    config: adapter_config.AnthropicCompletionAdapterConfig,
) -> None:
    request = _minimal_request()
    prepared = {
        "model": "m",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
    }

    class _FakeCompletion:
        def model_dump(self, **kwargs):  # noqa: ANN003
            return {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "ok"}],
                "model": "m",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }

        def model_dump_json(self, **kwargs):  # noqa: ANN003
            return json.dumps(self.model_dump())

    nonstream_payload = _FakeCompletion()

    with patch(
        "litellm.llms.anthropic.experimental_pass_through.adapters.handler."
        "LiteLLMMessagesToCompletionTransformationHandler.async_anthropic_messages_handler",
        new=AsyncMock(return_value=nonstream_payload),
    ), patch.object(
        lpe,
        "_build_anthropic_response_from_completion_adapter_response",
        return_value=Response(content=b'{"type":"message"}', media_type="application/json"),
    ) as build_nonstream, patch.object(
        lpe, "_annotate_request_scope_for_adapted_access_log"
    ):
        result = await lpe._perform_anthropic_completion_adapter_messages_call(
            config=config,
            request=request,
            prepared_request_body=prepared,
            adapter_model="m",
            target_url="https://example.test/v1/chat/completions",
            api_key="k",
            api_base="https://example.test/v1",
            client_requested_stream=False,
        )
    assert build_nonstream.called
    assert isinstance(result, Response)

    prepared_stream = dict(prepared)
    prepared_stream["stream"] = True
    stream_payload = MagicMock(name="completion_stream")

    with patch(
        "litellm.llms.anthropic.experimental_pass_through.adapters.handler."
        "LiteLLMMessagesToCompletionTransformationHandler.async_anthropic_messages_handler",
        new=AsyncMock(return_value=stream_payload),
    ), patch.object(
        lpe,
        "_build_anthropic_streaming_response_from_completion_adapter_stream",
        return_value=StreamingResponse(iter([b"data: {}\n\n"]), media_type="text/event-stream"),
    ) as build_stream, patch.object(
        lpe, "_annotate_request_scope_for_adapted_access_log"
    ):
        streamed = await lpe._perform_anthropic_completion_adapter_messages_call(
            config=config,
            request=request,
            prepared_request_body=prepared_stream,
            adapter_model="m",
            target_url="https://example.test/v1/chat/completions",
            api_key="k",
            api_base="https://example.test/v1",
            client_requested_stream=True,
        )
    assert build_stream.called
    assert isinstance(streamed, StreamingResponse)


# ---------------------------------------------------------------------------
# Empty-success handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("adapter_name,config", RESPONSES_ADAPTERS, ids=[n for n, _ in RESPONSES_ADAPTERS])
def test_rr054_issue9_responses_empty_success_config_and_builder(
    adapter_name: str,
    config: adapter_config.AnthropicResponsesAdapterConfig,
) -> None:
    kwargs = adapter_config.responses_finalize_kwargs(
        config,
        adapter_model="m",
        translated_request_body={"model": "m", "stream": False},
    )
    if config.reject_empty_success:
        assert kwargs.get("response_builder_kwargs", {}).get("reject_empty_success") is True
        assert kwargs.get("stream_builder_kwargs", {}).get("reject_empty_success") is True
        with pytest.raises(HTTPException) as exc_info:
            lpe._build_anthropic_response_from_responses_response(
                _empty_success_responses_body(model="m"),
                **kwargs["response_builder_kwargs"],
            )
        assert exc_info.value.status_code == 502
        detail = exc_info.value.detail
        assert isinstance(detail, dict)
        assert "empty successful response" in str(detail.get("error", "")).lower()
    else:
        assert "response_builder_kwargs" not in kwargs
        assert "stream_builder_kwargs" not in kwargs
        # Empty success is allowed through the non-reject builder path without 502.
        # Translation may still succeed or depend on adapter internals; ensure no
        # empty-success HTTPException is raised by the reject gate.
        try:
            result = lpe._build_anthropic_response_from_responses_response(
                _empty_success_responses_body(model="m"),
                reject_empty_success=False,
            )
        except HTTPException as exc:
            # If translation fails for other reasons, it must not be the empty-success gate.
            assert "empty successful response" not in str(exc.detail).lower()
        else:
            assert isinstance(result, Response)


def test_rr054_issue9_empty_success_policy_matrix() -> None:
    """Document which Responses adapters reject empty success."""
    matrix = {
        name: cfg.reject_empty_success for name, cfg in RESPONSES_ADAPTERS
    }
    assert matrix == {
        "openai_responses": False,
        "xai_oauth_responses": False,
        "grok_native_responses": False,
        "openrouter_responses": True,
        "opencode_zen_responses": True,
    }


# ---------------------------------------------------------------------------
# Parallel-instruction policy parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("adapter_name,config", RESPONSES_ADAPTERS, ids=[n for n, _ in RESPONSES_ADAPTERS])
def test_rr054_issue9_parallel_policy_routing_by_config(
    adapter_name: str,
    config: adapter_config.AnthropicResponsesAdapterConfig,
) -> None:
    prepared = _forced_bash_prepared_body()
    translated = _parallel_policy_body()
    original_instructions = translated["instructions"]

    updated = lpe._apply_anthropic_responses_adapter_policies_from_config(
        prepared,
        dict(translated),
        config=config,
    )

    if config.use_openai_parallel_policy:
        assert original_instructions in (updated.get("instructions") or "")
        assert (updated.get("instructions") or "").startswith(
            lpe._OPENAI_ADAPTER_PARALLEL_FUNCTION_TOOL_INSTRUCTIONS[:40]
        ) or original_instructions in (updated.get("instructions") or "")
        # RR-054 #20: never wipe original system prompt.
        assert original_instructions in (updated.get("instructions") or "")
        meta = (updated.get("litellm_metadata") or {})
        assert meta.get("openai_adapter_parallel_instruction_policy_applied") is True
        assert meta.get("openai_adapter_parallel_instruction_mode") == "prepend"
    else:
        # OpenRouter / OpenCode Zen Responses keep a separate parallel helper
        # at the route level (not via use_openai_parallel_policy config path).
        assert updated.get("instructions") == original_instructions or (
            # forced bash may still mutate tool_choice but not wipe instructions
            original_instructions in (updated.get("instructions") or "")
        )
        meta = updated.get("litellm_metadata") or {}
        assert meta.get("openai_adapter_parallel_instruction_policy_applied") is not True


def test_rr054_issue9_openrouter_family_parallel_helper_still_applies() -> None:
    """Routes that set use_openai_parallel_policy=False still have a parallel helper."""
    body = _parallel_policy_body()
    updated, changes = lpe._apply_openrouter_adapter_parallel_instruction_policy(body)
    assert changes.get("openrouter_adapter_parallel_instruction_policy_applied") is True
    assert body["instructions"] in (updated.get("instructions") or "")
    assert changes.get("openrouter_adapter_parallel_instruction_mode") == "prepend"

    # Provider modules invoke injected policy hooks, and route wiring supplies
    # the canonical OpenRouter-family helpers.
    for provider_module, runtime in (
        (
            lpe._anthropic_openrouter_provider,
            lpe._ANTHROPIC_OPENROUTER_PROVIDER_RUNTIME,
        ),
        (
            lpe._anthropic_opencode_zen_provider,
            lpe._ANTHROPIC_OPENCODE_ZEN_PROVIDER_RUNTIME,
        ),
    ):
        provider_source = inspect.getsource(provider_module.prepare_responses_route)
        parallel_source = inspect.getsource(runtime.apply_parallel_policy)
        forced_source = inspect.getsource(runtime.apply_forced_tool_choice)
        assert "runtime.apply_parallel_policy" in provider_source
        assert "runtime.apply_forced_tool_choice" in provider_source
        assert "_apply_openrouter_adapter_parallel_instruction_policy" in parallel_source
        assert "_apply_forced_bash_tool_choice_for_responses_adapter" in forced_source

    # Config-driven Responses routes use the shared OpenAI parallel policy path.
    for name, provider_module, policy_hook in (
        ("openai", lpe._anthropic_openai_provider, "runtime.apply_policies"),
        (
            "xai",
            lpe._anthropic_xai_provider,
            "runtime.apply_responses_policies",
        ),
        ("grok", lpe._anthropic_grok_provider, "runtime.apply_policies"),
    ):
        source = inspect.getsource(provider_module.prepare_responses_route)
        assert policy_hook in source, name


def test_rr054_issue9_completion_adapters_do_not_use_responses_parallel_policy() -> None:
    for name, prepare_name in (
        ("xai_oauth_completion", "_prepare_anthropic_xai_oauth_completion_adapter_route"),
        ("nvidia_completion", "_prepare_anthropic_nvidia_completion_adapter_route"),
        ("openrouter_completion", "_prepare_anthropic_openrouter_completion_adapter_route"),
        ("opencode_zen_completion", "_prepare_anthropic_opencode_zen_completion_adapter_route"),
    ):
        source = inspect.getsource(getattr(lpe, prepare_name))
        assert "_apply_openai_adapter_parallel_instruction_policy" not in source, name
        assert "_apply_openrouter_adapter_parallel_instruction_policy" not in source, name
        assert "_apply_anthropic_responses_adapter_policies_from_config" not in source, name


# ---------------------------------------------------------------------------
# Forced bash (where applicable)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("adapter_name,config", RESPONSES_ADAPTERS, ids=[n for n, _ in RESPONSES_ADAPTERS])
def test_rr054_issue9_responses_forced_bash_parity(
    adapter_name: str,
    config: adapter_config.AnthropicResponsesAdapterConfig,
) -> None:
    prepared = _forced_bash_prepared_body()
    translated = _forced_bash_translated_body()

    if config.use_openai_parallel_policy:
        updated = lpe._apply_anthropic_responses_adapter_policies_from_config(
            prepared,
            dict(translated),
            config=config,
        )
    else:
        # OpenRouter / OpenCode Zen apply forced bash via the shared helper
        # outside the openai parallel config path.
        updated, changes = lpe._apply_forced_bash_tool_choice_for_responses_adapter(
            prepared,
            dict(translated),
        )
        assert changes.get("forced_explicit_bash_tool_choice") == "Bash"

    assert updated.get("tool_choice") == {"type": "function", "name": "Bash"}
    meta = updated.get("litellm_metadata") or {}
    assert meta.get("forced_explicit_bash_tool_choice") == "Bash" or (
        updated.get("tool_choice", {}).get("name") == "Bash"
    )


@pytest.mark.parametrize("adapter_name,config", COMPLETION_ADAPTERS, ids=[n for n, _ in COMPLETION_ADAPTERS])
def test_rr054_issue9_completion_forced_bash_parity(
    adapter_name: str,
    config: adapter_config.AnthropicCompletionAdapterConfig,
) -> None:
    prepared = _forced_bash_prepared_body()
    updated = lpe._prepare_anthropic_completion_adapter_request_body(
        dict(prepared),
        adapter_model="m",
        route_family=config.route_family,
        tag_prefix=config.tag_prefix,
        span_name=config.span_name,
        target_endpoint_label=config.target_endpoint_label,
    )
    assert updated.get("tool_choice") == {"type": "tool", "name": "Bash"}
    meta = updated.get("litellm_metadata") or {}
    assert meta.get("forced_explicit_bash_tool_choice") == "Bash"


def test_rr054_issue9_forced_bash_skips_when_tool_choice_already_set() -> None:
    prepared = _forced_bash_prepared_body()
    prepared["tool_choice"] = {"type": "tool", "name": "Read"}
    translated = _forced_bash_translated_body()
    translated["tool_choice"] = {"type": "function", "name": "Read"}

    resp_body, resp_changes = lpe._apply_forced_bash_tool_choice_for_responses_adapter(
        prepared, dict(translated)
    )
    assert resp_changes == {}
    assert resp_body.get("tool_choice") == {"type": "function", "name": "Read"}

    completion_changes = lpe._maybe_force_explicit_bash_tool_choice_for_completion_adapter(
        dict(prepared)
    )
    assert completion_changes == {}


def test_rr054_issue9_forced_bash_skips_without_explicit_prompt() -> None:
    prepared = _forced_bash_prepared_body()
    prepared["messages"] = [{"role": "user", "content": "just say hi"}]
    translated = _forced_bash_translated_body()

    _, resp_changes = lpe._apply_forced_bash_tool_choice_for_responses_adapter(
        prepared, dict(translated)
    )
    assert resp_changes == {}
    completion_changes = lpe._maybe_force_explicit_bash_tool_choice_for_completion_adapter(
        dict(prepared)
    )
    assert completion_changes == {}


# ---------------------------------------------------------------------------
# Cross-adapter divergence report helpers (assert known intentional drift)
# ---------------------------------------------------------------------------


def test_rr054_issue9_known_intentional_divergences() -> None:
    """Capture intentional residual divergences after shared-driver extraction."""
    divergences: list[str] = []

    # 1) Empty-success reject only on OpenRouter + OpenCode Zen Responses.
    rejectors = [n for n, c in RESPONSES_ADAPTERS if c.reject_empty_success]
    non_rejectors = [n for n, c in RESPONSES_ADAPTERS if not c.reject_empty_success]
    if set(rejectors) != {"openrouter_responses", "opencode_zen_responses"}:
        divergences.append(f"unexpected empty-success rejectors: {rejectors}")
    if set(non_rejectors) != {
        "openai_responses",
        "xai_oauth_responses",
        "grok_native_responses",
    }:
        divergences.append(f"unexpected empty-success non-rejectors: {non_rejectors}")

    # 2) Parallel policy path: openai-config vs openrouter helper.
    openai_parallel = [n for n, c in RESPONSES_ADAPTERS if c.use_openai_parallel_policy]
    openrouter_parallel = [
        n for n, c in RESPONSES_ADAPTERS if not c.use_openai_parallel_policy
    ]
    if set(openai_parallel) != {
        "openai_responses",
        "xai_oauth_responses",
        "grok_native_responses",
    }:
        divergences.append(f"unexpected openai parallel configs: {openai_parallel}")
    if set(openrouter_parallel) != {
        "openrouter_responses",
        "opencode_zen_responses",
    }:
        divergences.append(
            f"unexpected openrouter-family parallel configs: {openrouter_parallel}"
        )

    # 3) OpenCode Zen Responses skips stream probe validation.
    if not adapter_config.OPENCODE_ZEN_RESPONSES.skip_stream_probe_validation:
        divergences.append("opencode_zen_responses should skip stream probe validation")
    for name, cfg in RESPONSES_ADAPTERS:
        if name == "opencode_zen_responses":
            continue
        if cfg.skip_stream_probe_validation:
            divergences.append(f"{name} unexpectedly skips stream probe validation")

    # 4) Only OpenAI Responses defaults use_codex_native_tools=True.
    native_tools = [n for n, c in RESPONSES_ADAPTERS if c.default_use_codex_native_tools]
    if native_tools != ["openai_responses"]:
        divergences.append(f"unexpected default_use_codex_native_tools: {native_tools}")

    # 5) OpenRouter / OpenCode Zen provider modules retain their distinct
    # parallel/forced-bash hook path.
    for name, provider_module, runtime in (
        (
            "openrouter_responses",
            lpe._anthropic_openrouter_provider,
            lpe._ANTHROPIC_OPENROUTER_PROVIDER_RUNTIME,
        ),
        (
            "opencode_zen_responses",
            lpe._anthropic_opencode_zen_provider,
            lpe._ANTHROPIC_OPENCODE_ZEN_PROVIDER_RUNTIME,
        ),
    ):
        source = inspect.getsource(provider_module.prepare_responses_route)
        if "runtime.apply_policies" in source:
            divergences.append(
                f"{name}: expected residual inline policy path, found config policy helper"
            )
        if "runtime.apply_parallel_policy" not in source or (
            "_apply_openrouter_adapter_parallel_instruction_policy"
            not in inspect.getsource(runtime.apply_parallel_policy)
        ):
            divergences.append(f"{name}: missing openrouter parallel helper call")
        if "runtime.apply_forced_tool_choice" not in source or (
            "_apply_forced_bash_tool_choice_for_responses_adapter"
            not in inspect.getsource(runtime.apply_forced_tool_choice)
        ):
            divergences.append(f"{name}: missing forced-bash helper call")

    assert divergences == [], "Unexpected wrapper parity divergences:\n- " + "\n- ".join(
        divergences
    )


def test_rr054_issue9_shared_driver_surface_stable() -> None:
    for attr in (
        "_perform_anthropic_responses_adapter_pass_through",
        "_perform_anthropic_completion_adapter_messages_call",
        "_finalize_anthropic_responses_adapter_from_config",
        "_finalize_anthropic_responses_adapter_upstream_response",
        "_apply_anthropic_responses_adapter_policies_from_config",
        "_apply_anthropic_responses_adapter_common_request_policies",
        "_prepare_anthropic_completion_adapter_request_body",
        "_apply_forced_bash_tool_choice_for_responses_adapter",
        "_maybe_force_explicit_bash_tool_choice_for_completion_adapter",
    ):
        assert hasattr(lpe, attr), attr
