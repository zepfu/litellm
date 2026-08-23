"""Stubbed Codex passthrough dispatch for OpenCode Go ox-alpha.

Proves POST /openai_passthrough/v1/responses with
model=opencode-go/ox-alpha-free hits ``/zen/go/v1/chat/completions``.
HTTP is stubbed; this is not live OpenCode or OpenRouter fanout.
"""

from __future__ import annotations

import json
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import Response
from fastapi.responses import StreamingResponse

from litellm.proxy._types import ProxyException

from litellm.llms.anthropic.experimental_pass_through.providers.opencode_zen import (
    normalization,
)
from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime import (
    codex_candidate_calls,
    model_resolution,
)
from litellm.proxy.pass_through_endpoints.providers.opencode_zen import (
    runtime as zen_runtime,
)


_ADAPTER_MODEL = "ox-alpha-free"
_CODEX_GO_ROUTE_FAMILY = "codex_opencode_go_adapter"
_GO_CHAT_COMPLETIONS_URL = "https://opencode.ai/zen/go/v1/chat/completions"
_ZEN_CHAT_COMPLETIONS_URL = "https://opencode.ai/zen/v1/chat/completions"


def _request() -> MagicMock:
    request = MagicMock()
    request.headers = {}
    request.scope = {}
    request.state = MagicMock()
    return request


def _assemble_headers(*, api_key: str | None, request: Any) -> dict[str, str]:
    _ = request
    headers: dict[str, str] = {}
    if api_key is not None:
        headers = {
            "authorization": f"Bearer {api_key}",
            "api-key": api_key,
        }
    return headers


def _normalize_endpoint_for_target(endpoint: str, base_target_url: str) -> str:
    normalized_endpoint = httpx.URL(endpoint).path
    if not normalized_endpoint.startswith("/"):
        normalized_endpoint = "/" + normalized_endpoint
    if (
        httpx.URL(base_target_url).path.rstrip("/") == "/v1"
        and normalized_endpoint.startswith("/v1/")
    ):
        return normalized_endpoint[len("/v1") :]
    return normalized_endpoint


def _join_url_paths(base_url: httpx.URL, path: str, _provider: str) -> str:
    if not base_url.path or base_url.path == "/":
        return str(base_url.copy_with(path=path))
    return str(
        base_url.copy_with(path=f"{base_url.path.rstrip('/')}/{path.lstrip('/')}")
    )


def _close_logging_coroutine(async_coroutine, metadata=None):
    _ = metadata
    close = getattr(async_coroutine, "close", None)
    if callable(close):
        close()


def _chat_completion_payload(
    *,
    content: str | None,
    completion_id: str = "chatcmpl-opencode-go",
    empty_choices: bool = False,
    prompt_tokens: int = 4,
    completion_tokens: int = 2,
) -> dict[str, Any]:
    if empty_choices:
        choices: list[dict[str, Any]] = []
        completion_tokens = 0
        prompt_tokens = 0
    else:
        choices = [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content or ""},
                "finish_reason": "stop",
            }
        ]
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": 1,
        "model": _ADAPTER_MODEL,
        "choices": choices,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "output_tokens": completion_tokens,
        },
    }


def _validate_outgoing_egress_capturing(captured: dict[str, Any]):
    def _validate_outgoing_egress(
        *,
        url,
        headers,
        credential_family,
        expected_target_family,
        **kwargs,
    ):
        captured["url"] = str(url)
        captured["headers"] = headers
        captured["credential_family"] = credential_family
        captured["expected_target_family"] = expected_target_family
        captured["egress_kwargs"] = kwargs

    return _validate_outgoing_egress


def _go_adapter_host(
    *,
    litellm_module: Any,
    captured: dict[str, Any],
) -> dict[str, Any]:
    validate_outgoing_egress = _validate_outgoing_egress_capturing(captured)
    return {
        "__builtins__": __builtins__,
        "litellm": litellm_module,
        "httpx": httpx,
        "Response": Response,
        "StreamingResponse": StreamingResponse,
        "_get_opencode_go_target_base": zen_runtime._get_opencode_go_target_base,
        "_join_opencode_zen_passthrough_url": (
            zen_runtime._join_opencode_zen_passthrough_url
        ),
        "_load_opencode_zen_api_key_for_candidate": AsyncMock(
            return_value="opencode-go-test-key"
        ),
        "BaseOpenAIPassThroughHandler": MagicMock(
            _assemble_headers=MagicMock(
                return_value={"Authorization": "Bearer opencode-go-test-key"}
            )
        ),
        "HttpPassThroughEndpointHelpers": MagicMock(
            validate_outgoing_egress=validate_outgoing_egress
        ),
        "_annotate_request_scope_for_adapted_access_log": MagicMock(),
        "_build_adapted_route_rollup_kwargs": MagicMock(return_value={}),
        "_emit_adapted_route_access_log": MagicMock(),
        "_add_route_family_logging_metadata": lambda body, family: body,
        "_get_proxy_shared_aiohttp_session": lambda: None,
        "_opencode_zen_candidate_unavailable_detail": lambda exc: None,
        "_maybe_raise_opencode_zen_direct_rate_limit": lambda exc: None,
        "_OPENCODE_GO_FREE_MODELS": frozenset({"ox-alpha-free"}),
    }


async def _invoke_codex_opencode_go_adapter_route(
    monkeypatch: pytest.MonkeyPatch,
    *,
    prepared_request_body: dict[str, Any],
    completion: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    import litellm
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
    from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.payload_validation import (
        _is_codex_auto_agent_empty_success_responses_body,
        _raise_codex_auto_agent_empty_success_response,
    )
    from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.sse import (
        _responses_sse_from_iterator,
        _responses_sse_from_repaired_response_body,
    )

    handler = getattr(
        codex_candidate_calls, "_handle_codex_opencode_go_adapter_route", None
    )
    assert handler is not None, "missing _handle_codex_opencode_go_adapter_route"

    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        GLOBAL_LOGGING_WORKER,
        "ensure_initialized_and_enqueue",
        _close_logging_coroutine,
    )
    monkeypatch.setenv("LITELLM_OPENCODE_API_KEY", "opencode-go-test-key")
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)

    async def _fake_acompletion(**kwargs: Any) -> dict[str, Any]:
        captured["completion"] = kwargs
        return completion

    monkeypatch.setattr(litellm, "acompletion", _fake_acompletion)

    host = _go_adapter_host(litellm_module=litellm, captured=captured)
    host["_responses_sse_from_iterator"] = _responses_sse_from_iterator
    host["_responses_sse_from_repaired_response_body"] = (
        _responses_sse_from_repaired_response_body
    )
    host["_raise_codex_auto_agent_empty_success_response"] = (
        _raise_codex_auto_agent_empty_success_response
    )
    host["_is_codex_auto_agent_empty_success_responses_body"] = (
        _is_codex_auto_agent_empty_success_responses_body
    )
    codex_candidate_calls.install(host)
    host["HttpPassThroughEndpointHelpers"] = MagicMock(
        validate_outgoing_egress=_validate_outgoing_egress_capturing(captured)
    )
    host["_get_opencode_go_target_base"] = zen_runtime._get_opencode_go_target_base
    host["_join_opencode_zen_passthrough_url"] = (
        zen_runtime._join_opencode_zen_passthrough_url
    )
    host["_load_opencode_zen_api_key_for_candidate"] = AsyncMock(
        return_value="opencode-go-test-key"
    )

    async def _fake_perform(
        *,
        completion_call_kwargs: dict[str, Any],
        litellm_metadata: dict[str, Any],
        accepted_trace_user_id: Any = None,
        is_known_free_direct: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        captured["completion"] = completion_call_kwargs
        captured["perform_kwargs"] = {
            "litellm_metadata": litellm_metadata,
            "accepted_trace_user_id": accepted_trace_user_id,
            "is_known_free_direct": is_known_free_direct,
            **kwargs,
        }
        return completion

    host["_perform_opencode_zen_completion_call"] = _fake_perform
    bound_handler = host.get(
        "_handle_codex_opencode_go_adapter_route",
        handler,
    )
    try:
        response = await bound_handler(
            endpoint="/v1/responses",
            request=_request(),
            fastapi_response=MagicMock(spec=Response),
            user_api_key_dict=MagicMock(),
            prepared_request_body=prepared_request_body,
            adapter_model=_ADAPTER_MODEL,
            use_alias_candidate_probe=True,
        )
    finally:
        await GLOBAL_LOGGING_WORKER.stop()
        await litellm.close_litellm_async_clients()
    return response, captured


async def _collect_response_text(response: Any) -> str:
    iterator = getattr(response, "body_iterator", None)
    if iterator is not None:
        chunks: list[str] = []
        async for chunk in iterator:
            if isinstance(chunk, bytes):
                chunks.append(chunk.decode())
            else:
                chunks.append(str(chunk))
        return "".join(chunks)
    body = getattr(response, "body", b"")
    if isinstance(body, bytes):
        return body.decode()
    return str(body)


def _parse_sse_terminal_event(sse_text: str) -> dict[str, Any]:
    completed_at = sse_text.find("event: response.completed")
    done_at = sse_text.find("data: [DONE]")
    assert completed_at != -1, (
        "Ohmypi requires a terminal SSE event; missing "
        f"`event: response.completed` in {sse_text!r}"
    )
    assert done_at != -1, f"missing `data: [DONE]` in {sse_text!r}"
    assert completed_at < done_at, (
        "`event: response.completed` must appear before `data: [DONE]`; "
        f"got {sse_text!r}"
    )
    block = sse_text[completed_at:done_at]
    data_line = next(
        (line for line in block.splitlines() if line.startswith("data: ")),
        None,
    )
    assert data_line is not None, f"missing data payload in {block!r}"
    payload = json.loads(data_line[len("data: ") :])
    assert isinstance(payload, dict)
    return payload


def _empty_success_error_code(exc: BaseException) -> str | None:
    detail = getattr(exc, "detail", None)
    if isinstance(detail, dict):
        error = detail.get("error")
        if isinstance(error, dict):
            code = error.get("code")
            if isinstance(code, str):
                return code
    return None


async def _async_empty_payload(**_kwargs: Any) -> dict[str, Any]:
    return {}


def _normalization_runtime() -> normalization.Runtime:
    return normalization.Runtime(
        clean_secret_string=lambda value: value.strip() if isinstance(value, str) else None,
        merge_metadata=lambda body, **_kwargs: body,
        add_logging_metadata=lambda body, **_kwargs: body,
        build_span=lambda **kwargs: kwargs,
        transform_responses_api_request_to_chat_completion_request=(
            lambda **_kwargs: {}
        ),
        async_responses_api_session_handler=_async_empty_payload,
        iterate_responses_sse_events=lambda iterator: iterator,
        coerce_namespace_to_mapping=lambda value: value,
        responses_output_item_has_meaningful_content=lambda item: bool(item),
        streaming_response_factory=Response,
    )


def _merge_metadata(
    body: dict[str, Any],
    *,
    tags_to_add: list[str],
    extra_fields: dict[str, Any],
) -> dict[str, Any]:
    updated = dict(body)
    updated["tags"] = tags_to_add
    updated.update(extra_fields)
    return updated


@pytest.fixture()
def configured_go_runtime(monkeypatch: pytest.MonkeyPatch):
    prior_runtime = zen_runtime._runtime
    for name in (
        "OPENCODE_GO_API_BASE",
        "AAWM_OPENCODE_GO_API_BASE",
        "OPENCODE_ZEN_API_BASE",
        "AAWM_OPENCODE_ZEN_API_BASE",
        "LITELLM_OPENCODE_API_KEY",
        "OPENCODE_API_KEY",
        "LITELLM_OPENCODE_AUTH_FILE",
        "OPENCODE_AUTH_FILE",
    ):
        monkeypatch.delenv(name, raising=False)

    zen_runtime.configure_runtime(
        zen_runtime.Runtime(
            get_secret_str=lambda name: os.getenv(name),
            assemble_headers=_assemble_headers,
            normalize_endpoint_for_target=_normalize_endpoint_for_target,
            join_url_paths=_join_url_paths,
            extract_exception_status_code=lambda exc: getattr(
                exc, "status_code", None
            ),
            extract_exception_detail=lambda exc: getattr(exc, "detail", None),
            merge_metadata=_merge_metadata,
            add_route_family_logging_metadata=lambda body, family: {
                **body,
                "route_family": family,
            },
            build_langfuse_span_descriptor=lambda **kwargs: kwargs,
            normalization_runtime_factory=_normalization_runtime,
            is_openai_responses_endpoint=lambda endpoint: (
                httpx.URL(endpoint).path in {"/responses", "/v1/responses"}
            ),
            has_anthropic_responses_adapter_endpoint=lambda endpoint: (
                httpx.URL(endpoint).path in {"/messages", "/v1/messages"}
            ),
            get_anthropic_adapter_model_candidates=lambda body: (
                [body["model"]] if isinstance(body.get("model"), str) else []
            ),
        )
    )
    yield
    zen_runtime._runtime = prior_runtime
    zen_runtime._get_anthropic_opencode_zen_normalization_runtime.cache_clear()


def _openai_responses_endpoint(endpoint: str) -> bool:
    normalized = endpoint.strip()
    if not normalized.startswith("/"):
        normalized = f"/{normalized}"
    return normalized in {"/v1/responses", "/responses"} or normalized.startswith(
        "/v1/responses/"
    )


def test_should_resolve_prefixed_direct_go_models_on_responses() -> None:
    assert not hasattr(
        model_resolution, "_resolve_anthropic_opencode_go_adapter_model"
    )
    assert (
        "_resolve_anthropic_opencode_go_adapter_model"
        not in model_resolution._HOST_FUNCTION_NAMES
    )

    from litellm.proxy.pass_through_endpoints import (
        llm_passthrough_endpoints as lpe,
    )

    production_resolver = lpe._resolve_codex_opencode_go_adapter_model
    assert production_resolver is not None

    host: dict[str, Any] = {
        "_is_openai_responses_endpoint": _openai_responses_endpoint,
        "_OPENCODE_GO_PROVIDER": "opencode_go",
        "_OPENCODE_GO_FREE_MODELS": frozenset({"ox-alpha-free"}),
        "_OPENCODE_ZEN_PROVIDER": "opencode_zen",
        "_OPENCODE_ZEN_FREE_MODELS": frozenset(),
    }
    model_resolution.install(host)
    resolve = host["_resolve_codex_opencode_go_adapter_model"]

    assert (
        resolve({"model": "opencode-go/ox-alpha-free"}, endpoint="/v1/responses")
        == _ADAPTER_MODEL
    )
    assert (
        resolve({"model": "opencode_go/ox-alpha-free"}, endpoint="v1/responses")
        == _ADAPTER_MODEL
    )
    assert resolve({"model": "ox-alpha-free"}, endpoint="/v1/responses") is None
    assert (
        resolve({"model": "opencode/ox-alpha-free"}, endpoint="/v1/responses")
        is None
    )
    assert (
        resolve(
            {"model": "opencode-zen/ox-alpha-free"},
            endpoint="/v1/responses",
        )
        is None
    )
    assert (
        resolve(
            {"model": "opencode-go/ox-alpha-free"},
            endpoint="/v1/chat/completions",
        )
        is None
    )

    assert production_resolver.__name__ == (
        "_resolve_codex_opencode_go_adapter_model"
    )


@pytest.mark.asyncio
async def test_should_prepare_or_handle_codex_go_route_to_zen_go_chat_url(
    monkeypatch: pytest.MonkeyPatch,
    configured_go_runtime,
) -> None:
    response, captured = await _invoke_codex_opencode_go_adapter_route(
        monkeypatch,
        prepared_request_body={
            "model": f"opencode-go/{_ADAPTER_MODEL}",
            "input": "hello from openai_passthrough",
            "stream": False,
        },
        completion=_chat_completion_payload(content="ok"),
    )

    target_url = captured.get("url") or ""
    completion_kwargs = captured.get("completion") or {}
    api_base = str(completion_kwargs.get("api_base") or "")
    joined = target_url or api_base
    assert "/zen/go/v1/" in joined
    assert joined != _ZEN_CHAT_COMPLETIONS_URL
    assert _GO_CHAT_COMPLETIONS_URL in joined or joined.endswith(
        "/zen/go/v1/chat/completions"
    )
    assert completion_kwargs.get("model") == _ADAPTER_MODEL
    assert captured.get("credential_family") == "opencode"
    assert captured.get("expected_target_family") == "opencode"
    auth_header = ""
    headers = captured.get("headers") or {}
    if isinstance(headers, dict):
        auth_header = str(
            headers.get("Authorization") or headers.get("authorization") or ""
        )
    if not auth_header:
        auth_header = str(completion_kwargs.get("api_key") or "")
    assert "opencode-go-test-key" in auth_header
    assert not isinstance(response, StreamingResponse)
    assert str(getattr(response, "media_type", "")).startswith("application/json")
    response_body = json.loads(response.body)
    assert response_body["object"] == "response"


@pytest.mark.asyncio
async def test_opencode_go_stream_true_returns_sse_not_json(
    monkeypatch: pytest.MonkeyPatch,
    configured_go_runtime,
) -> None:
    """Ohmypi openai-responses stream:true must get SSE, not JSON 200.

    Live 2026-08-23 defect: `_handle_codex_opencode_go_adapter_route` always
    returns FastAPI JSON even when prepared_request_body[\"stream\"] is true.
    Ohmypi then throws incomplete-stream because there is no
    response.completed / response.incomplete / response.done terminal event.
    """
    response, captured = await _invoke_codex_opencode_go_adapter_route(
        monkeypatch,
        prepared_request_body={
            "model": f"opencode-go/{_ADAPTER_MODEL}",
            "input": "Reply with exactly the word PONG.",
            "stream": True,
        },
        completion=_chat_completion_payload(content="PONG"),
    )

    completion_kwargs = captured.get("completion") or {}
    assert completion_kwargs.get("stream") is True
    assert not isinstance(response, Response) or isinstance(response, StreamingResponse)
    assert isinstance(response, StreamingResponse), (
        "stream=true must wrap `_responses_sse_from_iterator` / "
        "`_responses_sse_from_repaired_response_body`; got JSON "
        f"{type(response).__name__} media_type="
        f"{getattr(response, 'media_type', None)!r}"
    )
    media_type = str(getattr(response, "media_type", "") or "")
    assert media_type.startswith("text/event-stream"), (
        f"expected text/event-stream, got {media_type!r}"
    )
    sse_text = await _collect_response_text(response)
    assert "event: response.completed" in sse_text
    assert "data: [DONE]" in sse_text
    payload = _parse_sse_terminal_event(sse_text)
    assert payload.get("type") == "response.completed"
    completed = payload.get("response")
    assert isinstance(completed, dict)
    assert completed.get("object") == "response"
    assert completed.get("status") in {"completed", "incomplete"}
    assert completed.get("id")
    output = completed.get("output")
    assert isinstance(output, list)
    assert output, (
        "non-empty stubbed assistant content must not emit empty "
        f"`response.output`; got {output!r}"
    )
    assert any(
        isinstance(item, dict)
        and (
            item.get("type") in {"message", "output_text"}
            or "PONG" in json.dumps(item)
        )
        for item in output
    ), f"completed output must include a message/output_text item: {output!r}"


@pytest.mark.asyncio
async def test_opencode_go_stream_true_empty_success_must_not_return_empty_json(
    monkeypatch: pytest.MonkeyPatch,
    configured_go_runtime,
) -> None:
    """Empty ox-alpha-free stream success must fail closed, not JSON output:[].

    Live curl of stream:true against alpha returned HTTP 200 JSON
    {\"output\":[], \"status\":\"completed\", \"usage\":{\"output_tokens\":0}}.
    Desired contract: raise aawm_codex_auto_agent_empty_success rather than
    returning that empty JSON 200 (or, if SSE is used, it still must not be
    empty JSON success).
    """
    try:
        response = (
            await _invoke_codex_opencode_go_adapter_route(
                monkeypatch,
                prepared_request_body={
                    "model": f"opencode-go/{_ADAPTER_MODEL}",
                    "input": "Reply with exactly the word PONG.",
                    "stream": True,
                },
                completion=_chat_completion_payload(
                    content="",
                    completion_id="chatcmpl-empty-alpha",
                    empty_choices=True,
                    prompt_tokens=0,
                    completion_tokens=0,
                ),
            )
        )[0]
    except ProxyException as exc:
        assert _empty_success_error_code(exc) == "aawm_codex_auto_agent_empty_success"
        return
    except Exception as exc:
        code = _empty_success_error_code(exc)
        assert code == "aawm_codex_auto_agent_empty_success", (
            "empty upstream stream success must raise "
            "aawm_codex_auto_agent_empty_success; got "
            f"{type(exc).__name__}: {exc!r}"
        )
        return

    media_type = str(getattr(response, "media_type", "") or "")
    body_text = await _collect_response_text(response)
    json_body: dict[str, Any] | None
    try:
        json_body = json.loads(body_text)
    except (TypeError, ValueError):
        json_body = None

    is_empty_json_success = (
        media_type.startswith("application/json")
        and isinstance(json_body, dict)
        and json_body.get("object") == "response"
        and list(json_body.get("output") or []) == []
    )
    assert not is_empty_json_success, (
        "empty stream success must not return HTTP JSON "
        "`object=response` with `output: []`; that is the live Ohmypi "
        f"incomplete-stream defect. body={body_text!r}"
    )
    pytest.fail(
        "empty ox-alpha-free stream success must raise "
        "aawm_codex_auto_agent_empty_success (fail-closed). "
        f"Got {type(response).__name__} media_type={media_type!r} "
        f"body={body_text!r}"
    )


@pytest.mark.asyncio
async def test_alias_candidate_provider_handler_dispatches_opencode_go() -> None:
    sentinel = Response(content=b"opencode-go", status_code=200)
    captured: dict[str, Any] = {}

    async def _handle_go(**kwargs: Any) -> Response:
        captured.update(kwargs)
        return sentinel

    host: dict[str, Any] = {
        "__builtins__": __builtins__,
        "_CODEX_AUTO_AGENT_OPENCODE_PROVIDER": "opencode_zen",
        "_CODEX_AUTO_AGENT_OPENCODE_GO_PROVIDER": "opencode_go",
        "_CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER": "kimi_code",
        "_CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER": "alibaba_token_plan",
        "_CODEX_AUTO_AGENT_ZAI_CODING_PLAN_PROVIDER": "zai_coding_plan",
        "_CODEX_AUTO_AGENT_OPENROUTER_PROVIDER": "openrouter",
        "_CODEX_AUTO_AGENT_XAI_PROVIDER": "xai",
        "_CODEX_AUTO_AGENT_COHERE_PROVIDER": "cohere",
        "_CODEX_AUTO_AGENT_CURSOR_AGENT_PROVIDER": "cursor_agent",
        "_handle_codex_alibaba_token_plan_adapter_route": AsyncMock(
            side_effect=AssertionError("alibaba must not win")
        ),
        "_handle_codex_kimi_chat_completions_adapter_route": AsyncMock(
            side_effect=AssertionError("kimi must not win")
        ),
        "_handle_codex_opencode_zen_adapter_route": AsyncMock(
            side_effect=AssertionError("opencode zen must not win")
        ),
        "_perform_codex_auto_agent_native_openai_request": AsyncMock(
            side_effect=AssertionError("native openai must not win")
        ),
        "_dispatch_auto_agent_alias_candidate_request": None,
    }

    async def _dispatch(*, candidate, provider_handlers, **kwargs: Any) -> Response:
        handler = provider_handlers[candidate["provider"]]
        return await handler()

    host["_dispatch_auto_agent_alias_candidate_request"] = _dispatch
    codex_candidate_calls.install(host)
    host["_handle_codex_opencode_go_adapter_route"] = _handle_go

    result = await host["_perform_codex_auto_agent_alias_candidate_request"](
        endpoint="/v1/responses",
        request=MagicMock(),
        fastapi_response=MagicMock(),
        user_api_key_dict=MagicMock(),
        candidate={
            "provider": "opencode_go",
            "model": _ADAPTER_MODEL,
            "route_family": _CODEX_GO_ROUTE_FAMILY,
        },
        candidate_body={"model": _ADAPTER_MODEL},
        target_url="https://chatgpt.com/backend-api/codex/responses",
        api_key=None,
        forward_headers=False,
    )

    assert result is sentinel
    assert captured["adapter_model"] == _ADAPTER_MODEL
    assert captured["use_alias_candidate_probe"] is True

    with pytest.raises(ValueError, match="codex_opencode_go_adapter"):
        await host["_perform_codex_auto_agent_alias_candidate_request"](
            endpoint="/v1/responses",
            request=MagicMock(),
            fastapi_response=MagicMock(),
            user_api_key_dict=MagicMock(),
            candidate={
                "provider": "opencode_go",
                "model": _ADAPTER_MODEL,
                "route_family": "codex_opencode_zen_adapter",
            },
            candidate_body={"model": _ADAPTER_MODEL},
            target_url="https://chatgpt.com/backend-api/codex/responses",
            api_key=None,
            forward_headers=False,
        )


def test_go_target_base_strips_v1_and_joins_chat_completions(
    configured_go_runtime,
) -> None:
    target_base = zen_runtime._get_opencode_go_target_base()
    assert target_base == "https://opencode.ai/zen/go"
    joined = zen_runtime._join_opencode_zen_passthrough_url(
        base_target_url=target_base,
        endpoint="/v1/chat/completions",
    )
    assert joined == _GO_CHAT_COMPLETIONS_URL
    assert "/zen/go/v1/" in joined
    assert joined != _ZEN_CHAT_COMPLETIONS_URL
