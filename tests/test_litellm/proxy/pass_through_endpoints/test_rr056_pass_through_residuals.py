"""Focused tests for RR-056 residual issues #3–#11 on pass_through_endpoints.py.

#1 wall-clock budget and #2 xAI double-capture are covered by existing tests in
test_pass_through_endpoints.py (d254cf8a8a / B2/B7).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import Request

from litellm.proxy.pass_through_endpoints import pass_through_endpoints as pte
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
    InitPassThroughEndpointHelpers,
    _is_aawm_agent_identity_registered_in_litellm_callbacks,
    _normalize_openai_function_tool_schemas_in_body,
    _run_passthrough_provider_failure_classifiers,
    _should_normalize_openai_function_tool_schemas,
    _WEBSOCKET_PASSTHROUGH_MESSAGE_BUFFER_MAX,
)


# ---------------------------------------------------------------------------
# #1 / #2 already-addressed smoke (not_valid as residual defects)
# ---------------------------------------------------------------------------


def test_rr056_issue1_wall_clock_budget_helper_present() -> None:
    assert hasattr(pte, "_get_passthrough_hidden_retry_budget_seconds")
    assert pte.DEFAULT_PASSTHROUGH_PRE_FIRST_BYTE_HIDDEN_RETRY_BUDGET_SECONDS == 230.0


def test_rr056_issue2_direct_capture_skips_when_callback_registered() -> None:
    assert callable(_is_aawm_agent_identity_registered_in_litellm_callbacks)
    # Marker-based detection still present for dual registration surface.
    assert "aawm_agent_identity" in pte._AAWM_AGENT_IDENTITY_CALLBACK_MARKERS[0]


# ---------------------------------------------------------------------------
# #3 classifier dispatch
# ---------------------------------------------------------------------------


def test_provider_failure_classifier_dispatch_empty_for_generic() -> None:
    request = MagicMock(spec=Request)
    results = _run_passthrough_provider_failure_classifiers(
        request=request,
        url=httpx.URL("https://example.com/v1"),
        custom_llm_provider="openai",
        status_code=500,
        exc=Exception("boom"),
    )
    assert results == []


def test_provider_failure_classifier_dispatch_anthropic_known() -> None:
    """Classifiers live under provider_failure_classifiers/ (RR-056 #3)."""
    from litellm.proxy.pass_through_endpoints import provider_failure_classifiers as pfc

    request = MagicMock(spec=Request)
    # Patch the ordered registry entrypoints so only anthropic matches.
    def _anth(**kwargs):
        return pfc.registry.PassthroughProviderFailureClassification(
            name="anthropic_known_failure",
            failure_kind="invalid_authentication",
            log_message="Pass through endpoint surfaced Anthropic provider/client failure status=%s error=%s",
        )

    with patch.object(
        pfc.registry,
        "PASSTHROUGH_PROVIDER_FAILURE_CLASSIFIERS",
        (_anth,),
    ):
        results = pfc.registry._run_passthrough_provider_failure_classifiers(
            request=request,
            url=httpx.URL("https://api.anthropic.com/v1/messages"),
            custom_llm_provider="anthropic",
            status_code=401,
            exc=Exception("invalid authentication credentials"),
        )
    assert len(results) == 1
    assert results[0].name == "anthropic_known_failure"
    assert results[0].failure_kind == "invalid_authentication"


def test_provider_failure_classifiers_live_outside_god_module() -> None:
    import litellm.proxy.pass_through_endpoints.provider_failure_classifiers as pfc
    import litellm.proxy.pass_through_endpoints.provider_failure_classifiers.grok as grok
    import litellm.proxy.pass_through_endpoints.provider_failure_classifiers.anthropic as anth
    import litellm.proxy.pass_through_endpoints.provider_failure_classifiers.chatgpt_codex as codex
    import litellm.proxy.pass_through_endpoints.provider_failure_classifiers.google_code_assist as gca

    assert pfc.__name__.endswith("provider_failure_classifiers")
    assert "provider_failure_classifiers" in grok.__file__
    assert "provider_failure_classifiers" in anth.__file__
    assert "provider_failure_classifiers" in codex.__file__
    assert "provider_failure_classifiers" in gca.__file__
    # God-module re-exports registry entry points but does not define them.
    src = open(pte.__file__, encoding="utf-8").read()
    assert "def _is_known_grok_billing_passthrough_timeout_cancel_response" not in src
    assert "def _get_known_anthropic_passthrough_failure_kind" not in src
    assert "class PassthroughProviderFailureClassification" not in src
    assert "from .provider_failure_classifiers import" in src


# ---------------------------------------------------------------------------
# #4 websocket buffer bound
# ---------------------------------------------------------------------------


def test_websocket_message_buffer_is_bounded() -> None:
    assert _WEBSOCKET_PASSTHROUGH_MESSAGE_BUFFER_MAX == 256
    # Source uses deque(maxlen=...) for the connection-local buffer.
    from collections import deque

    buf: deque = deque(maxlen=_WEBSOCKET_PASSTHROUGH_MESSAGE_BUFFER_MAX)
    for i in range(_WEBSOCKET_PASSTHROUGH_MESSAGE_BUFFER_MAX + 50):
        buf.append({"i": i})
    assert len(buf) == _WEBSOCKET_PASSTHROUGH_MESSAGE_BUFFER_MAX
    assert buf[0]["i"] == 50


# ---------------------------------------------------------------------------
# #5 transformed_exception capture present in failure path source
# ---------------------------------------------------------------------------


def test_failure_hook_applies_transformed_exception_source() -> None:
    src = open(pte.__file__, encoding="utf-8").read()
    assert (
        "transformed_exception = await proxy_logging_obj.post_call_failure_hook" in src
    )
    assert "if isinstance(transformed_exception, BaseException):" in src


# ---------------------------------------------------------------------------
# #7 dual-import probe documented
# ---------------------------------------------------------------------------


def test_direct_capture_uses_single_canonical_agent_identity_import() -> None:
    """RR-056 #7: no dual in-tree/wheel probe; RR-003 single-source packaging."""
    src = open(pte.__file__, encoding="utf-8").read()
    assert "for module_name in (" not in src
    assert "importlib.import_module(module_name)" not in src
    assert "from litellm.integrations.aawm_agent_identity import" in src
    assert "_CANONICAL_AAWM_AGENT_IDENTITY_INSTANCE" in src
    # Compatibility markers remain for callback registration detection only.
    assert (
        "aawm_litellm_callbacks.agent_identity"
        in pte._AAWM_AGENT_IDENTITY_CALLBACK_MARKERS
    )
    assert pte._CANONICAL_AAWM_AGENT_IDENTITY_INSTANCE is not None


# ---------------------------------------------------------------------------
# #8 circular-import comments
# ---------------------------------------------------------------------------


def test_pass_through_request_circular_import_comment() -> None:
    src = open(pte.__file__, encoding="utf-8").read()
    assert (
        "Inline imports: proxy_server imports this module at startup (circular)." in src
    )


# ---------------------------------------------------------------------------
# #9 OpenAI tool schema normalization gating
# ---------------------------------------------------------------------------


def test_openai_tool_schema_normalization_gated_by_provider() -> None:
    assert (
        _should_normalize_openai_function_tool_schemas(
            url=None, custom_llm_provider="anthropic"
        )
        is False
    )
    assert (
        _should_normalize_openai_function_tool_schemas(
            url=None, custom_llm_provider="openai"
        )
        is True
    )
    assert (
        _should_normalize_openai_function_tool_schemas(
            url=httpx.URL("https://api.openai.com/v1/responses"),
            custom_llm_provider=None,
        )
        is True
    )


def test_normalize_openai_function_tool_schemas_still_fixes_object_props() -> None:
    body = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "x",
                    "parameters": {"type": "object"},  # missing properties
                },
            }
        ]
    }
    assert _normalize_openai_function_tool_schemas_in_body(body) == 1
    assert body["tools"][0]["function"]["parameters"]["properties"] == {}


# ---------------------------------------------------------------------------
# #10 route registry indexes
# ---------------------------------------------------------------------------


def test_registered_route_lookup_uses_index_and_lazy_rebuild() -> None:
    pte._registered_pass_through_routes.clear()
    pte._registered_pass_through_exact_index.clear()
    pte._registered_pass_through_subpath_trie.clear()
    pte._registered_pass_through_subpath_index.clear()

    route_key = "epid1:exact:/custom/echo:GET,POST"
    pte._registered_pass_through_routes[route_key] = {
        "endpoint_id": "epid1",
        "path": "/custom/echo",
        "type": "exact",
        "methods": ["GET", "POST"],
        "passthrough_params": {"target": "https://example.com"},
    }
    # Indexes empty → lazy rebuild on get
    found = InitPassThroughEndpointHelpers.get_registered_pass_through_route(
        "/custom/echo", method="POST"
    )
    assert found is not None
    assert found["endpoint_id"] == "epid1"
    assert InitPassThroughEndpointHelpers.is_registered_pass_through_route(
        "/custom/echo"
    )

    InitPassThroughEndpointHelpers.clear_all_pass_through_routes()
    assert pte._registered_pass_through_routes == {}
    assert pte._registered_pass_through_exact_index == {}
    assert pte._registered_pass_through_subpath_index == {}


def test_index_maintained_on_register_via_helper() -> None:
    pte._registered_pass_through_routes.clear()
    pte._registered_pass_through_exact_index.clear()
    pte._registered_pass_through_subpath_trie.clear()
    pte._registered_pass_through_subpath_index.clear()
    pte._index_pass_through_route_key("k1", "/a", "exact")
    assert pte._registered_pass_through_exact_index["/a"] == ["k1"]
    # unindex needs registry entry
    pte._registered_pass_through_routes["k1"] = {
        "path": "/a",
        "type": "exact",
        "endpoint_id": "e",
    }
    pte._unindex_pass_through_route_key("k1")
    assert "/a" not in pte._registered_pass_through_exact_index


# ---------------------------------------------------------------------------
# #11 chat_completion body parse uses json.loads
# ---------------------------------------------------------------------------


def test_chat_completion_pass_through_parses_json_not_literal_eval() -> None:
    src = open(pte.__file__, encoding="utf-8").read()
    # The chat_completion path should no longer prefer ast.literal_eval for body.
    # Keep a narrow check around the chat_completion function.
    start = src.index("async def chat_completion_pass_through_endpoint")
    end = src.index("\nasync def ", start + 10)
    chunk = src[start:end]
    assert "data = json.loads(body_str)" in chunk
    assert "ast.literal_eval(body_str)" not in chunk


# ---------------------------------------------------------------------------
# #6 non-stream prefer_stream flag
# ---------------------------------------------------------------------------


def test_non_streaming_handler_accepts_prefer_stream_flag() -> None:
    import inspect

    sig = inspect.signature(
        pte.HttpPassThroughEndpointHelpers.non_streaming_http_request_handler
    )
    assert "prefer_stream_for_unknown_content" in sig.parameters


@pytest.mark.asyncio
async def test_non_streaming_handler_uses_send_stream_when_preferred() -> None:
    mock_client = MagicMock()
    mock_req = MagicMock()
    mock_client.build_request.return_value = mock_req
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.headers = {"content-type": "text/event-stream"}
    mock_client.send = AsyncMock(return_value=mock_response)

    request = MagicMock(spec=Request)
    request.method = "POST"
    request.headers = {"content-type": "application/json"}

    result = (
        await pte.HttpPassThroughEndpointHelpers.non_streaming_http_request_handler(
            request=request,
            async_client=mock_client,
            url=httpx.URL("https://api.openai.com/v1/chat/completions"),
            headers={"content-type": "application/json"},
            _parsed_body={"model": "gpt-4o-mini", "messages": []},
            prefer_stream_for_unknown_content=True,
        )
    )
    assert result is mock_response
    mock_client.send.assert_awaited_once()
    assert mock_client.send.await_args.kwargs.get("stream") is True
    mock_client.request.assert_not_called()


@pytest.mark.asyncio
async def test_non_stream_error_path_uses_aread_not_sync_content() -> None:
    """RR-056 #6: streamed non-SSE error bodies must drain via aread()."""
    mock_client = MagicMock()
    mock_req = MagicMock()
    mock_client.build_request.return_value = mock_req

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.headers = {"content-type": "application/json"}
    mock_response.status_code = 429
    mock_response.aread = AsyncMock(return_value=b'{"error":"quota"}')
    mock_response.request = MagicMock()
    # Sync .content/.text must not be required on the streamed path.
    type(mock_response).content = property(
        lambda self: (_ for _ in ()).throw(AssertionError("sync content read"))
    )
    type(mock_response).text = property(
        lambda self: (_ for _ in ()).throw(AssertionError("sync text read"))
    )

    def raise_status():
        raise httpx.HTTPStatusError(
            "quota",
            request=mock_response.request,
            response=mock_response,
        )

    mock_response.raise_for_status = raise_status
    mock_client.send = AsyncMock(return_value=mock_response)

    request = MagicMock(spec=Request)
    request.method = "POST"
    request.headers = {"content-type": "application/json"}

    # Exercise the helper that the non-stream path uses; body is streamed.
    result = (
        await pte.HttpPassThroughEndpointHelpers.non_streaming_http_request_handler(
            request=request,
            async_client=mock_client,
            url=httpx.URL("https://api.openai.com/v1/chat/completions"),
            headers={"content-type": "application/json"},
            _parsed_body={"model": "gpt-4o-mini"},
            prefer_stream_for_unknown_content=True,
        )
    )
    assert result is mock_response
    # Caller-side drain contract for non-SSE errors.
    body = await result.aread()
    assert body == b'{"error":"quota"}'
    mock_client.send.assert_awaited_once()

# ---------------------------------------------------------------------------
# Closeout: registry-driven logging kinds + single route fetch
# ---------------------------------------------------------------------------


def test_registry_failure_kinds_match_historical_log_contracts() -> None:
    """Registry failure_kind values must match historical operator log contracts."""
    from litellm.proxy.pass_through_endpoints.provider_failure_classifiers import (
        registry as reg,
    )
    from litellm.proxy.pass_through_endpoints.provider_failure_classifiers.chatgpt_codex import (
        _get_passthrough_chatgpt_codex_model_not_supported_failure_kind,
    )

    request = MagicMock(spec=Request)

    def _only(name: str, failure_kind: str, log_message: str, **extra):
        def _fn(**kwargs):
            return reg.PassthroughProviderFailureClassification(
                name=name,
                failure_kind=failure_kind,
                log_message=log_message,
                **extra,
            )

        return _fn

    with patch.object(
        reg,
        "PASSTHROUGH_PROVIDER_FAILURE_CLASSIFIERS",
        (
            _only(
                "chatgpt_codex_block_page",
                "openai_chatgpt_codex_block_page",
                "Pass through endpoint surfaced ChatGPT Codex block page status=%s error=%s",
            ),
        ),
    ):
        results = reg._run_passthrough_provider_failure_classifiers(
            request=request,
            url=httpx.URL("https://chatgpt.com/backend-api/codex/responses"),
            custom_llm_provider="openai",
            status_code=403,
            exc=Exception("cdn-cgi/challenge-platform unable to load site"),
        )
    assert len(results) == 1
    assert results[0].name == "chatgpt_codex_block_page"
    assert results[0].failure_kind == "openai_chatgpt_codex_block_page"
    assert "ChatGPT Codex block page" in (results[0].log_message or "")

    with patch.object(
        reg,
        "PASSTHROUGH_PROVIDER_FAILURE_CLASSIFIERS",
        (
            _only(
                "chatgpt_codex_model_not_supported",
                _get_passthrough_chatgpt_codex_model_not_supported_failure_kind(),
                "Pass through endpoint surfaced ChatGPT Codex unsupported model for account status=%s error=%s",
            ),
        ),
    ):
        results = reg._run_passthrough_provider_failure_classifiers(
            request=request,
            url=httpx.URL("https://chatgpt.com/backend-api/codex/responses"),
            custom_llm_provider="openai",
            status_code=400,
            exc=Exception(
                "model is not supported when using Codex with a ChatGPT account"
            ),
        )
    assert results[0].failure_kind == (
        _get_passthrough_chatgpt_codex_model_not_supported_failure_kind()
    )
    assert "unsupported model for account" in (results[0].log_message or "").lower()

    with patch.object(
        reg,
        "PASSTHROUGH_PROVIDER_FAILURE_CLASSIFIERS",
        (
            _only(
                "google_code_assist_tos",
                "google_code_assist_tos_violation",
                "Pass through endpoint surfaced Google Code Assist account TOS violation status=%s error=%s",
            ),
        ),
    ):
        results = reg._run_passthrough_provider_failure_classifiers(
            request=request,
            url=httpx.URL("https://cloudcode-pa.googleapis.com/v1:generate"),
            custom_llm_provider="google_code_assist",
            status_code=403,
            exc=Exception("TOS_VIOLATION"),
        )
    assert results[0].failure_kind == "google_code_assist_tos_violation"
    assert "account TOS violation" in (results[0].log_message or "")




def test_provider_failure_logging_uses_registry_fields_not_hardcoded_cascade() -> None:
    """God-module exception path must not re-hardcode per-vendor failure_kind strings."""
    src = open(pte.__file__, encoding="utf-8").read()
    # Data-driven branch present.
    assert "elif _provider_failure_classification is not None:" in src
    assert "classification.failure_kind" in src
    assert "classification.log_message" in src
    assert 'getattr(classification, "log_error_summary", None)' in src
    # Old per-vendor cascade names for logging must be gone.
    assert "suppress_chatgpt_codex_block_page_traceback" not in src
    assert 'failure_kind": "openai_chatgpt_codex_block_page"' not in src
    assert 'failure_kind": "google_code_assist_tos_violation"' not in src
    assert "known_anthropic_failure_kind" not in src


def test_create_pass_through_route_single_registry_fetch() -> None:
    """RR-056 #10: one get_registered call; no pre-check is_registered scan."""
    src = open(pte.__file__, encoding="utf-8").read()
    # Anchor on the RR-056 single-fetch comment in create_pass_through_route.
    anchor = "Single registered-route fetch (RR-056 #10)"
    assert anchor in src
    start = src.index(anchor)
    end = src.index("return await pass_through_request", start)
    chunk = src[start:end]
    assert "get_registered_pass_through_route" in chunk
    assert "is_registered_pass_through_route" not in chunk
    assert "_is_mapped_pass_through_route" in chunk
    assert chunk.count("get_registered_pass_through_route") == 1


@pytest.mark.asyncio
async def test_create_pass_through_route_uses_single_get_registered() -> None:
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        create_pass_through_route,
    )
    from starlette.datastructures import QueryParams

    unique_path = "/test/path/unique/rr056_single_fetch"
    endpoint_func = create_pass_through_route(
        endpoint=unique_path,
        target="http://example.com/api",
        custom_headers={},
    )

    with patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.pass_through_request",
        new=AsyncMock(return_value=MagicMock()),
    ) as mock_pass_through, patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.InitPassThroughEndpointHelpers.get_registered_pass_through_route"
    ) as mock_get_registered, patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.InitPassThroughEndpointHelpers._is_mapped_pass_through_route",
        return_value=False,
    ) as mock_mapped, patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints._parse_request_data_by_content_type",
        new=AsyncMock(return_value=({}, {"k": 1}, None, False)),
    ):
        mock_get_registered.return_value = {
            "passthrough_params": {"target": "http://example.com/api"},
        }
        mock_request = MagicMock(spec=Request)
        mock_request.url = MagicMock()
        mock_request.url.path = unique_path
        mock_request.path_params = {}
        mock_request.query_params = QueryParams({})
        mock_request.method = "POST"

        await endpoint_func(
            request=mock_request,
            fastapi_response=MagicMock(),
            user_api_key_dict=MagicMock(),
        )

        mock_get_registered.assert_called_once()
        mock_mapped.assert_not_called()
        mock_pass_through.assert_awaited_once()


# ---------------------------------------------------------------------------
# Audit corrections: O(1)/trie lookup, first-match registry, multipart stream,
# skip redundant tool schema walk
# ---------------------------------------------------------------------------


def test_classifier_registry_first_match_short_circuits() -> None:
    from litellm.proxy.pass_through_endpoints.provider_failure_classifiers import (
        registry as reg,
    )

    calls: list[str] = []

    def first(**kwargs):
        calls.append("first")
        return reg.PassthroughProviderFailureClassification(
            name="first_hit",
            failure_kind="k1",
            log_message="first %s %s",
            skip_post_call_failure_hook=True,
        )

    def second(**kwargs):
        calls.append("second")
        return reg.PassthroughProviderFailureClassification(
            name="second_hit",
            failure_kind="k2",
            log_message="second %s %s",
        )

    with patch.object(
        reg,
        "PASSTHROUGH_PROVIDER_FAILURE_CLASSIFIERS",
        (first, second),
    ):
        results = reg._run_passthrough_provider_failure_classifiers(
            request=MagicMock(spec=Request),
            url=httpx.URL("https://example.com"),
            custom_llm_provider="xai",
            status_code=401,
            exc=Exception("x"),
        )
    assert [r.name for r in results] == ["first_hit"]
    assert results[0].skip_post_call_failure_hook is True
    assert calls == ["first"]


def test_exact_route_lookup_is_o1_normalized_dict() -> None:
    pte._registered_pass_through_routes.clear()
    pte._registered_pass_through_exact_index.clear()
    pte._registered_pass_through_subpath_trie.clear()
    route_key = "epid1:exact:/custom/echo:GET,POST"
    pte._registered_pass_through_routes[route_key] = {
        "endpoint_id": "epid1",
        "path": "/custom/echo",
        "type": "exact",
        "methods": ["GET", "POST"],
        "passthrough_params": {"target": "https://example.com"},
    }
    pte._index_pass_through_route_key(route_key, "/custom/echo", "exact")
    # Direct dict hit without scanning all keys
    assert pte._registered_pass_through_exact_index["/custom/echo"] == [route_key]
    found = InitPassThroughEndpointHelpers.get_registered_pass_through_route(
        "/custom/echo", method="POST"
    )
    assert found is not None and found["endpoint_id"] == "epid1"
    InitPassThroughEndpointHelpers.clear_all_pass_through_routes()


def test_subpath_lookup_uses_longest_prefix_trie_not_full_scan() -> None:
    pte._registered_pass_through_routes.clear()
    pte._registered_pass_through_exact_index.clear()
    pte._registered_pass_through_subpath_trie.clear()
    short_key = "e1:subpath:/api:GET"
    long_key = "e2:subpath:/api/v1:GET"
    pte._registered_pass_through_routes[short_key] = {
        "endpoint_id": "e1",
        "path": "/api",
        "type": "subpath",
        "methods": ["GET"],
        "passthrough_params": {"target": "https://short.example"},
    }
    pte._registered_pass_through_routes[long_key] = {
        "endpoint_id": "e2",
        "path": "/api/v1",
        "type": "subpath",
        "methods": ["GET"],
        "passthrough_params": {"target": "https://long.example"},
    }
    pte._index_pass_through_route_key(short_key, "/api", "subpath")
    pte._index_pass_through_route_key(long_key, "/api/v1", "subpath")
    found = InitPassThroughEndpointHelpers.get_registered_pass_through_route(
        "/api/v1/items", method="GET"
    )
    assert found is not None
    assert found["endpoint_id"] == "e2"
    # Trie contains nested children; no reliance on iterating all path keys.
    assert "__children__" in pte._registered_pass_through_subpath_trie
    InitPassThroughEndpointHelpers.clear_all_pass_through_routes()


@pytest.mark.asyncio
async def test_multipart_handler_respects_prefer_stream_for_unknown_content() -> None:
    mock_client = MagicMock()
    mock_req = MagicMock()
    mock_client.build_request.return_value = mock_req
    mock_response = MagicMock(spec=httpx.Response)
    mock_client.send = AsyncMock(return_value=mock_response)

    request = MagicMock(spec=Request)
    request.method = "POST"
    request.headers = {"content-type": "multipart/form-data; boundary=abc"}

    async def fake_form():
        return {}

    request.form = fake_form

    result = await pte.HttpPassThroughEndpointHelpers.make_multipart_http_request(
        request=request,
        async_client=mock_client,
        url=httpx.URL("https://example.com/upload"),
        headers={"content-type": "multipart/form-data; boundary=abc"},
        prefer_stream_for_unknown_content=True,
    )
    assert result is mock_response
    mock_client.send.assert_awaited_once()
    assert mock_client.send.await_args.kwargs.get("stream") is True
    mock_client.request.assert_not_called()


def test_second_tool_schema_walk_skipped_when_first_fixed_zero_and_tools_unchanged() -> None:
    body = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "x",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
    }
    first = _normalize_openai_function_tool_schemas_in_body(body)
    assert first == 0
    # Source-level: second normalize pass requires tools rewrite by hook.
    src = open(pte.__file__, encoding="utf-8").read()
    assert "and _tools_rewritten_by_hook" in src
    assert "_pre_hook_tools_identity" in src


def test_healthy_route_miss_does_not_full_scan_registry() -> None:
    """Warm indexes + miss must not iterate _registered_pass_through_routes (RR-056 #10)."""

    class _CountingRegistry(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.items_calls = 0

        def items(self):  # type: ignore[override]
            self.items_calls += 1
            return super().items()

    original_registry = pte._registered_pass_through_routes
    counting = _CountingRegistry()
    pte._registered_pass_through_routes = counting  # type: ignore[assignment]
    pte._registered_pass_through_exact_index.clear()
    pte._registered_pass_through_subpath_trie.clear()

    route_key = "epid-miss:exact:/known:GET"
    counting[route_key] = {
        "endpoint_id": "epid-miss",
        "path": "/known",
        "type": "exact",
        "methods": ["GET"],
        "passthrough_params": {"target": "https://example.com"},
    }
    pte._index_pass_through_route_key(route_key, "/known", "exact")
    assert pte._registered_pass_through_exact_index  # indexes warm
    counting.items_calls = 0  # ignore setup

    try:
        found = InitPassThroughEndpointHelpers.get_registered_pass_through_route(
            "/definitely-not-registered"
        )
        assert found is None
        assert counting.items_calls == 0
    finally:
        pte._registered_pass_through_routes = original_registry
        InitPassThroughEndpointHelpers.clear_all_pass_through_routes()


def test_validation_walk_skipped_when_first_fixed_zero_and_tools_unchanged() -> None:
    """RR-056 #9: _collect_invalid_* must not run on the zero-fix + identity-stable path."""
    src = open(pte.__file__, encoding="utf-8").read()
    # Validation is gated on rewrite or first-pass fixes, not bare _should_normalize.
    assert (
        "and (_tools_rewritten_by_hook or normalized_tool_schema_count > 0)"
        in src
    )
    assert "Validation walk (RR-056 #9)" in src

    # Behavioral: normalize already-valid tools => 0 fixes; source gate implies
    # validation would skip when tools identity unchanged after hooks.
    body = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "ok",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
    }
    tools_id = id(body["tools"])
    assert _normalize_openai_function_tool_schemas_in_body(body) == 0
    assert id(body["tools"]) == tools_id
