import os
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException, Request, Response
from starlette.datastructures import Headers, QueryParams

import litellm
from litellm.integrations import aawm_agent_identity
from litellm.integrations.aawm_agent_identity import (
    _build_rate_limit_observations,
    _build_session_history_record,
)
from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
    grok_proxy_route,
)
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
    HttpPassThroughEndpointHelpers,
)
from litellm.proxy.pass_through_endpoints.streaming_handler import (
    PassThroughStreamingHandler,
)
from litellm.proxy.pass_through_endpoints.success_handler import (
    PassThroughEndpointLogging,
)
from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType


class GrokBuildPassthroughHarness:
    """Deterministic offline harness for native Grok Build pass-through."""

    upstream_base = "https://cli-chat-proxy.grok.com"
    model = "grok-build"
    session_id = "grok-harness-session"
    user_id = "user_harness"
    litellm_api_key = "litellm-harness-key"

    def headers(self, *, content_type: str = "application/json") -> Headers:
        return Headers(
            {
                "authorization": "Bearer oidc-token",
                "x-litellm-api-key": self.litellm_api_key,
                "x-xai-token-auth": "xai-grok-cli",
                "x-grok-agent-id": "agent_harness",
                "x-grok-client-version": "0.1.210",
                "x-grok-conv-id": "conv_harness",
                "x-grok-model-override": self.model,
                "x-grok-req-id": "req_harness",
                "x-grok-session-id": self.session_id,
                "x-grok-turn-idx": "1",
                "x-grok-user-id": self.user_id,
                "x-email": "user@example.com",
                "x-teamid": "team_harness",
                "x-userid": self.user_id,
                "user-agent": "grok/0.1.210",
                "content-type": content_type,
            }
        )

    def request(
        self,
        *,
        method: str = "POST",
        endpoint: str = "v1/responses",
        content_type: str = "application/json",
        query_params: dict[str, str] | None = None,
    ) -> MagicMock:
        request = MagicMock(spec=Request)
        request.method = method
        request.url = f"http://localhost:4000/grok/{endpoint}"
        request.headers = self.headers(content_type=content_type)
        request.query_params = QueryParams(query_params or {"debug": "1"})
        request.cookies = {}
        return request

    def request_body(self, *, stream: bool = False) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "input": "Return pong.",
        }
        if stream:
            body["stream"] = True
        return body

    def logging_kwargs(
        self,
        *,
        litellm_call_id: str = "call-grok-harness",
        url: str = "https://cli-chat-proxy.grok.com/v1/responses",
        response_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "model": self.model,
            "custom_llm_provider": "xai",
            "call_type": "pass_through_endpoint",
            "litellm_call_id": litellm_call_id,
            "litellm_params": {
                "metadata": {
                    "client_name": "grok-build",
                    "client_version": "0.1.210",
                    "grok_cli_chat_proxy": True,
                    "grok_model_override": self.model,
                    "model_group": self.model,
                    "passthrough_route_family": "grok_cli_chat_proxy",
                    "session_id": self.session_id,
                    "tags": ["grok-build", "route:grok_cli_chat_proxy"],
                },
                "proxy_server_request": {
                    "headers": dict(self.headers()),
                    "body": self.request_body(),
                },
            },
            "standard_logging_object": {"metadata": {}, "request_tags": []},
            "passthrough_logging_payload": {
                "url": url,
                "request_body": self.request_body(),
                "request_headers": dict(self.headers()),
                "response_body": response_body or {},
            },
        }

    async def invoke_route(
        self,
        *,
        endpoint: str = "v1/responses",
        request_body: dict[str, Any] | None = None,
        content_type: str = "application/json",
        query_params: dict[str, str] | None = None,
    ) -> tuple[dict[str, Any], AsyncMock, AsyncMock, AsyncMock]:
        request = self.request(
            endpoint=endpoint,
            content_type=content_type,
            query_params=query_params,
        )
        auth_mock = AsyncMock(return_value=MagicMock())
        get_body_mock = AsyncMock(return_value=request_body or self.request_body())
        pass_through_mock = AsyncMock(return_value={"ok": True})

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.user_api_key_auth",
            auth_mock,
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_request_body",
            get_body_mock,
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.pass_through_request",
            pass_through_mock,
        ):
            result = await grok_proxy_route(
                endpoint=endpoint,
                request=request,
                fastapi_response=MagicMock(spec=Response),
            )

        assert result == {"ok": True}
        return pass_through_mock.await_args.kwargs, auth_mock, get_body_mock, pass_through_mock

    def forwarded_headers(self, call_kwargs: dict[str, Any]) -> dict[str, Any]:
        return HttpPassThroughEndpointHelpers.forward_headers_from_request(
            request_headers=dict(self.headers()),
            headers={},
            forward_headers=True,
            allowed_forward_headers=call_kwargs["allowed_forward_headers"],
        )

    def final_response_body(self) -> dict[str, Any]:
        return {
            "id": "resp_grok_harness",
            "object": "response",
            "model": self.model,
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "pong"}],
                }
            ],
            "usage": {
                "input_tokens": 11,
                "output_tokens": 2,
                "total_tokens": 13,
                "input_tokens_details": {"cached_tokens": 3},
            },
        }


@pytest.mark.asyncio
async def test_grok_build_harness_routes_json_headers_and_filters_litellm_auth():
    harness = GrokBuildPassthroughHarness()

    call_kwargs, auth_mock, _, _ = await harness.invoke_route(
        endpoint="v1/responses",
        request_body=harness.request_body(),
        query_params={"key": "query-litellm-key", "debug": "1"},
    )

    assert auth_mock.await_args.kwargs["api_key"] == f"Bearer {harness.litellm_api_key}"
    assert call_kwargs["target"] == f"{harness.upstream_base}/v1/responses"
    assert call_kwargs["custom_llm_provider"] == litellm.LlmProviders.XAI.value
    assert call_kwargs["egress_credential_family"] == "xai"
    assert call_kwargs["expected_target_family"] == "xai"
    assert call_kwargs["query_params"] == {"debug": "1"}
    assert "authorization" in call_kwargs["allowed_forward_headers"]
    assert "x-xai-token-auth" in call_kwargs["allowed_forward_headers"]
    assert "x-grok-client-version" in call_kwargs["allowed_forward_headers"]
    assert "x-grok-model-override" in call_kwargs["allowed_forward_headers"]
    assert "x-litellm-api-key" not in call_kwargs["allowed_forward_headers"]

    forwarded_headers = harness.forwarded_headers(call_kwargs)
    assert forwarded_headers["authorization"] == "Bearer oidc-token"
    assert forwarded_headers["x-xai-token-auth"] == "xai-grok-cli"
    assert forwarded_headers["x-grok-model-override"] == harness.model
    assert "x-litellm-api-key" not in forwarded_headers

    metadata = call_kwargs["custom_body"]["litellm_metadata"]
    assert metadata["client_name"] == "grok-build"
    assert metadata["grok_model_override"] == harness.model
    assert metadata["model_group"] == harness.model
    assert metadata["passthrough_route_family"] == "grok_cli_chat_proxy"
    assert metadata["session_id"] == harness.session_id
    assert call_kwargs["passthrough_logging_metadata"] == metadata


@pytest.mark.asyncio
async def test_grok_build_harness_routes_protobuf_raw_body_without_json_parse():
    harness = GrokBuildPassthroughHarness()

    call_kwargs, _, get_body_mock, _ = await harness.invoke_route(
        endpoint="v1/traces",
        content_type="application/x-protobuf",
        query_params={},
    )

    get_body_mock.assert_not_awaited()
    assert call_kwargs["target"] == f"{harness.upstream_base}/v1/traces"
    assert call_kwargs["custom_body"] is None
    assert call_kwargs["raw_body_passthrough"] is True
    metadata = call_kwargs["passthrough_logging_metadata"]
    assert metadata["client_name"] == "grok-build"
    assert metadata["passthrough_route_family"] == "grok_cli_chat_proxy"
    assert metadata["grok_cli_chat_proxy"] is True
    assert "route:grok_cli_chat_proxy" in metadata["tags"]


def test_grok_build_harness_normalizes_final_response_and_session_history_identity():
    harness = GrokBuildPassthroughHarness()
    response_body = harness.final_response_body()
    request_body = harness.request_body()
    httpx_response = httpx.Response(
        200,
        json=response_body,
        request=httpx.Request("POST", f"{harness.upstream_base}/v1/responses"),
    )
    logging_obj = MagicMock()
    logging_obj.model_call_details = {}
    logging_obj.litellm_call_id = "call-grok-harness-final"
    logging_obj.call_type = "pass_through_endpoint"
    logging_kwargs = harness.logging_kwargs(
        litellm_call_id="call-grok-harness-final",
        response_body=response_body,
    )
    custom_llm_provider = logging_kwargs.pop("custom_llm_provider")

    normalized = PassThroughEndpointLogging().normalize_llm_passthrough_logging_payload(
        httpx_response=httpx_response,
        response_body=response_body,
        request_body=request_body,
        logging_obj=logging_obj,
        url_route=f"{harness.upstream_base}/v1/responses",
        result="",
        start_time=datetime(2026, 6, 1, 19, 50, tzinfo=timezone.utc),
        end_time=datetime(2026, 6, 1, 19, 50, 1, tzinfo=timezone.utc),
        cache_hit=False,
        custom_llm_provider=custom_llm_provider,
        **logging_kwargs,
    )

    result = normalized["standard_logging_response_object"]
    assert result is not None
    assert result.model == harness.model
    assert result.usage.total_tokens == 13
    assert normalized["kwargs"]["custom_llm_provider"] == "xai"
    assert normalized["kwargs"]["model"] == harness.model
    assert normalized["kwargs"]["response_cost"] > 0

    record = _build_session_history_record(
        kwargs=normalized["kwargs"],
        result=result,
        start_time="2026-06-01T19:50:00Z",
        end_time="2026-06-01T19:50:01Z",
    )

    assert record is not None
    assert record["provider"] == "xai"
    assert record["model"] == "grok-build"
    assert record["model_group"] == "grok-build"
    assert record["client_name"] == "grok-build"
    assert record["client_version"] == "0.1.210"
    assert record["session_id"] == harness.session_id
    assert record["input_tokens"] == 11
    assert record["output_tokens"] == 2
    assert record["response_cost_usd"] is not None
    assert record["response_cost_usd"] > 0
    assert record["metadata"]["passthrough_route_family"] == "grok_cli_chat_proxy"


@pytest.mark.asyncio
async def test_grok_build_harness_routes_streaming_final_response_to_logging():
    harness = GrokBuildPassthroughHarness()
    logging_obj = MagicMock()
    logging_obj.model_call_details = {
        "custom_llm_provider": "xai",
        "litellm_params": harness.logging_kwargs()["litellm_params"],
        "passthrough_logging_payload": harness.logging_kwargs()[
            "passthrough_logging_payload"
        ],
    }
    logging_obj.async_success_handler = AsyncMock()
    logging_obj.success_handler = MagicMock()
    logging_obj._should_run_sync_callbacks_for_async_calls.return_value = False
    success_handler_kwargs = harness.logging_kwargs(
        litellm_call_id="call-grok-harness-stream"
    )

    await PassThroughStreamingHandler._route_streaming_logging_to_handler(
        litellm_logging_obj=logging_obj,
        passthrough_success_handler_obj=PassThroughEndpointLogging(),
        response=httpx.Response(
            200,
            request=httpx.Request("POST", f"{harness.upstream_base}/v1/responses"),
        ),
        url_route=f"{harness.upstream_base}/v1/responses",
        request_body=harness.request_body(stream=True),
        endpoint_type=EndpointType.OPENAI,
        start_time=datetime.now() - timedelta(milliseconds=20),
        raw_bytes=[
            b'data: {"type":"response.output_text.delta","delta":"pong","output_index":0}\n\n',
            b"data: [DONE]\n\n",
        ],
        end_time=datetime.now(),
        custom_llm_provider="xai",
        success_handler_kwargs=success_handler_kwargs,
        local_prepare_ms=2.5,
    )

    logging_obj.async_success_handler.assert_awaited_once()
    call_kwargs = logging_obj.async_success_handler.await_args.kwargs
    result = call_kwargs["result"]
    assert result.model == harness.model
    assert result.choices[0].message.content == "pong"
    assert call_kwargs["custom_llm_provider"] == "xai"
    assert call_kwargs["model"] == harness.model
    assert call_kwargs["response_cost"] > 0
    metadata = call_kwargs["litellm_params"]["metadata"]
    assert metadata["passthrough_route_family"] == "grok_cli_chat_proxy"
    assert metadata["aawm_stream_logging_custom_llm_provider"] == "xai"
    assert metadata["aawm_stream_logging_is_openai_responses"] is True
    assert metadata["aawm_local_stream_finalize_ms"] >= 0


def test_grok_build_harness_captures_provider_error_and_quota_metadata():
    harness = GrokBuildPassthroughHarness()
    kwargs = harness.logging_kwargs(litellm_call_id="call-grok-harness-error")
    error = HTTPException(
        status_code=401,
        detail='{"error":"Invalid or expired credentials"}',
    )

    observation = aawm_agent_identity._build_provider_error_observation(
        kwargs=kwargs,
        result=error,
        start_time="2026-06-01T19:55:00Z",
        end_time="2026-06-01T19:55:01Z",
    )

    assert observation is not None
    assert observation["provider"] == "xai"
    assert observation["model"] == "grok-build"
    assert observation["route_family"] == "grok_cli_chat_proxy"
    assert observation["error_class"] == "auth_failed"
    assert observation["metadata"]["passthrough_route_family"] == "grok_cli_chat_proxy"

    billing_kwargs = harness.logging_kwargs(litellm_call_id="call-grok-harness-billing")
    billing_kwargs["passthrough_logging_payload"]["url"] = (
        f"{harness.upstream_base}/v1/billing"
    )
    billing_kwargs["standard_pass_through_logging_payload"] = billing_kwargs[
        "passthrough_logging_payload"
    ]
    billing_payload = {
        "config": {
            "monthlyLimit": {"val": 60000},
            "used": {"val": 600},
            "billingPeriodStart": "2026-06-01T00:00:00+00:00",
            "billingPeriodEnd": "2026-07-01T00:00:00+00:00",
        }
    }
    billing_kwargs["passthrough_logging_payload"]["response_body"] = billing_payload

    observations = _build_rate_limit_observations(
        kwargs=billing_kwargs,
        result=billing_payload,
        start_time="2026-06-01T19:55:00Z",
        end_time="2026-06-01T19:55:01Z",
    )

    assert len(observations) == 1
    quota = observations[0]
    assert quota["source"] == "grok_billing"
    assert quota["provider"] == "xai"
    assert quota["client_family"] == "grok-build"
    assert quota["model"] == "grok-build"
    assert quota["limit_id"] == "xai_grok_build_monthly_requests"
    assert quota["quota_type"] == "requests"
    assert quota["remaining_pct"] == 99.0


@pytest.mark.asyncio
async def test_grok_build_harness_live_smoke_is_explicitly_gated():
    if os.getenv("AAWM_GROK_BUILD_LIVE_SMOKE") != "1":
        pytest.skip("set AAWM_GROK_BUILD_LIVE_SMOKE=1 to run live Grok smoke")

    required_env = {
        "AAWM_GROK_BUILD_LIVE_BASE_URL": os.getenv("AAWM_GROK_BUILD_LIVE_BASE_URL"),
        "AAWM_GROK_BUILD_LIVE_LITELLM_KEY": os.getenv(
            "AAWM_GROK_BUILD_LIVE_LITELLM_KEY"
        ),
        "AAWM_GROK_BUILD_LIVE_OIDC_TOKEN": os.getenv(
            "AAWM_GROK_BUILD_LIVE_OIDC_TOKEN"
        ),
        "AAWM_GROK_BUILD_LIVE_XAI_TOKEN_AUTH": os.getenv(
            "AAWM_GROK_BUILD_LIVE_XAI_TOKEN_AUTH"
        ),
    }
    missing = [key for key, value in required_env.items() if not value]
    if missing:
        pytest.skip(f"missing live Grok smoke env vars: {', '.join(missing)}")

    base_url = required_env["AAWM_GROK_BUILD_LIVE_BASE_URL"].rstrip("/")
    headers = {
        "authorization": f"Bearer {required_env['AAWM_GROK_BUILD_LIVE_OIDC_TOKEN']}",
        "x-litellm-api-key": required_env["AAWM_GROK_BUILD_LIVE_LITELLM_KEY"],
        "x-xai-token-auth": required_env["AAWM_GROK_BUILD_LIVE_XAI_TOKEN_AUTH"],
        "x-grok-client-version": os.getenv(
            "AAWM_GROK_BUILD_LIVE_CLIENT_VERSION",
            "0.1.210",
        ),
        "x-grok-model-override": "grok-build",
        "x-grok-session-id": "grok-live-smoke-session",
        "user-agent": "grok-live-smoke",
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{base_url}/grok/v1/responses",
            headers=headers,
            json={"model": "grok-build", "input": "Reply exactly: grok live smoke"},
        )

    assert response.status_code < 500
