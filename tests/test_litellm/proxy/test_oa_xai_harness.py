import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException

import litellm
from litellm.integrations import aawm_agent_identity
from litellm.integrations.aawm_agent_identity import (
    _build_session_history_record,
    _finalize_rate_limit_observation,
)
from litellm.llms.xai import oauth
from litellm.proxy.route_llm_request import route_request


class OaXaiHarness:
    public_to_upstream = {
        "oa_xai/grok-4.3": "xai/grok-4.3",
        "oa_xai/grok-4.20-0309-reasoning": "xai/grok-4.20-0309-reasoning",
        "oa_xai/grok-4.20-0309-non-reasoning": "xai/grok-4.20-0309-non-reasoning",
        "oa_xai/grok-4.20-multi-agent-0309": "xai/grok-4.20-multi-agent-0309",
    }
    api_base = "https://api.x.ai/v1"
    session_id = "oa-xai-harness-session"

    def credential_payload(
        self,
        *,
        token: str = "old-access-token",
        refresh_token: str = "refresh-token",
        expires_at: datetime | None = None,
        scoped: bool = False,
    ) -> dict[str, Any]:
        record = {
            "key": token,
            "access_token": token,
            "refresh_token": refresh_token,
            "expires_at": (
                expires_at or datetime.now(timezone.utc) + timedelta(hours=1)
            ).isoformat(),
            "oidc_client_id": "xai-oauth-client-id",
            "token_endpoint": "https://auth.test/token",
        }
        if scoped:
            return {oauth._DEFAULT_XAI_OAUTH_SCOPE: record}
        return record

    def write_credential(self, tmp_path, payload: dict[str, Any]):
        credential_path = tmp_path / "litellm-xai-oauth.json"
        credential_path.write_text(json.dumps(payload), encoding="utf-8")
        return credential_path

    def request_data(self, public_model: str) -> dict[str, Any]:
        return {
            "model": public_model,
            "messages": [{"role": "user", "content": "Reply with ack."}],
            "metadata": {"session_id": self.session_id, "tags": ["existing-tag"]},
        }

    async def route_with_token(
        self,
        public_model: str,
        *,
        access_token: str = "managed-oauth-token",
    ) -> dict[str, Any]:
        llm_router = MagicMock()
        llm_router.model_names = []
        llm_router.has_model_id.return_value = False
        llm_router.acompletion.return_value = {"ok": True}
        data = self.request_data(public_model)

        with patch(
            "litellm.llms.xai.oauth.get_xai_oauth_access_token",
            new=AsyncMock(return_value=access_token),
        ):
            mock_completion = MagicMock(return_value={"ok": True})
            original_acompletion = litellm.acompletion
            litellm.acompletion = mock_completion
            try:
                response = await route_request(data, llm_router, None, "acompletion")
            finally:
                litellm.acompletion = original_acompletion

        assert response == {"ok": True}
        llm_router.acompletion.assert_not_called()
        mock_completion.assert_called_once()
        return mock_completion.call_args.kwargs

    def kwargs_for_observability(self, public_model: str) -> dict[str, Any]:
        upstream_model = self.public_to_upstream[public_model]
        metadata = oauth.build_oa_xai_metadata(public_model, upstream_model)
        metadata["session_id"] = self.session_id
        return {
            "model": upstream_model,
            "custom_llm_provider": "xai",
            "call_type": "acompletion",
            "litellm_call_id": "call-oa-xai-harness",
            "litellm_params": {
                "metadata": metadata,
                "proxy_server_request": {
                    "headers": {
                        "authorization": "Bearer litellm-client-key",
                        "x-litellm-api-key": "litellm-client-key",
                    },
                    "body": self.request_data(public_model),
                },
            },
            "standard_logging_object": {"metadata": {}, "request_tags": []},
        }

    def response_body(self) -> dict[str, Any]:
        return {
            "id": "resp-oa-xai-harness",
            "model": "grok-4.3",
            "usage": {
                "prompt_tokens": 17,
                "completion_tokens": 4,
                "total_tokens": 21,
            },
            "choices": [{"message": {"role": "assistant", "content": "ack"}}],
        }


@pytest.mark.parametrize(
    "public_model,upstream_model", OaXaiHarness.public_to_upstream.items()
)
def test_oa_xai_harness_maps_all_public_models(public_model, upstream_model):
    metadata = oauth.build_oa_xai_metadata(public_model, upstream_model)
    catalog = json.loads(
        Path("model_prices_and_context_window.json").read_text(encoding="utf-8")
    )

    assert oauth.resolve_oa_xai_upstream_model(public_model) == upstream_model
    assert catalog[public_model]["mode"] == "responses"
    assert metadata["xai_oauth_public_model"] == public_model
    assert metadata["xai_oauth_upstream_model"] == upstream_model
    assert metadata["auth_mode"] == "oauth"
    assert metadata["credential_family"] == "xai_oauth"
    assert metadata["shared_quota_family"] == "xai_grok_subscription"
    assert metadata["grok_subscription_quota_shared"] is True
    assert "route:xai_oauth_api" in metadata["tags"]


@pytest.mark.asyncio
async def test_oa_xai_harness_loads_litellm_owned_scoped_credential(
    tmp_path,
    monkeypatch,
):
    harness = OaXaiHarness()
    credential_path = harness.write_credential(
        tmp_path,
        harness.credential_payload(token="scoped-managed-token", scoped=True),
    )
    monkeypatch.setenv("LITELLM_XAI_OAUTH_AUTH_FILE", str(credential_path))

    assert await oauth.get_xai_oauth_access_token() == "scoped-managed-token"


@pytest.mark.asyncio
async def test_grok_native_oauth_loads_default_grok_auth_json_scoped_record(
    tmp_path,
    monkeypatch,
):
    harness = OaXaiHarness()
    grok_home = tmp_path / ".grok"
    grok_home.mkdir()
    credential_path = grok_home / "auth.json"
    credential_path.write_text(
        json.dumps(
            harness.credential_payload(token="grok-native-token", scoped=True)
        ),
        encoding="utf-8",
    )
    credential_path.chmod(0o600)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("LITELLM_XAI_GROK_AUTH_FILE", raising=False)
    monkeypatch.delenv("LITELLM_XAI_OAUTH_GROK_AUTH_FILE", raising=False)
    monkeypatch.delenv("GROK_AUTH_FILE", raising=False)
    monkeypatch.delenv("GROK_HOME", raising=False)

    assert await oauth.get_grok_native_oauth_access_token() == "grok-native-token"


@pytest.mark.asyncio
async def test_grok_native_oauth_refreshes_with_auth_json_lock_and_preserves_mode(
    tmp_path,
    monkeypatch,
):
    harness = OaXaiHarness()
    grok_home = tmp_path / ".grok"
    grok_home.mkdir()
    credential_path = grok_home / "auth.json"
    credential_path.write_text(
        json.dumps(
            harness.credential_payload(
                token="expiring-grok-token",
                expires_at=datetime.now(timezone.utc) + timedelta(seconds=5),
                scoped=True,
            )
        ),
        encoding="utf-8",
    )
    credential_path.chmod(0o600)
    monkeypatch.setenv("LITELLM_XAI_GROK_AUTH_FILE", str(credential_path))
    monkeypatch.setenv("LITELLM_XAI_OAUTH_TOKEN_ENDPOINT", "https://auth.test/token")
    refresh_calls: list[dict[str, Any]] = []

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def post(self, url, data, headers):
            refresh_calls.append({"url": url, "data": data, "headers": headers})
            return httpx.Response(
                200,
                json={
                    "access_token": "refreshed-grok-native-token",
                    "refresh_token": "refreshed-grok-refresh-token",
                    "expires_in": 3600,
                    "token_type": "Bearer",
                },
            )

    with patch("litellm.llms.xai.oauth.httpx.AsyncClient", FakeAsyncClient):
        assert (
            await oauth.get_grok_native_oauth_access_token()
            == "refreshed-grok-native-token"
        )

    assert len(refresh_calls) == 1
    assert (grok_home / "auth.json.lock").exists()
    refreshed_payload = json.loads(credential_path.read_text(encoding="utf-8"))
    refreshed_record = refreshed_payload[oauth._DEFAULT_XAI_OAUTH_SCOPE]
    assert refreshed_record["access_token"] == "refreshed-grok-native-token"
    assert refreshed_record["refresh_token"] == "refreshed-grok-refresh-token"
    assert credential_path.stat().st_mode & 0o777 == 0o600


@pytest.mark.asyncio
async def test_oa_xai_harness_refreshes_and_serializes_near_expiry_credentials(
    tmp_path,
    monkeypatch,
):
    harness = OaXaiHarness()
    credential_path = harness.write_credential(
        tmp_path,
        harness.credential_payload(
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=5)
        ),
    )
    monkeypatch.setenv("LITELLM_XAI_OAUTH_AUTH_FILE", str(credential_path))
    monkeypatch.setenv("LITELLM_XAI_OAUTH_TOKEN_ENDPOINT", "https://auth.test/token")
    refresh_calls: list[dict[str, Any]] = []

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def post(self, url, data, headers):
            refresh_calls.append({"url": url, "data": data, "headers": headers})
            await asyncio.sleep(0)
            return httpx.Response(
                200,
                json={
                    "access_token": "refreshed-managed-token",
                    "refresh_token": "refreshed-refresh-token",
                    "expires_in": 3600,
                    "token_type": "Bearer",
                },
            )

    with patch("litellm.llms.xai.oauth.httpx.AsyncClient", FakeAsyncClient):
        tokens = await asyncio.gather(
            oauth.get_xai_oauth_access_token(),
            oauth.get_xai_oauth_access_token(),
        )

    assert tokens == ["refreshed-managed-token", "refreshed-managed-token"]
    assert len(refresh_calls) == 1
    assert refresh_calls[0]["url"] == "https://auth.test/token"
    assert refresh_calls[0]["data"]["grant_type"] == "refresh_token"
    assert refresh_calls[0]["data"]["client_id"] == "xai-oauth-client-id"
    refreshed_payload = json.loads(credential_path.read_text(encoding="utf-8"))
    assert refreshed_payload["access_token"] == "refreshed-managed-token"
    assert refreshed_payload["refresh_token"] == "refreshed-refresh-token"


@pytest.mark.asyncio
async def test_oa_xai_harness_returns_reseed_errors_for_missing_or_terminal_credentials(
    tmp_path,
    monkeypatch,
):
    harness = OaXaiHarness()
    monkeypatch.delenv("LITELLM_XAI_OAUTH_AUTH_FILE", raising=False)

    with pytest.raises(ValueError, match="LITELLM_XAI_OAUTH_AUTH_FILE"):
        await oauth.get_xai_oauth_access_token()

    credential_path = harness.write_credential(
        tmp_path,
        {
            "key": "expired-token",
            "expires_at": (
                datetime.now(timezone.utc) + timedelta(seconds=5)
            ).isoformat(),
            "oidc_client_id": "xai-oauth-client-id",
        },
    )
    monkeypatch.setenv("LITELLM_XAI_OAUTH_AUTH_FILE", str(credential_path))

    with pytest.raises(ValueError, match="Reseed or relogin"):
        await oauth.get_xai_oauth_access_token()

    credential_path.write_text(
        json.dumps(
            harness.credential_payload(
                expires_at=datetime.now(timezone.utc) + timedelta(seconds=5)
            )
        ),
        encoding="utf-8",
    )

    class FailingAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def post(self, url, data, headers):
            return httpx.Response(400, json={"error": "invalid_grant"})

    with patch("litellm.llms.xai.oauth.httpx.AsyncClient", FailingAsyncClient):
        with pytest.raises(ValueError, match="invalid_grant"):
            await oauth.get_xai_oauth_access_token()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "public_model,upstream_model", OaXaiHarness.public_to_upstream.items()
)
async def test_oa_xai_harness_routes_litellm_client_to_upstream_oauth(
    public_model,
    upstream_model,
):
    harness = OaXaiHarness()

    call_kwargs = await harness.route_with_token(public_model)

    assert call_kwargs["model"] == upstream_model
    assert call_kwargs["api_key"] == "managed-oauth-token"
    assert call_kwargs["api_base"] == harness.api_base
    assert call_kwargs["custom_llm_provider"] == "xai"
    metadata = call_kwargs["metadata"]
    assert metadata["session_id"] == harness.session_id
    assert metadata["auth_mode"] == "oauth"
    assert metadata["credential_family"] == "xai_oauth"
    assert metadata["passthrough_route_family"] == "xai_oauth_api"
    assert metadata["xai_oauth_public_model"] == public_model
    assert metadata["xai_oauth_upstream_model"] == upstream_model
    assert metadata["shared_quota_family"] == "xai_grok_subscription"
    assert "existing-tag" in metadata["tags"]
    assert "route:xai_oauth_api" in metadata["tags"]
    assert "authorization" not in harness.request_data(public_model)
    assert "api_key" not in harness.request_data(public_model)
    assert "api_base" not in harness.request_data(public_model)


@pytest.mark.parametrize(
    "public_model",
    [
        "oa_xai/grok-4.3",
        "oa_xai/grok-4.20-multi-agent-0309",
    ],
)
def test_oa_xai_harness_validates_session_history_provider_error_and_quota_metadata(
    public_model,
):
    harness = OaXaiHarness()
    kwargs = harness.kwargs_for_observability(public_model)
    record = _build_session_history_record(
        kwargs=kwargs,
        result=harness.response_body(),
        start_time="2026-06-01T20:05:00Z",
        end_time="2026-06-01T20:05:01Z",
    )

    assert record is not None
    assert record["provider"] == "xai"
    assert record["model"] == public_model
    assert record["model_group"] == public_model
    assert record["input_tokens"] == 17
    assert record["output_tokens"] == 4
    assert record["response_cost_usd"] is not None
    assert record["response_cost_usd"] > 0
    metadata = record["metadata"]
    assert metadata["auth_mode"] == "oauth"
    assert metadata["credential_family"] == "xai_oauth"
    assert metadata["passthrough_route_family"] == "xai_oauth_api"
    assert metadata["shared_quota_family"] == "xai_grok_subscription"
    assert metadata["grok_subscription_quota_shared"] is True

    error = HTTPException(
        status_code=401,
        detail={"error": "invalid_grant", "message": "expired managed OAuth token"},
    )
    provider_error = aawm_agent_identity._build_provider_error_observation(
        kwargs=kwargs,
        result=error,
        start_time="2026-06-01T20:05:00Z",
        end_time="2026-06-01T20:05:01Z",
    )

    assert provider_error is not None
    assert provider_error["provider"] == "xai"
    assert provider_error["model"] == public_model
    assert provider_error["route_family"] == "xai_oauth_api"
    assert provider_error["error_class"] == "auth_failed"
    assert provider_error["metadata"]["credential_family"] == "xai_oauth"

    quota_observation = _finalize_rate_limit_observation(
        observation={
            "source": "xai_oauth_api_headers",
            "provider": "xai",
            "model": public_model,
            "limit_id": "xai_grok_subscription",
            "quota_type": "requests",
            "remaining_pct": 80.0,
            "used_percentage": 20.0,
            "observed_at": datetime(2026, 6, 1, 20, 5, tzinfo=timezone.utc),
            "metadata": kwargs["litellm_params"]["metadata"],
        },
        context={
            "provider": "xai",
            "model": "oa_xai/grok-4.3",
            "metadata": kwargs["litellm_params"]["metadata"],
            "observed_at": datetime(2026, 6, 1, 20, 5, tzinfo=timezone.utc),
        },
    )

    assert quota_observation["provider"] == "xai"
    assert quota_observation["model"] == public_model
    assert quota_observation["metadata"]["auth_mode"] == "oauth"
    assert quota_observation["metadata"]["credential_family"] == "xai_oauth"
    assert quota_observation["metadata"]["shared_quota_family"] == (
        "xai_grok_subscription"
    )
    assert quota_observation["limit_key"].startswith("xai:")


@pytest.mark.asyncio
async def test_oa_xai_harness_live_smoke_is_explicitly_gated():
    if os.getenv("AAWM_OA_XAI_LIVE_SMOKE") != "1":
        pytest.skip("set AAWM_OA_XAI_LIVE_SMOKE=1 to run live oa_xai smoke")

    required_env = {
        "AAWM_OA_XAI_LIVE_BASE_URL": os.getenv("AAWM_OA_XAI_LIVE_BASE_URL"),
        "AAWM_OA_XAI_LIVE_LITELLM_KEY": os.getenv("AAWM_OA_XAI_LIVE_LITELLM_KEY"),
        "LITELLM_XAI_OAUTH_AUTH_FILE": os.getenv("LITELLM_XAI_OAUTH_AUTH_FILE"),
    }
    missing = [key for key, value in required_env.items() if not value]
    if missing:
        pytest.skip(f"missing live oa_xai smoke env vars: {', '.join(missing)}")

    base_url = required_env["AAWM_OA_XAI_LIVE_BASE_URL"].rstrip("/")
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{base_url}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {required_env['AAWM_OA_XAI_LIVE_LITELLM_KEY']}"
            },
            json={
                "model": "oa_xai/grok-4.3",
                "messages": [
                    {"role": "user", "content": "Reply exactly: oa xai live smoke"}
                ],
            },
        )

    assert response.status_code < 500
