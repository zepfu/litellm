import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import yaml
from fastapi import HTTPException

import litellm
from litellm.integrations import aawm_agent_identity
from litellm.integrations.aawm_agent_identity import (
    _build_session_history_record,
    _finalize_rate_limit_observation,
)
from litellm.llms.xai import oauth
from litellm.proxy.route_llm_request import route_request
from litellm.responses.utils import ResponsesAPIRequestUtils


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


@pytest.mark.parametrize(
    "model,normalized",
    [
        ("grok-build", "grok-build"),
        ("xai/grok-build", "grok-build"),
        ("grok-build-0.1", "grok-build-0.1"),
        ("xai/grok-build-0.1", "grok-build-0.1"),
        ("grok-composer-2.5-fast", "grok-composer-2.5-fast"),
    ],
)
def test_grok_native_oauth_model_selection_includes_build_0_1(
    model,
    normalized,
):
    assert oauth.normalize_grok_native_oauth_model(model) == normalized
    assert oauth.is_grok_native_oauth_model(model) is True


def test_litellm_dev_grok_native_oidc_auth_is_sidecar_refreshed_read_only() -> None:
    compose_path = Path(__file__).resolve().parents[3] / "docker-compose.dev.yml"
    compose = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
    litellm_dev = compose["services"]["litellm-dev"]
    provider_status = compose["services"]["provider-status-observations"]

    grok_mounts = [
        volume
        for volume in litellm_dev["volumes"]
        if isinstance(volume, str)
        and volume.startswith("/home/zepfu/.grok:/home/zepfu/.grok")
    ]
    assert grok_mounts == ["/home/zepfu/.grok:/home/zepfu/.grok:ro"]
    assert "/home/zepfu/.litellm/xai:/home/zepfu/.litellm/xai" in litellm_dev[
        "volumes"
    ]
    assert (
        "LITELLM_XAI_OAUTH_AUTH_FILE=${LITELLM_XAI_OAUTH_AUTH_FILE:-/home/zepfu/.litellm/xai/oauth-auth.json}"
        in litellm_dev["environment"]
    )
    assert (
        "LITELLM_XAI_GROK_AUTH_FILE=${LITELLM_XAI_GROK_AUTH_FILE:-/home/zepfu/.grok/auth.json}"
        in litellm_dev["environment"]
    )
    assert not any(
        str(value).startswith("LITELLM_XAI_GROK_SEED_AUTH_FILE=")
        for value in litellm_dev["environment"]
    )
    assert not any(
        str(value).startswith("LITELLM_XAI_GROK_AUTH_LOCK_FILE=")
        for value in litellm_dev["environment"]
    )
    assert "/home/zepfu/.grok:/home/zepfu/.grok" in provider_status["volumes"]
    assert (
        "AAWM_GROK_OIDC_REFRESH_ENABLED=${AAWM_GROK_OIDC_REFRESH_ENABLED:-1}"
        in provider_status["environment"]
    )
    assert (
        "AAWM_GROK_OIDC_AUTH_FILE=${AAWM_GROK_OIDC_AUTH_FILE:-/home/zepfu/.grok/auth.json}"
        in provider_status["environment"]
    )
    assert (
        "AAWM_GROK_OIDC_REFRESH_INTERVAL_SECONDS=${AAWM_GROK_OIDC_REFRESH_INTERVAL_SECONDS:-3600}"
        in provider_status["environment"]
    )
    assert (
        "AAWM_GROK_OIDC_FORCE_REFRESH=${AAWM_GROK_OIDC_FORCE_REFRESH:-1}"
        in provider_status["environment"]
    )


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
async def test_grok_native_oauth_does_not_copy_seed_auth_file(
    tmp_path,
    monkeypatch,
):
    harness = OaXaiHarness()
    seed_path = tmp_path / ".grok" / "auth.json"
    seed_path.parent.mkdir()
    seed_payload = harness.credential_payload(token="seed-grok-token", scoped=True)
    seed_path.write_text(json.dumps(seed_payload), encoding="utf-8")
    seed_path.chmod(0o600)

    managed_path = tmp_path / ".litellm" / "xai" / "grok-auth.json"
    monkeypatch.setenv("LITELLM_XAI_GROK_AUTH_FILE", str(managed_path))
    monkeypatch.setenv("LITELLM_XAI_GROK_SEED_AUTH_FILE", str(seed_path))

    with pytest.raises(ValueError, match="health/provider-status sidecar"):
        await oauth.get_grok_native_oauth_access_token()

    assert not managed_path.exists()
    assert json.loads(seed_path.read_text(encoding="utf-8")) == seed_payload


@pytest.mark.asyncio
async def test_grok_native_oauth_does_not_refresh_near_expiry_credential(
    tmp_path,
    monkeypatch,
):
    harness = OaXaiHarness()
    grok_home = tmp_path / ".grok"
    grok_home.mkdir()
    credential_path = grok_home / "auth.json"
    original_payload = harness.credential_payload(
        token="expiring-grok-token",
        expires_at=datetime.now(timezone.utc) + timedelta(seconds=5),
        scoped=True,
    )
    credential_path.write_text(json.dumps(original_payload), encoding="utf-8")
    credential_path.chmod(0o600)
    monkeypatch.setenv("LITELLM_XAI_GROK_AUTH_FILE", str(credential_path))
    monkeypatch.setenv("LITELLM_XAI_OAUTH_TOKEN_ENDPOINT", "https://auth.test/token")

    class UnexpectedRefreshClient:
        def __init__(self, *args, **kwargs):
            raise AssertionError("Grok native OIDC path must not call token refresh")

    with patch("litellm.llms.xai.oauth.httpx.AsyncClient", UnexpectedRefreshClient):
        with pytest.raises(ValueError, match="health/provider-status sidecar"):
            await oauth.get_grok_native_oauth_access_token()

    assert json.loads(credential_path.read_text(encoding="utf-8")) == original_payload
    assert not (grok_home / "auth.json.lock").exists()


@pytest.mark.asyncio
async def test_grok_native_oauth_missing_token_errors_without_mutating_file(
    tmp_path,
    monkeypatch,
):
    grok_home = tmp_path / ".grok"
    grok_home.mkdir()
    credential_path = grok_home / "auth.json"
    scope = oauth._DEFAULT_XAI_OAUTH_SCOPE
    payload = {
        scope: {
            "refresh_token": "refresh-only",
            "expires_at": (
                datetime.now(timezone.utc) + timedelta(hours=1)
            ).isoformat(),
            "oidc_client_id": "xai-oauth-client-id",
        }
    }
    credential_path.write_text(json.dumps(payload), encoding="utf-8")
    credential_path.chmod(0o600)
    monkeypatch.setenv("LITELLM_XAI_GROK_AUTH_FILE", str(credential_path))

    with pytest.raises(ValueError, match="does not contain an access token"):
        await oauth.get_grok_native_oauth_access_token()

    assert json.loads(credential_path.read_text(encoding="utf-8")) == payload


@pytest.mark.asyncio
async def test_grok_native_oauth_missing_file_errors_with_sidecar_wording(
    tmp_path,
    monkeypatch,
):
    missing_path = tmp_path / ".grok" / "auth.json"
    monkeypatch.setenv("LITELLM_XAI_GROK_AUTH_FILE", str(missing_path))

    with pytest.raises(ValueError) as exc_info:
        await oauth.get_grok_native_oauth_access_token()

    message = str(exc_info.value)
    assert "health/provider-status sidecar" in message
    assert "Grok CLI" in message
    assert "oa_xai/*" not in message


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
async def test_grok_native_oauth_near_expiry_uses_sidecar_wording_not_refresh(
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
                expires_at=datetime.now(timezone.utc) + timedelta(seconds=5),
                scoped=True,
            )
        ),
        encoding="utf-8",
    )
    credential_path.chmod(0o600)
    monkeypatch.setenv("LITELLM_XAI_GROK_AUTH_FILE", str(credential_path))

    class FailingAsyncClient:
        def __init__(self, *args, **kwargs):
            raise AssertionError("Grok native OIDC must not refresh via httpx")

    with patch("litellm.llms.xai.oauth.httpx.AsyncClient", FailingAsyncClient):
        with pytest.raises(ValueError) as exc_info:
            await oauth.get_grok_native_oauth_access_token()

    message = str(exc_info.value)
    assert "health/provider-status sidecar" in message
    assert "Grok CLI" in message
    assert "invalid_grant" not in message
    assert "managed xAI OAuth credential" not in message
    assert "oa_xai/*" not in message


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


@pytest.mark.asyncio
async def test_oa_xai_harness_decodes_previous_response_id_before_responses_egress():
    harness = OaXaiHarness()
    public_model = "oa_xai/grok-4.3"
    original_response_id = "resp_xai_upstream_compaction_blob"
    encoded_response_id = ResponsesAPIRequestUtils._build_responses_api_response_id(
        custom_llm_provider="xai",
        model_id="oa-xai-deployment-id",
        response_id=original_response_id,
    )
    data = {
        "model": public_model,
        "input": [
            {"type": "message", "role": "user", "content": "continue"},
            {
                "type": "reasoning",
                "id": "rs_direct_oa_xai_compaction",
                "summary": [],
                "encrypted_content": "encrypted-direct-oa-xai-compaction",
            },
            {"type": "function_call", "name": "exec_command", "call_id": "call_1"},
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "ok",
            },
        ],
        "previous_response_id": encoded_response_id,
        "metadata": {"session_id": harness.session_id, "tags": ["existing-tag"]},
        "litellm_metadata": {"tags": ["existing-litellm-tag"]},
    }
    llm_router = MagicMock()
    llm_router.model_names = []

    with patch(
        "litellm.llms.xai.oauth.get_xai_oauth_access_token",
        new=AsyncMock(return_value="managed-oauth-token"),
    ):
        mock_responses = MagicMock(return_value={"id": "resp_next"})
        original_aresponses = litellm.aresponses
        litellm.aresponses = mock_responses
        try:
            response = await route_request(data, llm_router, None, "aresponses")
        finally:
            litellm.aresponses = original_aresponses

    assert response == {"id": "resp_next"}
    mock_responses.assert_called_once()
    call_kwargs = mock_responses.call_args.kwargs
    assert call_kwargs["model"] == harness.public_to_upstream[public_model]
    assert call_kwargs["input"] == [
        {"type": "message", "role": "user", "content": "continue"},
        {"type": "function_call", "name": "exec_command", "call_id": "call_1"},
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "ok",
        },
    ]
    assert call_kwargs["previous_response_id"] == original_response_id
    assert call_kwargs["previous_response_id"] != encoded_response_id
    assert call_kwargs["metadata"]["session_id"] == harness.session_id
    assert call_kwargs["metadata"]["xai_responses_previous_response_id_decoded"] is True
    assert call_kwargs["litellm_metadata"][
        "xai_responses_previous_response_id_decoded"
    ] is True
    assert "existing-tag" in call_kwargs["metadata"]["tags"]
    assert "xai-responses-previous-response-id-decoded" in call_kwargs["metadata"][
        "tags"
    ]
    assert original_response_id not in call_kwargs["metadata"]["tags"]
    assert encoded_response_id not in call_kwargs["metadata"]["tags"]
    assert "existing-litellm-tag" in call_kwargs["litellm_metadata"]["tags"]
    assert "xai-responses-previous-response-id-decoded" in call_kwargs[
        "litellm_metadata"
    ]["tags"]
    assert call_kwargs["litellm_metadata"][
        "codex_unsupported_input_item_removed_count"
    ] == 1
    assert call_kwargs["litellm_metadata"][
        "codex_unsupported_input_items_removed"
    ] == [{"type": "reasoning", "index": 1, "encrypted_content": True}]


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
