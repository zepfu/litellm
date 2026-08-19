import os
from unittest.mock import patch

import httpx
import pytest

import litellm
from litellm.llms.cursor_agent.chat.transformation import CursorAgentConfig
from litellm.llms.cursor_agent.common_utils import (
    CLOUD_AGENTS_PROVIDER,
    CURSOR_AGENT_DASHBOARD_HOST,
    CURSOR_AGENT_PROVIDER,
    CURSOR_AGENT_RUN_PATH,
    CURSOR_AGENT_TURN_HOST,
    CursorAgentError,
    auth_exchange_url,
    build_run_request,
    build_turn_headers,
    current_period_usage_url,
    extract_text_from_agent_payload,
    resolve_access_token,
    resolve_dashboard_api_base,
    resolve_turn_api_base,
    run_url,
    strip_provider_prefix,
)
from litellm.types.utils import LlmProviders, ModelResponse


def test_provider_identity_is_not_cloud_agents_cursor():
    config = CursorAgentConfig()
    assert config.custom_llm_provider == "cursor_agent"
    assert CURSOR_AGENT_PROVIDER == "cursor_agent"
    assert CLOUD_AGENTS_PROVIDER == "cursor"
    assert LlmProviders.CURSOR_AGENT.value == "cursor_agent"
    assert LlmProviders.CURSOR.value == "cursor"
    assert LlmProviders.CURSOR_AGENT.value != LlmProviders.CURSOR.value


def test_not_openai_compatible_and_registered():
    assert "cursor_agent" not in litellm.openai_compatible_providers
    assert "cursor_agent" in litellm.provider_list
    assert "cursor" in litellm.provider_list


def test_default_hosts_are_split():
    assert resolve_turn_api_base(None) == CURSOR_AGENT_TURN_HOST
    assert resolve_dashboard_api_base(None) == CURSOR_AGENT_DASHBOARD_HOST
    assert CURSOR_AGENT_TURN_HOST == "https://agentn.global.api5.cursor.sh"
    assert CURSOR_AGENT_DASHBOARD_HOST == "https://api2.cursor.sh"
    assert run_url(None).endswith(CURSOR_AGENT_RUN_PATH)
    assert run_url(None) == (
        "https://agentn.global.api5.cursor.sh/agent.v1.AgentService/Run"
    )
    assert auth_exchange_url(None) == (
        "https://api2.cursor.sh/auth/exchange_user_api_key"
    )
    assert current_period_usage_url(None) == (
        "https://api2.cursor.sh/aiserver.v1.DashboardService/GetCurrentPeriodUsage"
    )
    assert CURSOR_AGENT_TURN_HOST not in auth_exchange_url(None)
    assert CURSOR_AGENT_TURN_HOST not in current_period_usage_url(None)
    assert CURSOR_AGENT_DASHBOARD_HOST not in run_url(None)
    assert "/v0/me" not in current_period_usage_url(None)


def test_complete_url_is_agent_service_run():
    url = CursorAgentConfig().get_complete_url(
        api_base=None,
        api_key=None,
        model="composer-2.5",
        optional_params={},
        litellm_params={},
    )
    assert url == "https://agentn.global.api5.cursor.sh/agent.v1.AgentService/Run"
    assert "RunSSE" not in url
    assert "BidiAppend" not in url


def test_cursor_cli_key_is_ignored(monkeypatch):
    monkeypatch.setenv("CURSOR_CLI_KEY", "cli-key-must-be-ignored")
    monkeypatch.delenv("CURSOR_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("CURSOR_API_KEY", raising=False)
    with pytest.raises(CursorAgentError) as exc:
        resolve_access_token(None)
    assert "CURSOR_CLI_KEY" in exc.value.message
    assert os.environ.get("CURSOR_CLI_KEY") == "cli-key-must-be-ignored"


def test_auth_token_preferred_over_api_key(monkeypatch):
    monkeypatch.setenv("CURSOR_AUTH_TOKEN", "stored-access-token")
    monkeypatch.setenv("CURSOR_API_KEY", "raw-api-key")
    monkeypatch.setenv("CURSOR_CLI_KEY", "cli-key")
    assert resolve_access_token(None) == "stored-access-token"


def test_explicit_api_key_wins(monkeypatch):
    monkeypatch.setenv("CURSOR_AUTH_TOKEN", "stored-access-token")
    assert resolve_access_token("explicit-access") == "explicit-access"


def test_http2_headers_omit_streaming_and_checksum():
    headers = build_turn_headers(
        "access-token",
        extra_headers={
            "X-Cursor-Streaming": "true",
            "x-cursor-checksum": "must-not-pass",
            "Authorization": "Basic ignored",
        },
        request_id="req-1",
        http2=True,
    )
    names = {key.lower() for key in headers}
    assert headers["authorization"] == "Bearer access-token"
    assert headers["x-cursor-client-type"] == "cli"
    assert headers["x-cursor-client-version"].startswith("cli-")
    assert headers["x-ghost-mode"] == "true"
    assert headers["x-request-id"] == "req-1"
    assert headers["connect-protocol-version"] == "1"
    assert "user-agent" in names
    assert headers["user-agent"].startswith("Cursor-CLI/")
    assert "x-cursor-streaming" not in names
    assert "x-cursor-checksum" not in names


def test_validate_environment_uses_http2_headers(monkeypatch):
    monkeypatch.setenv("CURSOR_AUTH_TOKEN", "stored-access-token")
    monkeypatch.setenv("CURSOR_API_KEY", "raw-api-key")
    monkeypatch.setenv("CURSOR_CLI_KEY", "cli-key")
    headers = CursorAgentConfig().validate_environment(
        headers={"X-Cursor-Streaming": "true"},
        model="composer-2.5",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={},
        litellm_params={},
        api_key=None,
    )
    assert headers["authorization"] == "Bearer stored-access-token"
    assert headers["x-cursor-client-type"] == "cli"
    assert headers["connect-protocol-version"] == "1"
    assert "x-cursor-streaming" not in {key.lower() for key in headers}


def test_prompt_maps_to_user_message_text():
    request = build_run_request(
        model="cursor_agent/composer-2.5",
        messages=[
            {"role": "system", "content": "ignore"},
            {"role": "user", "content": "hello from litellm"},
        ],
    )
    action = request["run_request"]["action"]["user_message_action"]
    assert action["user_message"]["text"] == "hello from litellm"
    assert request["run_request"]["requested_model"]["model_id"] == "composer-2.5"
    assert request["run_request"]["model_details"]["model_id"] == "composer-2.5"


def test_model_slugs_are_preserved():
    for slug in ("composer-2.5", "cursor-grok-4.6-high"):
        request = build_run_request(
            model=f"cursor_agent/{slug}",
            messages=[{"role": "user", "content": "ping"}],
        )
        assert request["run_request"]["requested_model"]["model_id"] == slug
        assert strip_provider_prefix(f"cursor_agent/{slug}") == slug


def test_text_delta_and_turn_ended_parse():
    text, ended = extract_text_from_agent_payload(
        {"interaction_update": {"text_delta": {"text": "Hello"}}}
    )
    assert text == "Hello"
    assert ended is False
    text, ended = extract_text_from_agent_payload(
        {"interaction_update": {"turn_ended": {}}}
    )
    assert text == ""
    assert ended is True


def test_transform_response_joins_text_deltas():
    config = CursorAgentConfig()
    raw = httpx.Response(
        200,
        json=[
            {"interaction_update": {"text_delta": {"text": "Hel"}}},
            {"interaction_update": {"text_delta": {"text": "lo"}}},
            {"interaction_update": {"turn_ended": {}}},
        ],
        request=httpx.Request("POST", run_url()),
    )
    response = config.transform_response(
        model="composer-2.5",
        raw_response=raw,
        model_response=ModelResponse(),
        logging_obj=None,
        request_data={},
        messages=[{"role": "user", "content": "hi"}],
        optional_params={},
        litellm_params={},
        encoding=None,
    )
    assert response.choices[0].message.content == "Hello"
    assert response.model == "cursor_agent/composer-2.5"


def test_get_llm_provider_resolves_cursor_agent():
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    model, provider, _key, api_base = get_llm_provider(
        model="cursor_agent/composer-2.5"
    )
    assert provider == "cursor_agent"
    assert model == "composer-2.5"
    assert api_base == CURSOR_AGENT_TURN_HOST


def test_provider_config_manager_returns_cursor_agent_config():
    from litellm.utils import ProviderConfigManager

    config = ProviderConfigManager.get_provider_chat_config(
        model="composer-2.5",
        provider=LlmProviders.CURSOR_AGENT,
    )
    assert isinstance(config, CursorAgentConfig)
    assert config.supports_stream_param_in_request_body is False
    assert config.should_fake_stream(model="composer-2.5", stream=True) is True


def test_no_cli_subprocess_on_request_build():
    with patch("subprocess.Popen") as mock_popen, patch(
        "subprocess.run"
    ) as mock_run:
        build_run_request(
            model="cursor_agent/composer-2.5",
            messages=[{"role": "user", "content": "ping"}],
        )
        build_turn_headers("token")
        CursorAgentConfig().validate_environment(
            headers={},
            model="composer-2.5",
            messages=[{"role": "user", "content": "ping"}],
            optional_params={},
            litellm_params={},
            api_key="token",
        )
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        get_llm_provider(model="cursor_agent/composer-2.5")
        mock_popen.assert_not_called()
        mock_run.assert_not_called()
