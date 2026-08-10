import json
import time
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest

import litellm
from litellm.llms.kimi_code import MANAGED_KIMI_CODE_MODEL_IDS
from litellm.llms.kimi_code.chat.transformation import (
    KIMI_CODE_API_BASE,
    KIMI_CODE_CHAT_COMPLETIONS_URL,
    KIMI_CODE_CREDENTIAL_PATH_ENV,
    KimiCodeAuthenticationError,
    KimiCodeChatCompletionStreamingHandler,
    KimiCodeChatConfig,
)
from litellm.types.utils import LlmProviders, ModelResponse
from litellm.utils import (
    ProviderConfigManager,
    _invalidate_model_cost_lowercase_map,
    get_model_info,
)


@pytest.fixture(autouse=True)
def _use_current_kimi_code_model_metadata(monkeypatch: pytest.MonkeyPatch):
    model_cost = dict(litellm.model_cost)
    model_cost.update(
        {
            "kimi_code/k3": {
                "default_reasoning_effort": "high",
                "litellm_provider": "kimi_code",
                "max_input_tokens": 1048576,
                "mode": "chat",
                "supports_high_reasoning_effort": True,
                "supports_low_reasoning_effort": True,
                "supports_max_reasoning_effort": True,
                "supports_reasoning": True,
                "supports_vision": True,
            },
            "kimi_code/kimi-for-coding": {
                "litellm_provider": "kimi_code",
                "max_input_tokens": 262144,
                "mode": "chat",
                "supports_reasoning": True,
                "supports_vision": True,
            },
            "kimi_code/kimi-for-coding-highspeed": {
                "litellm_provider": "kimi_code",
                "max_input_tokens": 262144,
                "mode": "chat",
                "supports_reasoning": True,
                "supports_vision": True,
            },
        }
    )
    monkeypatch.setattr(litellm, "model_cost", model_cost)
    _invalidate_model_cost_lowercase_map()
    get_model_info.cache_clear()
    yield
    _invalidate_model_cost_lowercase_map()
    get_model_info.cache_clear()


def _write_credentials(
    credentials_path: Path,
    access_token: str = "current-access-token",
    expires_at: float | None = None,
) -> None:
    credentials_path.write_text(
        json.dumps(
            {
                "access_token": access_token,
                "expires_at": expires_at if expires_at is not None else time.time() + 300,
            }
        ),
        encoding="utf-8",
    )


def _set_credentials_path(monkeypatch: pytest.MonkeyPatch, credentials_path: Path) -> None:
    monkeypatch.setenv(KIMI_CODE_CREDENTIAL_PATH_ENV, str(credentials_path))


def test_should_register_kimi_code_provider_and_config():
    assert LlmProviders.KIMI_CODE.value == "kimi_code"
    assert "kimi_code" in litellm.openai_compatible_providers
    assert isinstance(litellm.KimiCodeChatConfig(), KimiCodeChatConfig)
    assert isinstance(
        ProviderConfigManager.get_provider_chat_config(model="k3", provider=LlmProviders.KIMI_CODE),
        KimiCodeChatConfig,
    )


def test_should_resolve_kimi_code_through_managed_provider(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    credentials_path = tmp_path / "kimi-code.json"
    _set_credentials_path(monkeypatch, credentials_path)
    _write_credentials(credentials_path)

    model, provider, api_key, api_base = litellm.get_llm_provider(model="kimi_code/k3")

    assert model == "k3"
    assert provider == "kimi_code"
    assert api_key == "current-access-token"
    assert api_base == KIMI_CODE_API_BASE


def test_should_retain_managed_kimi_code_model_id_constants():
    assert MANAGED_KIMI_CODE_MODEL_IDS == {
        "k3",
        "kimi-for-coding",
        "kimi-for-coding-highspeed",
    }
    assert "k3-preview" not in MANAGED_KIMI_CODE_MODEL_IDS


@pytest.mark.parametrize(
    "model",
    ["", "  ", "moonshot/k3", "kimi_code/", "openai/gpt-5"],
)
def test_should_reject_empty_or_foreign_namespaced_model_ids(model: str):
    config = KimiCodeChatConfig()

    with pytest.raises(ValueError, match="Unsupported managed Kimi Code model"):
        config.get_supported_openai_params(model)


@pytest.mark.parametrize(
    "model",
    ["k3", "kimi-for-coding", "kimi_code/k3-preview", "kimi-for-coding-v2", "kimi_code/k4-preview"],
)
def test_should_admit_any_nonempty_namespaced_model_id(model: str):
    config = KimiCodeChatConfig()

    supported_params = config.get_supported_openai_params(model)

    assert "max_tokens" in supported_params
    assert config._model_id(model) == model.split("/", maxsplit=1)[-1]

def test_should_hot_read_credentials_without_writing_them(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    credentials_path = tmp_path / "kimi-code.json"
    _set_credentials_path(monkeypatch, credentials_path)
    _write_credentials(credentials_path, access_token="first-token")
    config = KimiCodeChatConfig()

    _, first_token = config._get_openai_compatible_provider_info(None, "ignored-api-key")
    first_mtime_ns = credentials_path.stat().st_mtime_ns
    _write_credentials(credentials_path, access_token="second-token")
    _, second_token = config._get_openai_compatible_provider_info(None, None)

    assert first_token == "first-token"
    assert second_token == "second-token"
    assert credentials_path.stat().st_mtime_ns >= first_mtime_ns


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"access_token": ""},
        {"access_token": "redacted-secret", "expires_at": "not-a-date"},
        {"access_token": "redacted-secret", "expires_at": 0},
    ],
)
def test_should_reject_missing_malformed_or_expired_credentials_without_leaking_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, payload: dict
):
    credentials_path = tmp_path / "kimi-code.json"
    _set_credentials_path(monkeypatch, credentials_path)
    credentials_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(KimiCodeAuthenticationError) as exc_info:
        KimiCodeChatConfig()._get_openai_compatible_provider_info(None, None)

    assert "Kimi Code OAuth credentials" in str(exc_info.value)
    assert "redacted-secret" not in str(exc_info.value)
    assert "login" not in str(exc_info.value).lower()
    assert "existing Kimi Code CLI credential" in str(exc_info.value)


def test_should_preserve_explicit_expiry_timezone_offset():
    assert KimiCodeChatConfig._parse_expiry("1970-01-01T01:00:00+01:00") == 0


def test_should_use_managed_endpoint_and_honest_headers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    credentials_path = tmp_path / "kimi-code.json"
    _set_credentials_path(monkeypatch, credentials_path)
    _write_credentials(credentials_path)
    config = KimiCodeChatConfig()

    # Canonical base (with trailing slash) is accepted as a no-op
    assert (
        config.get_complete_url(
            api_base=KIMI_CODE_API_BASE + "/",
            api_key="moonshot-key",
            model="k3",
            optional_params={},
            litellm_params={},
        )
        == KIMI_CODE_CHAT_COMPLETIONS_URL
    )
    # None api_base is accepted
    assert (
        config.get_complete_url(
            api_base=None,
            api_key="moonshot-key",
            model="k3",
            optional_params={},
            litellm_params={},
        )
        == KIMI_CODE_CHAT_COMPLETIONS_URL
    )
    headers = config.validate_environment(
        headers={
            "X-Msh-Device-Id": "not-permitted",
            "x-msh-platform": "not-permitted",
            "User-Agent": "spoofed-client",
        },
        model="k3",
        messages=[],
        optional_params={},
        litellm_params={},
    )

    assert headers["Authorization"] == "Bearer current-access-token"
    assert headers["User-Agent"] == "kimi-code-cli/0.29.1"
    assert not any(header.lower().startswith("x-msh-") for header in headers)
    assert sum(header.lower() == "authorization" for header in headers) == 1


def test_should_map_max_tokens_to_max_completion_tokens():
    params = KimiCodeChatConfig().map_openai_params(
        non_default_params={"max_tokens": 1234},
        optional_params={},
        model="k3",
        drop_params=False,
    )

    assert params == {"max_completion_tokens": 1234}


@pytest.mark.parametrize(
    "non_default_params",
    [
        {"max_tokens": 1234, "max_completion_tokens": 5678},
        {"max_completion_tokens": 5678, "max_tokens": 1234},
    ],
)
def test_should_prefer_max_completion_tokens_regardless_of_input_order(
    non_default_params: dict,
):
    params = KimiCodeChatConfig().map_openai_params(
        non_default_params=non_default_params,
        optional_params={},
        model="k3",
        drop_params=False,
    )

    assert params == {"max_completion_tokens": 5678}


@pytest.mark.parametrize("effort", ["low", "high", "max"])
def test_should_map_supported_k3_reasoning_effort_to_thinking(effort: str):
    params = KimiCodeChatConfig().map_openai_params(
        non_default_params={"reasoning_effort": effort},
        optional_params={},
        model="k3",
        drop_params=False,
    )

    assert params == {"extra_body": {"thinking": {"type": "enabled", "effort": effort, "keep": "all"}}}


def test_should_reject_unsupported_k3_reasoning_effort():
    with pytest.raises(ValueError, match="Supported efforts: low, high, max"):
        KimiCodeChatConfig().map_openai_params(
            non_default_params={"reasoning_effort": "medium"},
            optional_params={},
            model="k3",
            drop_params=False,
        )


@pytest.mark.parametrize("model", ["kimi-for-coding", "kimi-for-coding-highspeed"])
def test_should_not_synthesize_k3_reasoning_effort_for_k2_7_models(model: str):
    config = KimiCodeChatConfig()

    assert "reasoning_effort" not in config.get_supported_openai_params(model)
    with pytest.raises(ValueError, match="Supported efforts: none"):
        config.map_openai_params(
            non_default_params={"reasoning_effort": "high"},
            optional_params={},
            model=model,
            drop_params=False,
        )


def test_should_force_stream_usage_while_preserving_caller_stream_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    credentials_path = tmp_path / "kimi-code-credentials.json"
    _write_credentials(credentials_path)
    _set_credentials_path(monkeypatch, credentials_path)

    request = KimiCodeChatConfig().transform_request(
        model="k3",
        messages=[{"role": "user", "content": "hello"}],
        optional_params={
            "stream": True,
            "stream_options": {"include_usage": False, "custom_option": "preserve"},
        },
        litellm_params={},
        headers={},
    )

    assert request["stream_options"] == {
        "include_usage": True,
        "custom_option": "preserve",
    }


def test_should_forward_cache_key_and_parallel_tool_calls():
    params = KimiCodeChatConfig().map_openai_params(
        non_default_params={
            "prompt_cache_key": "stable-cache-key",
            "parallel_tool_calls": True,
        },
        optional_params={},
        model="k3",
        drop_params=False,
    )

    assert params == {
        "prompt_cache_key": "stable-cache-key",
        "parallel_tool_calls": True,
    }


def test_should_send_managed_k3_request_through_litellm_completion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    respx_mock,
):
    credentials_path = tmp_path / "kimi-code.json"
    _set_credentials_path(monkeypatch, credentials_path)
    _write_credentials(credentials_path)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post(KIMI_CODE_CHAT_COMPLETIONS_URL).respond(
        json={
            "id": "chatcmpl-kimi",
            "object": "chat.completion",
            "created": 1,
            "model": "k3",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "done"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5,
            },
        }
    )

    response = litellm.completion(
        model="kimi_code/k3",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=1234,
        max_completion_tokens=5678,
        reasoning_effort="high",
        prompt_cache_key="stable-cache-key",
        parallel_tool_calls=True,
        extra_headers={
            "Authorization": "Bearer caller-token",
            "User-Agent": "spoofed-client",
            "X-Msh-Device-Id": "not-permitted",
            "X-Custom-Header": "preserved",
        },
    )

    assert response.choices[0].message.content == "done"
    assert len(respx_mock.calls) == 1
    request = respx_mock.calls[0].request
    request_body = json.loads(request.content)
    assert str(request.url) == KIMI_CODE_CHAT_COMPLETIONS_URL
    assert request.headers["Authorization"] == "Bearer current-access-token"
    assert request.headers["User-Agent"] == "kimi-code-cli/0.29.1"
    # Caller headers are never forwarded; only managed headers are emitted.
    assert "X-Custom-Header" not in request.headers
    assert not any(header.lower().startswith("x-msh-") for header in request.headers)
    assert request_body["model"] == "k3"
    assert request_body["max_completion_tokens"] == 5678
    assert "max_tokens" not in request_body
    assert request_body["prompt_cache_key"] == "stable-cache-key"
    assert request_body["parallel_tool_calls"] is True
    assert request_body["thinking"] == {
        "type": "enabled",
        "effort": "high",
        "keep": "all",
    }


def test_should_send_synthetic_future_namespaced_model_through_litellm_completion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    respx_mock,
):
    """A future `kimi_code/<model-id>` config route needs no Python update."""
    credentials_path = tmp_path / "kimi-code.json"
    _set_credentials_path(monkeypatch, credentials_path)
    _write_credentials(credentials_path)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post(KIMI_CODE_CHAT_COMPLETIONS_URL).respond(
        json={
            "id": "chatcmpl-kimi-future",
            "object": "chat.completion",
            "created": 1,
            "model": "k4-preview",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "done"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5,
            },
        }
    )

    response = litellm.completion(
        model="kimi_code/k4-preview",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=1234,
    )

    assert response.choices[0].message.content == "done"
    assert len(respx_mock.calls) == 1
    request = respx_mock.calls[0].request
    request_body = json.loads(request.content)
    assert str(request.url) == KIMI_CODE_CHAT_COMPLETIONS_URL
    assert request.headers["Authorization"] == "Bearer current-access-token"
    assert request_body["model"] == "k4-preview"
    assert request_body["max_completion_tokens"] == 1234
    assert "thinking" not in request_body


def test_should_read_reasoning_efforts_and_context_from_model_metadata():
    k3_info = get_model_info(model="k3", custom_llm_provider="kimi_code")
    k2_info = get_model_info(model="kimi-for-coding", custom_llm_provider="kimi_code")

    assert k3_info["max_input_tokens"] == 1048576
    assert k3_info["max_output_tokens"] is None
    assert k3_info["default_reasoning_effort"] == "high"
    assert k3_info["supports_low_reasoning_effort"] is True
    assert k3_info["supports_high_reasoning_effort"] is True
    assert k3_info["supports_max_reasoning_effort"] is True
    assert k2_info["max_input_tokens"] == 262144
    assert k2_info["max_output_tokens"] is None
    assert k2_info.get("supports_low_reasoning_effort") is None


def test_should_load_kimi_code_metadata_from_bundled_model_map(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")

    local_model_cost = litellm.get_model_cost_map(url="unused")

    assert local_model_cost["kimi_code/k3"]["default_reasoning_effort"] == "high"
    assert local_model_cost["kimi_code/k3"]["supports_max_reasoning_effort"] is True
    assert local_model_cost["kimi_code/kimi-for-coding"]["max_input_tokens"] == 262144
    assert local_model_cost["kimi_code/kimi-for-coding-highspeed"]["max_input_tokens"] == 262144


def test_should_promote_nested_non_streaming_usage():
    raw_response = httpx.Response(
        status_code=200,
        json={
            "id": "chatcmpl-kimi",
            "object": "chat.completion",
            "created": 1,
            "model": "k3",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "done"},
                    "finish_reason": "stop",
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 2,
                        "total_tokens": 5,
                    },
                }
            ],
        },
        request=httpx.Request("POST", KIMI_CODE_CHAT_COMPLETIONS_URL),
    )

    result = KimiCodeChatConfig().transform_response(
        model="k3",
        raw_response=raw_response,
        model_response=ModelResponse(),
        logging_obj=Mock(),
        request_data={},
        messages=[{"role": "user", "content": "hello"}],
        optional_params={},
        litellm_params={},
        encoding=None,
    )

    assert result.usage.total_tokens == 5
    assert result.choices[0].message.content == "done"


def test_should_preserve_reasoning_content_tool_deltas_and_usage_only_chunks():
    handler = KimiCodeChatCompletionStreamingHandler(
        streaming_response=iter([]),
        sync_stream=True,
    )
    response_chunk = handler.chunk_parser(
        {
            "id": "chatcmpl-kimi",
            "created": 1,
            "model": "k3",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "reasoning": "inspect the repository",
                        "content": "I will make the change.",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "read_file", "arguments": "{}"},
                            }
                        ],
                    },
                    "finish_reason": None,
                }
            ],
        }
    )
    usage_chunk = handler.chunk_parser(
        {
            "id": "chatcmpl-kimi",
            "created": 1,
            "model": "k3",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 4,
                        "total_tokens": 7,
                    },
                }
            ],
        }
    )

    delta = response_chunk.choices[0].delta
    assert delta.reasoning_content == "inspect the repository"
    assert delta.content == "I will make the change."
    assert delta.tool_calls[0].function.name == "read_file"
    assert usage_chunk.usage.total_tokens == 7


# ---------------------------------------------------------------------------
# MS-030/MS-031: descriptor-present consumer tests
# ---------------------------------------------------------------------------


def _write_contract_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    client_version: str = "0.29.1",
) -> Path:
    """Write a valid contract descriptor and set the env path."""
    import time as _time
    from litellm.secret_managers.kimi_native_contract import (
        KIMI_NATIVE_BASE_URL,
        KIMI_NATIVE_CONTRACT_PATH_ENV,
        KIMI_NATIVE_SCHEMA_VERSION,
        compute_canonical_digest,
    )

    now = _time.time()
    payload = {
        "schema_version": KIMI_NATIVE_SCHEMA_VERSION,
        "client_name": "kimi-code",
        "client_version": client_version,
        "base_url": KIMI_NATIVE_BASE_URL,
        "user_agent": f"kimi-code-cli/{client_version}",
        "issued_at": now - 60,
        "expires_at": now + 3600,
        "x_msh_platform": "kimi_code_cli",
        "x_msh_version": client_version,
        "x_msh_device_name": "aawm-service-node",
        "x_msh_device_model": "aawm-managed",
        "x_msh_os_version": "linux-6.x",
        "x_msh_device_id": "0d3f8a2e-7b14-4c6a-9e5f-a1b2c3d4e5f6",
    }
    payload["digest"] = compute_canonical_digest(payload)
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv(KIMI_NATIVE_CONTRACT_PATH_ENV, str(contract_path))
    return contract_path


def test_should_emit_descriptor_user_agent_and_x_msh_headers_when_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """With a valid descriptor, headers carry the descriptor UA and all six
    X-Msh identity headers, Content-Type for chat, and no caller spoofing."""
    credentials_path = tmp_path / "kimi-code.json"
    _set_credentials_path(monkeypatch, credentials_path)
    _write_credentials(credentials_path)
    _write_contract_descriptor(tmp_path, monkeypatch, client_version="0.29.1")

    config = KimiCodeChatConfig()
    headers = config.validate_environment(
        headers={
            "User-Agent": "spoofed-client",
            "X-Msh-Device-Id": "spoofed-device",
            "X-Msh-Platform": "spoofed-platform",
        },
        model="k3",
        messages=[],
        optional_params={},
        litellm_params={},
    )

    # Descriptor-owned User-Agent
    assert headers["User-Agent"] == "kimi-code-cli/0.29.1"
    # All six X-Msh descriptor headers
    assert headers["X-Msh-Platform"] == "kimi_code_cli"
    assert headers["X-Msh-Version"] == "0.29.1"
    assert headers["X-Msh-Device-Name"] == "aawm-service-node"
    assert headers["X-Msh-Device-Model"] == "aawm-managed"
    assert headers["X-Msh-Os-Version"] == "linux-6.x"
    assert headers["X-Msh-Device-Id"] == "0d3f8a2e-7b14-4c6a-9e5f-a1b2c3d4e5f6"
    # Content-Type for chat (json_body=True)
    assert headers["Content-Type"] == "application/json"
    # Authorization present
    assert headers["Authorization"] == "Bearer current-access-token"
    # Caller spoof headers omitted
    assert "spoofed-client" not in headers.values()
    assert "spoofed-device" not in headers.values()
    assert "spoofed-platform" not in headers.values()


def test_should_use_honest_fallback_ua_without_descriptor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Without a descriptor, the fallback UA is kimi-code-cli/0.29.1 and no
    X-Msh headers are emitted."""
    from litellm.secret_managers.kimi_native_contract import (
        KIMI_NATIVE_CONTRACT_PATH_ENV,
    )

    credentials_path = tmp_path / "kimi-code.json"
    _set_credentials_path(monkeypatch, credentials_path)
    _write_credentials(credentials_path)
    monkeypatch.delenv(KIMI_NATIVE_CONTRACT_PATH_ENV, raising=False)

    config = KimiCodeChatConfig()
    headers = config.validate_environment(
        headers={},
        model="k3",
        messages=[],
        optional_params={},
        litellm_params={},
    )

    assert headers["User-Agent"] == "kimi-code-cli/0.29.1"
    assert not any(k.lower().startswith("x-msh-") for k in headers)


def test_should_use_builtin_identity_when_contract_required_but_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """MS-035: with LITELLM_KIMI_NATIVE_CONTRACT_REQUIRED=true and no
    descriptor path set, the provider continues with the conservative
    installed-client/built-in identity instead of failing the route; only
    actual OAuth/authentication or provider failures terminate the request."""
    from litellm.secret_managers.kimi_native_contract import (
        KIMI_NATIVE_CONTRACT_PATH_ENV,
        KIMI_NATIVE_CONTRACT_REQUIRED_ENV,
    )

    credentials_path = tmp_path / "kimi-code.json"
    _set_credentials_path(monkeypatch, credentials_path)
    _write_credentials(credentials_path)
    monkeypatch.delenv(KIMI_NATIVE_CONTRACT_PATH_ENV, raising=False)
    monkeypatch.setenv(KIMI_NATIVE_CONTRACT_REQUIRED_ENV, "true")

    config = KimiCodeChatConfig()
    headers = config.validate_environment(
        headers={},
        model="k3",
        messages=[],
        optional_params={},
        litellm_params={},
    )

    assert headers["User-Agent"].startswith("kimi-code-cli/")
    assert headers["X-Msh-Platform"] == "kimi_code_cli"
    assert headers["X-Msh-Version"] == headers["User-Agent"].split("/", 1)[1]
    assert headers["Authorization"] == "Bearer current-access-token"


def test_should_use_stale_descriptor_identity_when_expired_but_valid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """MS-035: an expired but structurally valid descriptor keeps serving
    requests with its older claimed identity and all six X-Msh headers."""
    import time as _time
    from litellm.secret_managers.kimi_native_contract import (
        KIMI_NATIVE_BASE_URL,
        KIMI_NATIVE_CONTRACT_PATH_ENV,
        KIMI_NATIVE_CONTRACT_REQUIRED_ENV,
        KIMI_NATIVE_SCHEMA_VERSION,
        compute_canonical_digest,
    )

    credentials_path = tmp_path / "kimi-code.json"
    _set_credentials_path(monkeypatch, credentials_path)
    _write_credentials(credentials_path)

    now = _time.time()
    payload = {
        "schema_version": KIMI_NATIVE_SCHEMA_VERSION,
        "client_name": "kimi-code",
        "client_version": "0.28.0",
        "base_url": KIMI_NATIVE_BASE_URL,
        "user_agent": "kimi-code-cli/0.28.0",
        "issued_at": now - 7200,
        "expires_at": now - 60,
        "x_msh_platform": "kimi_code_cli",
        "x_msh_version": "0.28.0",
        "x_msh_device_name": "aawm-service-node",
        "x_msh_device_model": "aawm-managed",
        "x_msh_os_version": "linux-6.x",
        "x_msh_device_id": "0d3f8a2e-7b14-4c6a-9e5f-a1b2c3d4e5f6",
    }
    payload["digest"] = compute_canonical_digest(payload)
    contract_path = tmp_path / "contract-expired.json"
    contract_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv(KIMI_NATIVE_CONTRACT_PATH_ENV, str(contract_path))
    monkeypatch.setenv(KIMI_NATIVE_CONTRACT_REQUIRED_ENV, "true")

    config = KimiCodeChatConfig()
    headers = config.validate_environment(
        headers={},
        model="k3",
        messages=[],
        optional_params={},
        litellm_params={},
    )

    # Stale descriptor identity is served, not the built-in fallback.
    assert headers["User-Agent"] == "kimi-code-cli/0.28.0"
    assert headers["X-Msh-Version"] == "0.28.0"
    assert headers["X-Msh-Platform"] == "kimi_code_cli"
    assert headers["X-Msh-Device-Id"] == "0d3f8a2e-7b14-4c6a-9e5f-a1b2c3d4e5f6"
    assert headers["Authorization"] == "Bearer current-access-token"


def test_should_send_descriptor_headers_through_litellm_completion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    respx_mock,
):
    """End-to-end: descriptor-present request emits descriptor UA, all six
    X-Msh headers, Content-Type, and omits caller spoof headers."""
    credentials_path = tmp_path / "kimi-code.json"
    _set_credentials_path(monkeypatch, credentials_path)
    _write_credentials(credentials_path)
    _write_contract_descriptor(tmp_path, monkeypatch, client_version="0.29.1")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)

    respx_mock.post(KIMI_CODE_CHAT_COMPLETIONS_URL).respond(
        json={
            "id": "chatcmpl-kimi",
            "object": "chat.completion",
            "created": 1,
            "model": "k3",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "done"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )

    response = litellm.completion(
        model="kimi_code/k3",
        messages=[{"role": "user", "content": "hello"}],
        extra_headers={
            "User-Agent": "spoofed-client",
            "X-Msh-Device-Id": "spoofed-device",
        },
    )

    assert response.choices[0].message.content == "done"
    request = respx_mock.calls[0].request
    assert request.headers["User-Agent"] == "kimi-code-cli/0.29.1"
    assert request.headers["X-Msh-Platform"] == "kimi_code_cli"
    assert request.headers["X-Msh-Version"] == "0.29.1"
    assert request.headers["X-Msh-Device-Name"] == "aawm-service-node"
    assert request.headers["X-Msh-Device-Model"] == "aawm-managed"
    assert request.headers["X-Msh-Os-Version"] == "linux-6.x"
    assert request.headers["X-Msh-Device-Id"] == "0d3f8a2e-7b14-4c6a-9e5f-a1b2c3d4e5f6"
    assert request.headers["Content-Type"] == "application/json"
    assert "spoofed-client" not in dict(request.headers).values()
    assert "spoofed-device" not in dict(request.headers).values()


# ---------------------------------------------------------------------------
# MS-030/MS-031: caller-supplied api_base rejection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "api_base",
    [
        "https://api.moonshot.ai/v1",
        "https://api.moonshot.ai/v1/",
        "https://api.openai.com/v1",
        "https://evil.example.com/v1",
        "http://localhost:8080",
    ],
)
def test_should_reject_noncanonical_api_base_in_get_complete_url(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, api_base: str
):
    from litellm.llms.kimi_code.chat.transformation import KimiCodeApiBaseError

    credentials_path = tmp_path / "kimi-code.json"
    _set_credentials_path(monkeypatch, credentials_path)
    _write_credentials(credentials_path)
    config = KimiCodeChatConfig()

    with pytest.raises(KimiCodeApiBaseError) as exc_info:
        config.get_complete_url(
            api_base=api_base,
            api_key=None,
            model="k3",
            optional_params={},
            litellm_params={},
        )

    assert exc_info.value.status_code == 400
    assert "api_base" in str(exc_info.value)
    # Sanitized: no caller-supplied URL leaked into the error
    assert api_base not in str(exc_info.value)


@pytest.mark.parametrize(
    "api_base",
    [
        "https://api.moonshot.ai/v1",
        "https://api.openai.com/v1",
    ],
)
def test_should_reject_noncanonical_api_base_in_validate_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, api_base: str
):
    from litellm.llms.kimi_code.chat.transformation import KimiCodeApiBaseError

    credentials_path = tmp_path / "kimi-code.json"
    _set_credentials_path(monkeypatch, credentials_path)
    _write_credentials(credentials_path)
    config = KimiCodeChatConfig()

    with pytest.raises(KimiCodeApiBaseError):
        config.validate_environment(
            headers={},
            model="k3",
            messages=[],
            optional_params={},
            litellm_params={},
            api_base=api_base,
        )


@pytest.mark.parametrize(
    "api_base",
    [
        "https://api.moonshot.ai/v1",
        "https://api.openai.com/v1",
    ],
)
def test_should_reject_noncanonical_api_base_in_provider_info(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, api_base: str
):
    from litellm.llms.kimi_code.chat.transformation import KimiCodeApiBaseError

    credentials_path = tmp_path / "kimi-code.json"
    _set_credentials_path(monkeypatch, credentials_path)
    _write_credentials(credentials_path)
    config = KimiCodeChatConfig()

    with pytest.raises(KimiCodeApiBaseError):
        config._get_openai_compatible_provider_info(api_base, None)


@pytest.mark.parametrize(
    "api_base",
    [
        None,
        KIMI_CODE_API_BASE,
        KIMI_CODE_API_BASE + "/",
    ],
)
def test_should_accept_canonical_or_none_api_base(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, api_base
):
    credentials_path = tmp_path / "kimi-code.json"
    _set_credentials_path(monkeypatch, credentials_path)
    _write_credentials(credentials_path)
    config = KimiCodeChatConfig()

    # None and canonical base (with optional trailing slash) are no-ops
    url = config.get_complete_url(
        api_base=api_base,
        api_key=None,
        model="k3",
        optional_params={},
        litellm_params={},
    )
    assert url == KIMI_CODE_CHAT_COMPLETIONS_URL
