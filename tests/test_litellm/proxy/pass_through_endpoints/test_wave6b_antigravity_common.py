from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

import httpx
import pytest
from fastapi import HTTPException
from starlette.requests import Request

from litellm.proxy._types import ProxyException
from litellm.proxy.pass_through_endpoints.providers import common
from litellm.proxy.pass_through_endpoints.providers.antigravity import runtime


def _clean(value: object) -> Optional[str]:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _format_api_key(value: Optional[str]) -> str:
    cleaned = _clean(value)
    if cleaned is None:
        return ""
    if cleaned.lower().startswith("bearer "):
        return cleaned
    return f"Bearer {cleaned}"


def _runtime(
    *,
    environment: Optional[dict[str, str]] = None,
) -> runtime.Runtime:
    environment = environment or {}

    def merge_metadata(
        request_body: dict[str, Any],
        *,
        tags_to_add: list[str],
        extra_fields: dict[str, Any],
    ) -> dict[str, Any]:
        updated = dict(request_body)
        updated["litellm_metadata"] = {
            "tags": list(tags_to_add),
            **extra_fields,
        }
        return updated

    def prepare_observability(
        *,
        request: object,
        request_body: dict[str, Any],
    ) -> dict[str, Any]:
        assert request is not None
        return {**request_body, "observability_prepared": True}

    return runtime.Runtime(
        clean_value=_clean,
        merge_metadata=merge_metadata,
        prepare_observability=prepare_observability,
        split_provider_prefix=lambda _value: (None, None),
        format_api_key=_format_api_key,
        oauth_error_code=lambda response: _clean(
            response.json().get("error")
        ),
        getenv=environment.get,
    )


def _request(
    *,
    headers: Optional[list[tuple[bytes, bytes]]] = None,
    query_string: bytes = b"",
) -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/",
            "headers": headers or [],
            "query_string": query_string,
        }
    )


def test_antigravity_cli_binary_discovery_preserves_precedence_and_dedup(
    tmp_path: Path,
) -> None:
    environment_binary = tmp_path / "environment-agy"
    default_binary = tmp_path / "default-agy"
    environment_binary.write_bytes(b"agy")
    default_binary.write_bytes(b"agy")
    configured = replace(
        _runtime(
            environment={
                "LITELLM_ANTIGRAVITY_CLI_PATH": str(environment_binary),
                "ANTIGRAVITY_CLI_PATH": str(environment_binary),
            }
        ),
        default_cli_binary_paths=(
            str(environment_binary),
            str(default_binary),
            str(default_binary),
        ),
    )

    assert runtime._iter_antigravity_cli_binary_candidates(
        runtime=configured
    ) == [environment_binary, default_binary]


def test_antigravity_target_headers_and_endpoint_behavior() -> None:
    configured = _runtime(
        environment={
            "ANTIGRAVITY_CODE_ASSIST_ENDPOINT": "https://example.test/root",
            "ANTIGRAVITY_CLI_CODE_ASSIST_ENDPOINT": (
                "https://ignored.example.test"
            ),
            "AAWM_ANTIGRAVITY_CLIENT_HEADER": " custom-client/9 ",
        }
    )

    assert runtime._get_antigravity_passthrough_target_base(
        runtime=configured
    ) == "https://example.test/root"
    assert runtime._get_antigravity_client_header(
        runtime=configured
    ) == "custom-client/9"
    assert runtime._build_antigravity_native_headers(
        "access-token",
        runtime=configured,
    ) == {
        "Authorization": "Bearer access-token",
        "Content-Type": "application/json",
        "User-Agent": "custom-client/9",
        "x-goog-api-client": "custom-client/9",
        "Accept": "application/json",
    }
    assert runtime._ANTIGRAVITY_FORWARD_HEADER_ALLOWLIST == frozenset(
        {
            "accept",
            "authorization",
            "content-type",
            "user-agent",
            "x-goog-api-client",
            "x-goog-fieldmask",
            "x-goog-request-params",
            "x-goog-request-reason",
        }
    )
    assert runtime._join_antigravity_passthrough_url(
        "https://example.test/root/",
        "/v1:generateContent?key=secret",
    ) == "https://example.test/root/v1:generateContent"
    assert runtime._normalize_antigravity_endpoint_for_target("") == "/"

    oauth_request = _request(
        headers=[(b"authorization", b"Bearer ya29.token")]
    )
    assert runtime._request_has_google_oauth_bearer(oauth_request) is True
    assert runtime._get_antigravity_litellm_auth_header(
        oauth_request,
        runtime=configured,
    ) == "Bearer ya29.token"

    key_request = _request(
        headers=[(b"x-litellm-api-key", b"proxy-key")],
        query_string=b"key=query-key",
    )
    assert runtime._get_antigravity_litellm_auth_header(
        key_request,
        runtime=configured,
    ) == "Bearer proxy-key"
    assert runtime._is_antigravity_streaming_endpoint(
        "v1:streamGenerateContent",
        key_request,
    )


def test_antigravity_request_body_and_logging_metadata_behavior() -> None:
    configured = _runtime()
    request = _request()
    source_body = {"project": " project-1 ", "request": {"contents": []}}

    prepared = runtime._prepare_antigravity_request_body_for_passthrough(
        request=request,
        request_body=source_body,
        runtime=configured,
    )

    assert source_body == {
        "project": " project-1 ",
        "request": {"contents": []},
    }
    assert prepared == {
        **source_body,
        "litellm_metadata": {
            "tags": [
                "antigravity-code-assist",
                "route:antigravity_code_assist",
            ],
            "client_name": "antigravity-cli",
            "antigravity_code_assist": True,
            "passthrough_route_family": "antigravity_code_assist",
        },
        "observability_prepared": True,
    }
    assert runtime._get_antigravity_request_project(
        source_body,
        runtime=configured,
    ) == "project-1"
    assert runtime._get_antigravity_passthrough_logging_metadata(
        request,
        runtime=configured,
    ) == {
        "tags": [
            "antigravity-code-assist",
            "route:antigravity_code_assist",
        ],
        "client_name": "antigravity-cli",
        "antigravity_code_assist": True,
        "passthrough_route_family": "antigravity_code_assist",
    }


def test_antigravity_refresh_error_formatting_preserves_vocabulary() -> None:
    configured = _runtime()
    response = httpx.Response(
        400,
        json={"error": "invalid_grant"},
    )

    assert runtime._format_antigravity_oauth_refresh_failure_detail(
        response=response,
        runtime=configured,
    ) == (
        "Failed to refresh Antigravity OAuth access token "
        "(status=400, error=invalid_grant). Re-authenticate Antigravity CLI "
        "or configure valid OAuth client environment overrides."
    )


class _ProviderError(Exception):
    def __init__(
        self,
        detail: object,
        *,
        status_code: Optional[int] = None,
        message: Optional[str] = None,
        code: Optional[str] = None,
    ) -> None:
        super().__init__(str(detail))
        self.detail = detail
        self.status_code = status_code
        self.message = message
        self.code = code


def _common_runtime() -> common.Runtime:
    return common.Runtime(
        extract_status_code=lambda exc: getattr(exc, "status_code", None),
        extract_detail=lambda exc: getattr(exc, "detail", None),
    )


def test_provider_candidate_unavailable_detail_extraction() -> None:
    configured = _common_runtime()
    opencode_error = _ProviderError(
        b"invalid API key",
        status_code=401,
        message="OpenCode authentication failed",
        code="unauthorized",
    )
    assert common._opencode_zen_candidate_unavailable_detail(
        opencode_error,
        runtime=configured,
    ) == (
        "OpenCode authentication failed unauthorized invalid API key "
        "b'invalid API key'"
    )

    antigravity_error = HTTPException(
        status_code=500,
        detail={"reason": "Antigravity OAuth credential missing"},
    )
    assert common._antigravity_candidate_unavailable_detail(
        antigravity_error,
        runtime=configured,
    ) == '{"reason": "Antigravity OAuth credential missing"}'
    assert (
        common._antigravity_candidate_unavailable_detail(
            ValueError("Antigravity OAuth credential missing"),
            runtime=configured,
        )
        is None
    )

    codex_error = _ProviderError(
        (
            "Model is not supported when using Codex with a ChatGPT account"
        ),
        status_code=400,
    )
    assert common._codex_native_openai_candidate_unavailable_detail(
        codex_error,
        runtime=configured,
    ) == "Model is not supported when using Codex with a ChatGPT account"

    grok_error = _ProviderError(
        "Grok does not support parameter reasoning_effort",
        status_code=400,
    )
    assert common._grok_native_candidate_unavailable_detail(
        grok_error,
        runtime=configured,
    ) == "Grok does not support parameter reasoning_effort"

    xai_error = _ProviderError(
        {"error": "managed xAI OAuth credential unavailable"}
    )
    assert common._xai_oauth_candidate_unavailable_detail(
        xai_error
    ) == '{"error": "managed xAI OAuth credential unavailable"}'


@pytest.mark.parametrize(
    (
        "raise_candidate",
        "source_error",
        "message_prefix",
        "error_type",
        "status_code",
    ),
    [
        (
            lambda exc, configured: (
                common._raise_opencode_zen_auto_agent_candidate_unavailable(
                    exc
                )
            ),
            ValueError("missing key"),
            (
                "OpenCode Zen auto-agent candidate requires a valid OpenCode "
                "API-key credential: missing key"
            ),
            "rate_limit_error",
            "429",
        ),
        (
            lambda exc, configured: (
                common._raise_antigravity_auto_agent_candidate_unavailable(
                    exc,
                    runtime=configured,
                )
            ),
            HTTPException(
                status_code=500,
                detail="Antigravity OAuth credential missing",
            ),
            (
                "Antigravity auto-agent candidate requires a valid "
                "Antigravity OAuth credential: Antigravity OAuth credential "
                "missing"
            ),
            "invalid_request_error",
            "502",
        ),
        (
            lambda exc, configured: (
                common._raise_codex_native_openai_auto_agent_candidate_unavailable(
                    exc,
                    runtime=configured,
                )
            ),
            _ProviderError(
                (
                    "Model is not supported when using Codex with a ChatGPT "
                    "account"
                ),
                status_code=400,
            ),
            (
                "ChatGPT/Codex native OpenAI auto-agent candidate is "
                "unavailable for this account: Model is not supported when "
                "using Codex with a ChatGPT account"
            ),
            "rate_limit_error",
            "429",
        ),
        (
            lambda exc, configured: (
                common._raise_grok_native_auto_agent_candidate_unavailable(
                    exc,
                    runtime=configured,
                )
            ),
            _ProviderError("Grok native credential unavailable"),
            (
                "Grok native auto-agent candidate requires a valid managed "
                "xAI/Grok credential: Grok native credential unavailable"
            ),
            "rate_limit_error",
            "429",
        ),
        (
            lambda exc, configured: (
                common._raise_xai_oauth_auto_agent_candidate_unavailable(exc)
            ),
            _ProviderError("managed xAI OAuth credential unavailable"),
            (
                "xAI OAuth auto-agent candidate requires a valid managed xAI "
                "OAuth credential: managed xAI OAuth credential unavailable"
            ),
            "rate_limit_error",
            "429",
        ),
    ],
)
def test_provider_candidate_unavailable_raises_preserve_error_shape(
    raise_candidate: Any,
    source_error: Exception,
    message_prefix: str,
    error_type: str,
    status_code: str,
) -> None:
    with pytest.raises(ProxyException) as raised:
        raise_candidate(source_error, _common_runtime())

    proxy_error = raised.value
    assert proxy_error.message == message_prefix
    assert proxy_error.type == error_type
    assert proxy_error.param == "model"
    assert proxy_error.code == status_code
    assert proxy_error.detail == {
        "error": {
            "message": message_prefix,
            "code": "aawm_codex_auto_agent_candidate_unavailable",
        }
    }
    assert proxy_error.__cause__ is source_error
