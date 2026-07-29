"""Direct behavior tests for the Wave 6B OpenCode Zen runtime extraction."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from starlette.requests import Request
from starlette.responses import StreamingResponse

from litellm.llms.anthropic.experimental_pass_through.providers.opencode_zen import (
    normalization,
)
from litellm.proxy.pass_through_endpoints.providers.opencode_zen import (
    runtime as zen_runtime,
)
from litellm.proxy._types import ProxyException


def _assemble_headers(
    *,
    api_key: str | None,
    request: Request,
) -> dict[str, str]:
    headers: dict[str, str] = {}
    if api_key is not None:
        headers = {
            "authorization": f"Bearer {api_key}",
            "api-key": api_key,
        }
    if (
        "thread" in request.url.path or "assistant" in request.url.path
    ) and "OpenAI-Beta" not in headers:
        headers["OpenAI-Beta"] = "assistants=v2"
    return headers


def _normalize_endpoint_for_target(
    endpoint: str,
    base_target_url: str,
) -> str:
    normalized_endpoint = httpx.URL(endpoint).path
    if not normalized_endpoint.startswith("/"):
        normalized_endpoint = "/" + normalized_endpoint
    if (
        httpx.URL(base_target_url).path.rstrip("/") == "/v1"
        and normalized_endpoint.startswith("/v1/")
    ):
        return normalized_endpoint[len("/v1"):]
    return normalized_endpoint


def _join_url_paths(
    base_url: httpx.URL,
    path: str,
    _provider: str,
) -> str:
    if not base_url.path or base_url.path == "/":
        return str(base_url.copy_with(path=path))
    return str(
        base_url.copy_with(
            path=f"{base_url.path.rstrip('/')}/{path.lstrip('/')}"
        )
    )


def _clean_secret(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


async def _iterate_events(iterator: object) -> AsyncIterator[object]:
    assert isinstance(iterator, list)
    for event in iterator:
        yield event


def _meaningful_output(item: object) -> bool:
    if not isinstance(item, dict):
        return False
    if item.get("type") != "message":
        return item.get("type") in {"function_call", "mcp_call"}
    return bool(item.get("content"))


def _normalization_runtime() -> normalization.Runtime:
    return normalization.Runtime(
        clean_secret_string=_clean_secret,
        merge_metadata=lambda body, **_kwargs: body,
        add_logging_metadata=lambda body, **_kwargs: body,
        build_span=lambda **kwargs: kwargs,
        transform_responses_api_request_to_chat_completion_request=(
            lambda **_kwargs: {}
        ),
        async_responses_api_session_handler=_async_empty_payload,
        iterate_responses_sse_events=_iterate_events,
        coerce_namespace_to_mapping=lambda value: value,
        responses_output_item_has_meaningful_content=_meaningful_output,
        streaming_response_factory=StreamingResponse,
    )


async def _async_empty_payload(**_kwargs: Any) -> dict[str, Any]:
    return {}


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


@pytest.fixture(autouse=True)
def configured_runtime(monkeypatch: pytest.MonkeyPatch):
    """Configure a test runtime and restore the prior singleton on teardown."""
    prior_runtime = zen_runtime._runtime

    for name in (
        "LITELLM_OPENCODE_API_KEY",
        "OPENCODE_API_KEY",
        "LITELLM_OPENCODE_AUTH_FILE",
        "OPENCODE_AUTH_FILE",
        "OPENCODE_ZEN_API_BASE",
        "AAWM_OPENCODE_ZEN_API_BASE",
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
            extract_exception_detail=lambda exc: getattr(
                exc, "detail", None
            ),
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
            get_anthropic_adapter_model_candidates=lambda body: [
                body["model"]
            ]
            if isinstance(body.get("model"), str)
            else [],
        )
    )

    yield

    # Restore the prior runtime singleton so test order does not matter.
    zen_runtime._runtime = prior_runtime
    zen_runtime._get_anthropic_opencode_zen_normalization_runtime.cache_clear()


@pytest.mark.asyncio
async def test_credential_resolution_prefers_explicit_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LITELLM_OPENCODE_API_KEY", '  "zen-key"  ')

    assert await zen_runtime._load_local_opencode_zen_api_key() == "zen-key"


@pytest.mark.asyncio
async def test_credential_resolution_reads_opencode_auth_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps(
            {"opencode": {"type": "api", "key": "file-key"}}
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("LITELLM_OPENCODE_AUTH_FILE", str(auth_path))

    assert await zen_runtime._load_local_opencode_zen_api_key() == "file-key"


@pytest.mark.asyncio
async def test_candidate_loader_uses_monkeypatchable_local_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _patched_loader() -> str:
        return "patched-key"

    monkeypatch.setattr(
        zen_runtime,
        "_load_local_opencode_zen_api_key",
        _patched_loader,
    )

    assert (
        await zen_runtime._load_opencode_zen_api_key_for_candidate()
        == "patched-key"
    )


@pytest.mark.asyncio
async def test_headers_target_and_url_join(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENCODE_API_KEY", "header-key")
    monkeypatch.setenv(
        "OPENCODE_ZEN_API_BASE",
        '"https://zen.example.test/custom/v1/"',
    )
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/responses",
            "headers": [],
            "query_string": b"",
            "server": ("testserver", 80),
            "scheme": "http",
        }
    )

    target = zen_runtime._get_opencode_zen_target_base()
    headers = await zen_runtime._build_opencode_zen_headers(request)

    assert target == "https://zen.example.test/custom"
    assert headers == {
        "authorization": "Bearer header-key",
        "api-key": "header-key",
    }
    assert (
        zen_runtime._join_opencode_zen_passthrough_url(
            target, "/v1/responses"
        )
        == "https://zen.example.test/custom/v1/responses"
    )


def test_unavailable_detail_delegates_to_common_owner() -> None:
    """OpenCode unavailable detail is owned by providers/common.py."""
    from litellm.proxy.pass_through_endpoints.providers import common

    configured = common.Runtime(
        extract_status_code=lambda exc: getattr(exc, "status_code", None),
        extract_detail=lambda exc: getattr(exc, "detail", None),
    )
    billing_error = SimpleNamespace(
        status_code=402,
        detail={"error": "No payment method; billing required"},
        message="OpenCode request failed",
        code="payment_required",
    )
    unrelated_error = SimpleNamespace(
        status_code=500,
        detail="upstream socket closed",
    )

    billing_detail = common._opencode_zen_candidate_unavailable_detail(
        billing_error,
        runtime=configured,
    )

    assert billing_detail is not None
    assert "billing" in billing_detail.lower()
    assert (
        common._opencode_zen_candidate_unavailable_detail(
            unrelated_error,
            runtime=configured,
        )
        is None
    )


def test_model_resolution_owned_by_canonical_module() -> None:
    """Model normalize/resolve is owned by aawm_adapter_runtime/model_resolution."""
    from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime import (
        model_resolution,
    )

    # The canonical module defines these functions
    assert hasattr(model_resolution, "_normalize_opencode_zen_adapter_model_name")
    assert hasattr(model_resolution, "_resolve_codex_opencode_zen_adapter_model")
    assert hasattr(model_resolution, "_resolve_anthropic_opencode_zen_adapter_model")

    # The zen runtime must NOT define its own copies
    assert not hasattr(zen_runtime, "_normalize_opencode_zen_adapter_model_name")
    assert not hasattr(zen_runtime, "_resolve_codex_opencode_zen_adapter_model")
    assert not hasattr(zen_runtime, "_resolve_anthropic_opencode_zen_adapter_model")
    assert not hasattr(zen_runtime, "_split_opencode_zen_provider_prefix")


def test_install_order_identity_model_resolution_wins() -> None:
    """Production install order: zen install() must NOT overwrite model_resolution.

    In the god module, _wave6b_opencode_zen_runtime.install(globals()) runs
    first, then _aawm_adapter_model_resolution.install(globals()) runs later.
    The zen runtime's _HOST_FUNCTION_NAMES must not include model-resolution
    names, so model_resolution's rebound functions are the final owners.
    """
    from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime import (
        model_resolution,
    )

    model_resolution_names = set(model_resolution._HOST_FUNCTION_NAMES)
    zen_published_names = set(zen_runtime._HOST_FUNCTION_NAMES)

    # No overlap: zen must not publish model-resolution functions
    overlap = model_resolution_names & zen_published_names
    assert not overlap, (
        f"zen runtime install() would overwrite model_resolution owners: "
        f"{sorted(overlap)}"
    )

    # Specifically verify the three OpenCode model functions are in
    # model_resolution but NOT in zen
    opencode_model_names = {
        "_normalize_opencode_zen_adapter_model_name",
        "_resolve_codex_opencode_zen_adapter_model",
        "_resolve_anthropic_opencode_zen_adapter_model",
    }
    assert opencode_model_names.issubset(model_resolution_names)
    assert not (opencode_model_names & zen_published_names)


def test_install_observes_late_generic_exception_helper_patches() -> None:
    prior_runtime = zen_runtime._runtime
    host_globals: dict[str, Any] = {
        "_extract_adapter_exception_status_code": lambda exc: 418,
        "_extract_adapter_exception_detail": lambda exc: "initial-detail",
    }
    zen_runtime.install(host_globals)

    try:
        runtime = zen_runtime._require_runtime()
        exc = RuntimeError("boom")
        assert runtime.extract_exception_status_code(exc) == 418
        assert runtime.extract_exception_detail(exc) == "initial-detail"

        host_globals["_extract_adapter_exception_status_code"] = lambda exc: 429
        host_globals["_extract_adapter_exception_detail"] = (
            lambda exc: "patched-detail"
        )

        assert runtime.extract_exception_status_code(exc) == 429
        assert runtime.extract_exception_detail(exc) == "patched-detail"
    finally:
        zen_runtime._runtime = prior_runtime
        zen_runtime._get_anthropic_opencode_zen_normalization_runtime.cache_clear()


@pytest.mark.asyncio
async def test_responses_stream_normalization_for_codex() -> None:
    response = SimpleNamespace(
        body_iterator=[
            {
                "type": "response.output_text.delta",
                "response_id": "resp_zen",
                "item_id": "msg_zen",
                "model": "big-pickle",
                "delta": "hello",
            },
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_zen",
                    "model": "big-pickle",
                    "status": "completed",
                },
            },
        ],
        headers={"x-provider": "opencode"},
        status_code=200,
    )

    chunks = [
        chunk
        async for chunk in (
            zen_runtime._normalize_opencode_zen_responses_stream_for_codex(
                response,
                adapter_model="big-pickle",
            )
        )
    ]
    joined = "".join(chunks)

    assert "event: response.created" in joined
    assert "event: response.output_item.added" in joined
    assert '"delta":"hello"' in joined
    assert "event: response.output_text.done" in joined
    assert '"text":"hello"' in joined
    assert "event: response.completed" in joined
    assert joined.endswith("data: [DONE]\n\n")


def test_build_codex_streaming_response_preserves_transport_fields() -> None:
    response = SimpleNamespace(
        body_iterator=[],
        headers={"x-provider": "opencode"},
        status_code=202,
    )

    normalized = (
        zen_runtime._build_codex_opencode_zen_streaming_response(
            response,
            adapter_model="big-pickle",
        )
    )

    assert normalized.status_code == 202
    assert normalized.headers["x-provider"] == "opencode"
    assert normalized.media_type == "text/event-stream"


# ---------------------------------------------------------------------------
# D1-545: Auth fail-closed tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_api_key_env_precedence_litellm_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LITELLM_OPENCODE_API_KEY takes precedence over OPENCODE_API_KEY."""
    monkeypatch.setenv("LITELLM_OPENCODE_API_KEY", "primary-key")
    monkeypatch.setenv("OPENCODE_API_KEY", "secondary-key")

    assert await zen_runtime._load_local_opencode_zen_api_key() == "primary-key"


@pytest.mark.asyncio
async def test_api_key_env_precedence_second_used_when_first_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENCODE_API_KEY", "fallback-key")

    assert await zen_runtime._load_local_opencode_zen_api_key() == "fallback-key"


@pytest.mark.asyncio
async def test_auth_file_env_precedence_litellm_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LITELLM_OPENCODE_AUTH_FILE takes precedence over OPENCODE_AUTH_FILE."""
    primary = tmp_path / "primary.json"
    secondary = tmp_path / "secondary.json"
    primary.write_text(json.dumps({"opencode": {"type": "api", "key": "pk"}}))
    secondary.write_text(json.dumps({"opencode": {"type": "api", "key": "sk"}}))
    monkeypatch.setenv("LITELLM_OPENCODE_AUTH_FILE", str(primary))
    monkeypatch.setenv("OPENCODE_AUTH_FILE", str(secondary))

    assert await zen_runtime._load_local_opencode_zen_api_key() == "pk"


@pytest.mark.asyncio
async def test_explicit_missing_auth_file_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A configured but missing auth file must fail without trying defaults."""
    missing = tmp_path / "nonexistent.json"
    monkeypatch.setenv("LITELLM_OPENCODE_AUTH_FILE", str(missing))

    with pytest.raises(ValueError, match="LITELLM_OPENCODE_AUTH_FILE"):
        await zen_runtime._load_local_opencode_zen_api_key()


@pytest.mark.asyncio
async def test_explicit_non_file_auth_path_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A directory configured as auth file must fail."""
    monkeypatch.setenv("LITELLM_OPENCODE_AUTH_FILE", str(tmp_path))

    with pytest.raises(ValueError, match="LITELLM_OPENCODE_AUTH_FILE"):
        await zen_runtime._load_local_opencode_zen_api_key()


@pytest.mark.asyncio
async def test_explicit_unreadable_auth_file_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unreadable auth file must fail with sanitized message."""
    auth_file = tmp_path / "auth.json"
    auth_file.write_text("{}")
    monkeypatch.setenv("LITELLM_OPENCODE_AUTH_FILE", str(auth_file))

    original_read = Path.read_text

    def _deny_read(self: Path, *args: Any, **kwargs: Any) -> str:
        if self == auth_file:
            raise PermissionError("denied")
        return original_read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _deny_read)

    with pytest.raises(ValueError, match="not readable") as exc_info:
        await zen_runtime._load_local_opencode_zen_api_key()
    # Sanitized: no path value, no raw exception text
    assert str(auth_file) not in str(exc_info.value)
    assert "denied" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_explicit_malformed_json_auth_file_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth_file = tmp_path / "auth.json"
    auth_file.write_text("{not json")
    monkeypatch.setenv("LITELLM_OPENCODE_AUTH_FILE", str(auth_file))

    with pytest.raises(ValueError, match="valid JSON") as exc_info:
        await zen_runtime._load_local_opencode_zen_api_key()
    assert "{not json" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_explicit_invalid_auth_shape_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auth file with wrong provider shape must fail."""
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(json.dumps({"other_provider": {"key": "x"}}))
    monkeypatch.setenv("LITELLM_OPENCODE_AUTH_FILE", str(auth_file))

    with pytest.raises(ValueError, match="API-key auth"):
        await zen_runtime._load_local_opencode_zen_api_key()


@pytest.mark.asyncio
async def test_explicit_auth_type_not_api_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(
        json.dumps({"opencode": {"type": "oauth", "key": "tok"}})
    )
    monkeypatch.setenv("LITELLM_OPENCODE_AUTH_FILE", str(auth_file))

    with pytest.raises(ValueError, match="API-key auth"):
        await zen_runtime._load_local_opencode_zen_api_key()


@pytest.mark.asyncio
async def test_unconfigured_default_discovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When no auth-file env is set, HOME-relative defaults are used."""
    default_dir = tmp_path / ".local" / "share" / "opencode"
    default_dir.mkdir(parents=True)
    default_auth = default_dir / "auth.json"
    default_auth.write_text(
        json.dumps({"opencode": {"type": "api", "key": "default-key"}})
    )
    monkeypatch.setenv("HOME", str(tmp_path))

    assert await zen_runtime._load_local_opencode_zen_api_key() == "default-key"


@pytest.mark.asyncio
async def test_no_default_auth_file_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When no auth-file env and no default file exists, fail clearly."""
    monkeypatch.setenv("HOME", str(tmp_path / "empty_home"))

    with pytest.raises(FileNotFoundError, match="auth file not found"):
        await zen_runtime._load_local_opencode_zen_api_key()


@pytest.mark.asyncio
async def test_absolute_container_style_auth_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Absolute paths (container mounts) work without HOME expansion."""
    container_path = tmp_path / "secrets" / "opencode" / "auth.json"
    container_path.parent.mkdir(parents=True)
    container_path.write_text(
        json.dumps({"opencode": {"type": "api", "key": "container-key"}})
    )
    monkeypatch.setenv("OPENCODE_AUTH_FILE", str(container_path))

    assert await zen_runtime._load_local_opencode_zen_api_key() == "container-key"


@pytest.mark.asyncio
async def test_error_messages_do_not_expose_path_or_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Errors may name the env var but never the path value or content."""
    secret_path = tmp_path / "super-secret-location.json"
    secret_path.write_text("SENSITIVE_CONTENT")
    monkeypatch.setenv("LITELLM_OPENCODE_AUTH_FILE", str(secret_path))
    # Make it a directory to trigger missing-file error
    secret_path.unlink()
    secret_path.mkdir()

    with pytest.raises(ValueError) as exc_info:
        await zen_runtime._load_local_opencode_zen_api_key()
    msg = str(exc_info.value)
    assert "LITELLM_OPENCODE_AUTH_FILE" in msg
    assert str(secret_path) not in msg
    assert "SENSITIVE_CONTENT" not in msg


@pytest.mark.asyncio
async def test_first_auth_env_authoritative_no_later_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If LITELLM_OPENCODE_AUTH_FILE is set but missing, OPENCODE_AUTH_FILE
    is NOT consulted."""
    missing = tmp_path / "missing.json"
    valid = tmp_path / "valid.json"
    valid.write_text(json.dumps({"opencode": {"type": "api", "key": "v"}}))
    monkeypatch.setenv("LITELLM_OPENCODE_AUTH_FILE", str(missing))
    monkeypatch.setenv("OPENCODE_AUTH_FILE", str(valid))

    with pytest.raises(ValueError, match="LITELLM_OPENCODE_AUTH_FILE"):
        await zen_runtime._load_local_opencode_zen_api_key()


# ---------------------------------------------------------------------------
# D1-546: Tool policy tests
# ---------------------------------------------------------------------------


def _tool_policy_runtime() -> normalization.Runtime:
    """Minimal runtime for tool-policy tests."""
    return normalization.Runtime(
        clean_secret_string=_clean_secret,
        merge_metadata=_merge_metadata_for_tools,
        add_logging_metadata=lambda body, **_kwargs: body,
        build_span=lambda **kwargs: kwargs,
        transform_responses_api_request_to_chat_completion_request=(
            lambda **_kwargs: {}
        ),
        async_responses_api_session_handler=_async_empty_payload,
        iterate_responses_sse_events=_iterate_events,
        coerce_namespace_to_mapping=lambda value: value,
        responses_output_item_has_meaningful_content=_meaningful_output,
        streaming_response_factory=StreamingResponse,
    )


def _codex_normalization_runtime() -> normalization.Runtime:
    def _transform(**kwargs: Any) -> dict[str, Any]:
        responses_request = kwargs["responses_api_request"]
        tool_choice = responses_request.get("tool_choice")
        if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            tool_choice = "required"
        return (
            {"tool_choice": tool_choice}
            if tool_choice is not None
            else {}
        )

    return normalization.Runtime(
        clean_secret_string=_clean_secret,
        merge_metadata=_merge_metadata_for_tools,
        add_logging_metadata=lambda body, **_kwargs: body,
        build_span=lambda **kwargs: kwargs,
        transform_responses_api_request_to_chat_completion_request=_transform,
        async_responses_api_session_handler=_async_empty_payload,
        iterate_responses_sse_events=_iterate_events,
        coerce_namespace_to_mapping=lambda value: value,
        responses_output_item_has_meaningful_content=_meaningful_output,
        streaming_response_factory=StreamingResponse,
    )


def _merge_metadata_for_tools(
    body: dict[str, Any],
    *,
    tags_to_add: list[str] | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    updated = dict(body)
    meta = dict(updated.get("litellm_metadata") or {})
    if tags_to_add:
        meta.setdefault("tags", []).extend(tags_to_add)
    if extra_fields:
        meta.update(extra_fields)
    updated["litellm_metadata"] = meta
    return updated


def _fn_tool(name: str) -> dict[str, Any]:
    return {"type": "function", "name": name, "parameters": {}}


def _hosted_tool(tool_type: str = "web_search") -> dict[str, Any]:
    return {"type": tool_type}


def test_strict_mode_default_rejects_non_function_tools() -> None:
    """Default (no mode metadata) is strict: rejects unsupported tools."""
    rt = _tool_policy_runtime()
    body: dict[str, Any] = {
        "tools": [_fn_tool("read"), _hosted_tool("web_search")],
    }
    with pytest.raises(ProxyException) as exc_info:
        normalization.strip_unsupported_responses_tools(rt, body)
    assert str(exc_info.value.code) == "400"


def test_strict_mode_rejects_malformed_function_tool() -> None:
    """A function tool without a name is malformed and rejected in strict."""
    rt = _tool_policy_runtime()
    body: dict[str, Any] = {
        "tools": [{"type": "function", "parameters": {}}],
    }
    with pytest.raises(ProxyException) as exc_info:
        normalization.strip_unsupported_responses_tools(rt, body)
    assert str(exc_info.value.code) == "400"


def test_strict_mode_does_not_mutate_caller_body() -> None:
    rt = _tool_policy_runtime()
    original_tools = [_fn_tool("a"), _hosted_tool()]
    body: dict[str, Any] = {"tools": list(original_tools)}
    with pytest.raises(ProxyException):
        normalization.strip_unsupported_responses_tools(rt, body)
    assert body["tools"] == original_tools


def test_drop_mode_retains_function_tools_and_records_metadata() -> None:
    rt = _tool_policy_runtime()
    body: dict[str, Any] = {
        "tools": [_fn_tool("read"), _hosted_tool("code_interpreter"), _fn_tool("write")],
        "litellm_metadata": {"opencode_zen_unsupported_tools_mode": "drop"},
    }
    result = normalization.strip_unsupported_responses_tools(rt, body)
    assert len(result["tools"]) == 2
    meta = result["litellm_metadata"]
    assert meta["opencode_zen_removed_unsupported_tool_count"] == 1
    assert meta["opencode_zen_removed_unsupported_tool_types"] == [
        "code_interpreter"
    ]
    assert meta["opencode_zen_removed_unsupported_tool_names"] == []


def test_drop_mode_all_unsupported_removes_tools_key() -> None:
    rt = _tool_policy_runtime()
    body: dict[str, Any] = {
        "tools": [_hosted_tool("web_search"), _hosted_tool("code_interpreter")],
        "litellm_metadata": {"opencode_zen_unsupported_tools_mode": "drop"},
    }
    result = normalization.strip_unsupported_responses_tools(rt, body)
    assert "tools" not in result
    meta = result["litellm_metadata"]
    assert meta["opencode_zen_removed_unsupported_tool_count"] == 2


def test_drop_mode_tool_choice_auto_omitted_when_no_functions() -> None:
    rt = _tool_policy_runtime()
    body: dict[str, Any] = {
        "tools": [_hosted_tool()],
        "tool_choice": "auto",
        "litellm_metadata": {"opencode_zen_unsupported_tools_mode": "drop"},
    }
    result = normalization.strip_unsupported_responses_tools(rt, body)
    assert "tool_choice" not in result


def test_empty_tools_auto_omitted_without_caller_mutation() -> None:
    rt = _tool_policy_runtime()
    body: dict[str, Any] = {
        "tools": [],
        "tool_choice": "auto",
    }

    result = normalization.strip_unsupported_responses_tools(rt, body)

    assert result == {"tools": []}
    assert result is not body
    assert body == {
        "tools": [],
        "tool_choice": "auto",
    }


@pytest.mark.asyncio
async def test_normalize_codex_request_preserves_named_function_choice() -> None:
    rt = _codex_normalization_runtime()
    body: dict[str, Any] = {
        "model": "opencode/big-pickle",
        "input": "test",
        "tools": [_fn_tool("read")],
        "tool_choice": {"type": "function", "name": "read"},
    }

    result = await normalization.normalize_codex_request(
        rt,
        body,
        adapter_model="big-pickle",
    )

    assert result.completion_kwargs["tool_choice"] == {
        "type": "function",
        "function": {"name": "read"},
    }
    assert body["tool_choice"] == {"type": "function", "name": "read"}
    assert body["tools"] == [_fn_tool("read")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tools", "tool_choice"),
    [
        (
            [_fn_tool("read")],
            {"type": "function", "name": "missing"},
        ),
        (
            [_fn_tool("read")],
            {"type": "web_search", "name": "search"},
        ),
        (
            None,
            {"type": "function", "name": "read"},
        ),
        (None, "required"),
        (None, "any"),
        (None, "tool"),
    ],
)
async def test_normalize_codex_request_rejects_unsatisfied_tool_choice(
    tools: list[dict[str, Any]] | None,
    tool_choice: object,
) -> None:
    rt = _codex_normalization_runtime()
    body: dict[str, Any] = {
        "model": "opencode/big-pickle",
        "input": "test",
        "tool_choice": tool_choice,
    }
    if tools is not None:
        body["tools"] = tools
    original_body = json.loads(json.dumps(body))

    with pytest.raises(ProxyException) as exc_info:
        await normalization.normalize_codex_request(
            rt,
            body,
            adapter_model="big-pickle",
        )

    assert str(exc_info.value.code) == "400"
    assert body == original_body


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_choice", "expected_tool_choice"),
    [
        ("auto", None),
        ("none", "none"),
    ],
)
async def test_normalize_codex_request_handles_no_tools_satisfiable_choices(
    tool_choice: str,
    expected_tool_choice: str | None,
) -> None:
    rt = _codex_normalization_runtime()
    body: dict[str, Any] = {
        "model": "opencode/big-pickle",
        "input": "test",
        "tool_choice": tool_choice,
    }
    original_body = dict(body)

    result = await normalization.normalize_codex_request(
        rt,
        body,
        adapter_model="big-pickle",
    )

    if expected_tool_choice is None:
        assert "tool_choice" not in result.completion_kwargs
    else:
        assert result.completion_kwargs["tool_choice"] == expected_tool_choice
    assert body == original_body


def test_drop_mode_tool_choice_auto_kept_when_functions_remain() -> None:
    rt = _tool_policy_runtime()
    body: dict[str, Any] = {
        "tools": [_fn_tool("x"), _hosted_tool()],
        "tool_choice": "auto",
        "litellm_metadata": {"opencode_zen_unsupported_tools_mode": "drop"},
    }
    result = normalization.strip_unsupported_responses_tools(rt, body)
    assert result["tool_choice"] == "auto"


def test_drop_mode_tool_choice_none_preserved() -> None:
    rt = _tool_policy_runtime()
    body: dict[str, Any] = {
        "tools": [_hosted_tool()],
        "tool_choice": "none",
        "litellm_metadata": {"opencode_zen_unsupported_tools_mode": "drop"},
    }
    result = normalization.strip_unsupported_responses_tools(rt, body)
    assert result["tool_choice"] == "none"


def test_drop_mode_tool_choice_required_rejected_no_functions() -> None:
    rt = _tool_policy_runtime()
    body: dict[str, Any] = {
        "tools": [_hosted_tool()],
        "tool_choice": "required",
        "litellm_metadata": {"opencode_zen_unsupported_tools_mode": "drop"},
    }
    with pytest.raises(ProxyException) as exc_info:
        normalization.strip_unsupported_responses_tools(rt, body)
    assert str(exc_info.value.code) == "400"


def test_drop_mode_tool_choice_any_rejected_no_functions() -> None:
    rt = _tool_policy_runtime()
    body: dict[str, Any] = {
        "tools": [_hosted_tool()],
        "tool_choice": "any",
        "litellm_metadata": {"opencode_zen_unsupported_tools_mode": "drop"},
    }
    with pytest.raises(ProxyException) as exc_info:
        normalization.strip_unsupported_responses_tools(rt, body)
    assert str(exc_info.value.code) == "400"


def test_drop_mode_named_function_choice_retained() -> None:
    rt = _tool_policy_runtime()
    body: dict[str, Any] = {
        "tools": [_fn_tool("read"), _hosted_tool()],
        "tool_choice": {"type": "function", "name": "read"},
        "litellm_metadata": {"opencode_zen_unsupported_tools_mode": "drop"},
    }
    result = normalization.strip_unsupported_responses_tools(rt, body)
    assert result["tool_choice"] == {"type": "function", "name": "read"}


def test_drop_mode_named_function_choice_removed_rejected() -> None:
    rt = _tool_policy_runtime()
    body: dict[str, Any] = {
        "tools": [_fn_tool("read"), _hosted_tool("shell")],
        "tool_choice": {"type": "function", "name": "shell"},
        "litellm_metadata": {"opencode_zen_unsupported_tools_mode": "drop"},
    }
    with pytest.raises(ProxyException) as exc_info:
        normalization.strip_unsupported_responses_tools(rt, body)
    assert str(exc_info.value.code) == "400"


def test_drop_mode_non_function_tool_choice_dict_rejected() -> None:
    rt = _tool_policy_runtime()
    body: dict[str, Any] = {
        "tools": [_fn_tool("a"), _hosted_tool()],
        "tool_choice": {"type": "web_search"},
        "litellm_metadata": {"opencode_zen_unsupported_tools_mode": "drop"},
    }
    with pytest.raises(ProxyException) as exc_info:
        normalization.strip_unsupported_responses_tools(rt, body)
    assert str(exc_info.value.code) == "400"


def test_drop_mode_unknown_tool_choice_string_rejected() -> None:
    rt = _tool_policy_runtime()
    body: dict[str, Any] = {
        "tools": [_fn_tool("a"), _hosted_tool()],
        "tool_choice": "banana",
        "litellm_metadata": {"opencode_zen_unsupported_tools_mode": "drop"},
    }
    with pytest.raises(ProxyException) as exc_info:
        normalization.strip_unsupported_responses_tools(rt, body)
    assert str(exc_info.value.code) == "400"


def test_drop_mode_allowed_tools_rejected() -> None:
    rt = _tool_policy_runtime()
    body: dict[str, Any] = {
        "tools": [_fn_tool("a"), _hosted_tool()],
        "allowed_tools": ["a"],
        "litellm_metadata": {"opencode_zen_unsupported_tools_mode": "drop"},
    }
    with pytest.raises(ProxyException) as exc_info:
        normalization.strip_unsupported_responses_tools(rt, body)
    assert str(exc_info.value.code) == "400"


def test_unknown_mode_yields_400() -> None:
    rt = _tool_policy_runtime()
    body: dict[str, Any] = {
        "tools": [_fn_tool("a"), _hosted_tool()],
        "litellm_metadata": {"opencode_zen_unsupported_tools_mode": "yolo"},
    }
    with pytest.raises(ProxyException) as exc_info:
        normalization.strip_unsupported_responses_tools(rt, body)
    assert str(exc_info.value.code) == "400"
    assert "strict" in str(exc_info.value.message)
    assert "drop" in str(exc_info.value.message)


def test_unknown_mode_without_tools_yields_400() -> None:
    rt = _tool_policy_runtime()
    body: dict[str, Any] = {
        "litellm_metadata": {"opencode_zen_unsupported_tools_mode": "secret-mode"},
    }
    with pytest.raises(ProxyException) as exc_info:
        normalization.strip_unsupported_responses_tools(rt, body)
    assert str(exc_info.value.code) == "400"
    assert "secret-mode" not in str(exc_info.value.message)


@pytest.mark.parametrize("mode", ["strict", "drop"])
def test_malformed_non_list_tools_rejected(mode: str) -> None:
    rt = _tool_policy_runtime()
    body: dict[str, Any] = {
        "tools": {"type": "function", "name": "not-a-list"},
        "litellm_metadata": {"opencode_zen_unsupported_tools_mode": mode},
    }
    with pytest.raises(ProxyException) as exc_info:
        normalization.strip_unsupported_responses_tools(rt, body)
    assert str(exc_info.value.code) == "400"
    assert "not-a-list" not in str(exc_info.value.message)


def test_drop_mode_metadata_no_value_leakage() -> None:
    """Metadata records only bounded type/name, never definitions/schemas."""
    rt = _tool_policy_runtime()
    body: dict[str, Any] = {
        "tools": [
            _fn_tool("safe"),
            {
                "type": "mcp",
                "name": "secret_tool",
                "server_url": "https://internal.corp/mcp",
                "headers": {"Authorization": "Bearer tok123"},
                "schema": {"type": "object", "properties": {"x": {"type": "string"}}},
            },
        ],
        "litellm_metadata": {"opencode_zen_unsupported_tools_mode": "drop"},
    }
    result = normalization.strip_unsupported_responses_tools(rt, body)
    meta = result["litellm_metadata"]
    meta_str = json.dumps(meta)
    assert "internal.corp" not in meta_str
    assert "tok123" not in meta_str
    assert "properties" not in meta_str
    assert meta["opencode_zen_removed_unsupported_tool_types"] == ["mcp"]
    assert meta["opencode_zen_removed_unsupported_tool_names"] == []


def test_drop_mode_metadata_is_bounded_for_many_secret_like_tools() -> None:
    rt = _tool_policy_runtime()
    secret_values = [
        f"secret-name-{index}-" + ("n" * 1000) for index in range(40)
    ]
    secret_types = [
        f"secret-type-{index}-" + ("t" * 1000) for index in range(40)
    ]
    body: dict[str, Any] = {
        "tools": [
            {
                "type": secret_type,
                "name": secret_name,
                "description": "secret-description-" + ("d" * 1000),
                "parameters": {
                    "type": "object",
                    "properties": {"secret": {"type": "string"}},
                },
                "server_url": "https://secret.example.invalid/mcp",
                "arguments": {"token": "secret-token-value"},
            }
            for secret_name, secret_type in zip(secret_values, secret_types)
        ],
        "litellm_metadata": {"opencode_zen_unsupported_tools_mode": "drop"},
    }

    result = normalization.strip_unsupported_responses_tools(rt, body)
    meta = result["litellm_metadata"]
    adaptation_metadata = {
        key: value
        for key, value in meta.items()
        if key.startswith("opencode_zen_removed_unsupported_tool_")
    }
    serialized = json.dumps(adaptation_metadata)

    assert adaptation_metadata == {
        "opencode_zen_removed_unsupported_tool_count": 40,
        "opencode_zen_removed_unsupported_tool_types": ["unknown"],
        "opencode_zen_removed_unsupported_tool_names": [],
    }
    assert len(serialized) < 300
    for secret_value in (*secret_values, *secret_types):
        assert secret_value not in serialized
    assert "secret-description" not in serialized
    assert "secret.example.invalid" not in serialized
    assert "secret-token-value" not in serialized


def test_all_function_tools_no_removal_passthrough() -> None:
    """When all tools are valid functions, body is returned unchanged."""
    rt = _tool_policy_runtime()
    body: dict[str, Any] = {
        "tools": [_fn_tool("a"), _fn_tool("b")],
        "litellm_metadata": {"opencode_zen_unsupported_tools_mode": "strict"},
    }
    result = normalization.strip_unsupported_responses_tools(rt, body)
    assert result is body


def test_non_dict_tool_rejected_in_strict() -> None:
    rt = _tool_policy_runtime()
    body: dict[str, Any] = {
        "tools": ["not_a_dict"],
    }
    with pytest.raises(ProxyException) as exc_info:
        normalization.strip_unsupported_responses_tools(rt, body)
    assert str(exc_info.value.code) == "400"

@pytest.mark.asyncio
async def test_installed_host_facade_patch_reaches_candidate_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A patch on the host facade must reach _build_opencode_zen_headers.

    After Wave 6B extraction the responses/header path resolves the candidate
    loader through the runtime's live-host callback. Simulate the god-module
    ``install(globals())`` facade, then patch the host-bound candidate-key
    function *after* install (as the alias-failover test does on the god
    module), and prove the installed ``_build_opencode_zen_headers`` facade
    observes the patch.
    """
    calls: list[dict[str, Any]] = []

    async def _patched_candidate_key(**kwargs: Any) -> str:
        calls.append(kwargs)
        return "patched-candidate-key"

    # Minimal host globals; install() stores live-lookup lambdas that are only
    # resolved when the runtime functions are actually invoked.
    host_globals: dict[str, Any] = {
        "BaseOpenAIPassThroughHandler": SimpleNamespace(
            _assemble_headers=lambda *, api_key, request: {
                "authorization": f"Bearer {api_key}",
                "api-key": api_key,
            },
        ),
    }
    zen_runtime.install(host_globals)

    try:
        # The facade published into the host namespace is the same object as
        # the runtime module function.
        assert (
            host_globals["_build_opencode_zen_headers"]
            is zen_runtime._build_opencode_zen_headers
        )

        # Patch *after* install, exactly like the failing alias-failover test
        # patches the god-module attribute at test time.
        host_globals["_load_opencode_zen_api_key_for_candidate"] = (
            _patched_candidate_key
        )

        request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/v1/responses",
                "headers": [],
                "query_string": b"",
                "server": ("testserver", 80),
                "scheme": "http",
            }
        )

        headers = await host_globals["_build_opencode_zen_headers"](
            request,
            use_alias_candidate_probe=True,
        )

        assert headers == {
            "authorization": "Bearer patched-candidate-key",
            "api-key": "patched-candidate-key",
        }
        assert calls == [{"use_alias_candidate_probe": True}]
    finally:
        # install() reconfigured the module singleton with a live-host runtime
        # bound to the throwaway facade dict; restore the autouse fixture's
        # standalone runtime so later tests are unaffected.
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
                get_anthropic_adapter_model_candidates=lambda body: [
                    body["model"]
                ]
                if isinstance(body.get("model"), str)
                else [],
            )
        )
