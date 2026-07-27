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
