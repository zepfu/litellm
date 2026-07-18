"""Behavioral coverage for RR-054 Google and Antigravity provider owners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest

from litellm.llms.anthropic.experimental_pass_through.providers.antigravity import (
    adapter as antigravity_adapter,
)
from litellm.llms.anthropic.experimental_pass_through.providers.google import (
    adapter as google_adapter,
)
from litellm.llms.anthropic.experimental_pass_through.providers.google import (
    process_cache,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.types import Payload


@pytest.fixture(autouse=True)
def _clear_google_provider_process_state() -> None:
    caches = (
        process_cache._google_code_assist_project_cache,
        process_cache._google_code_assist_prime_until_monotonic_by_key,
        process_cache._google_code_assist_prime_quota_by_key,
        process_cache._google_adapter_semaphores,
        process_cache._google_adapter_user_prompt_turn_counters,
        process_cache._codex_google_code_assist_tool_call_name_cache,
        process_cache._codex_google_code_assist_tool_call_arguments_cache,
    )
    for cache in caches:
        cache.clear()
    yield
    for cache in caches:
        cache.clear()


@dataclass
class _FakeResponse:
    body: object
    status_code: int = 200
    text: str = "ok"
    content: bytes = b"{}"

    def json(self) -> object:
        return self.body


def _raise_http_error(*, status_code: int, detail: str) -> object:
    raise AssertionError(f"unexpected HTTP error {status_code}: {detail}")


def _process_runtime(
    *,
    post_json: object,
    now: float = 100.0,
    max_concurrent: int = 2,
) -> process_cache.Runtime:
    return process_cache.Runtime(
        get_target_base=lambda provider: f"https://{provider}.example.test",
        build_headers=lambda **_kwargs: {"Authorization": "Bearer test"},
        validate_egress=lambda **_kwargs: None,
        post_json=post_json,  # type: ignore[arg-type]
        capture_shape=lambda **_kwargs: None,
        clean_value=lambda value: (
            value.strip() if isinstance(value, str) and value.strip() else None
        ),
        raise_http_error=_raise_http_error,
        get_prime_ttl_seconds=lambda: 60.0,
        get_prime_cache_key=lambda token, project: f"{token}:{project}",
        sanitize_quota=lambda body, source: (
            {"source": source, "body": body} if isinstance(body, dict) else None
        ),
        get_max_concurrent=lambda: max_concurrent,
        get_rate_limit_key=lambda model, **_kwargs: f"model:{model or 'default'}",
        monotonic=lambda: now,
        debug_enabled=lambda: False,
        log_info=lambda _message, _value: None,
    )


def test_bound_google_adapter_token_cache_evicts_oldest_entries() -> None:
    cache = {"first": 1, "second": 2, "third": 3}

    process_cache._bound_google_adapter_token_cache(cache, max_size=2)

    assert cache == {"second": 2, "third": 3}
    assert (
        process_cache._google_code_assist_project_cache
        is process_cache.__dict__["_google_code_assist_project_cache"]
    )


@pytest.mark.asyncio
async def test_google_project_cache_loads_once_per_provider_token() -> None:
    post_calls: list[str] = []
    captures: list[tuple[str, str]] = []

    async def post_json(
        *,
        url: str,
        headers: dict[str, str],
        body: Payload,
        timeout: float,
    ) -> _FakeResponse:
        assert headers == {"Authorization": "Bearer test"}
        assert body["metadata"]["pluginType"] == "GEMINI"
        assert timeout == 30.0
        post_calls.append(url)
        return _FakeResponse({"cloudaicompanionProject": " project-1 "})

    runtime = _process_runtime(post_json=post_json)
    runtime = process_cache.Runtime(
        **{
            **runtime.__dict__,
            "capture_shape": lambda **kwargs: captures.append(
                (kwargs["mode"], kwargs["provider"])
            ),
        }
    )

    first = await process_cache._get_or_load_google_code_assist_project(
        "token-1",
        runtime=runtime,
        adapter_provider="antigravity",
    )
    second = await process_cache._get_or_load_google_code_assist_project(
        "token-1",
        runtime=runtime,
        adapter_provider="antigravity",
    )

    assert first == second == "project-1"
    assert post_calls == [
        "https://antigravity.example.test/v1internal:loadCodeAssist"
    ]
    assert captures == [
        ("google_code_assist_loadCodeAssist", "antigravity")
    ]
    assert len(process_cache._google_code_assist_project_cache) == 1


@pytest.mark.asyncio
async def test_google_prime_cache_reuses_sanitized_quota_observation() -> None:
    post_calls: list[str] = []

    async def post_json(
        *,
        url: str,
        headers: dict[str, str],
        body: Payload,
        timeout: float,
    ) -> _FakeResponse:
        assert headers == {"Authorization": "Bearer test"}
        assert body["project"] == "project-1"
        assert timeout == 20.0
        post_calls.append(url)
        return _FakeResponse({"remaining": 7})

    runtime = _process_runtime(post_json=post_json)
    first = await process_cache._prime_google_code_assist_session(
        "token-1",
        "project-1",
        runtime=runtime,
        adapter_provider="antigravity",
    )
    second = await process_cache._prime_google_code_assist_session(
        "token-1",
        "project-1",
        runtime=runtime,
        adapter_provider="antigravity",
    )

    assert first == second == {
        "source": "antigravity_retrieve_user_quota",
        "body": {"remaining": 7},
    }
    assert len(post_calls) == 3
    assert post_calls[0].endswith(":retrieveUserQuota")
    assert post_calls[1].endswith(":fetchAdminControls")
    assert post_calls[2].endswith(":listExperiments")
    assert len(process_cache._google_code_assist_prime_quota_by_key) == 1


def test_google_semaphore_cache_is_lane_scoped_and_bounded() -> None:
    async def post_json(**_kwargs: object) -> _FakeResponse:
        return _FakeResponse({})

    runtime = _process_runtime(post_json=post_json, max_concurrent=3)
    first = process_cache._get_google_adapter_semaphore(
        runtime=runtime,
        rate_limit_key="lane-0",
    )
    same = process_cache._get_google_adapter_semaphore(
        runtime=runtime,
        rate_limit_key="lane-0",
    )
    for index in range(1, 257):
        process_cache._get_google_adapter_semaphore(
            runtime=runtime,
            rate_limit_key=f"lane-{index}",
        )

    assert first is same
    assert first._value == 3
    assert len(process_cache._google_adapter_semaphores) == 256
    assert ("lane-0", 3) not in process_cache._google_adapter_semaphores
    assert ("lane-256", 3) in process_cache._google_adapter_semaphores


def test_google_request_shape_policy_preserves_transform_order() -> None:
    calls: list[tuple[str, Optional[str]]] = []

    def transform(name: str):
        def apply(request_block: Payload) -> Payload:
            assert request_block is request
            calls.append((name, None))
            return {"last_transform": name, name: True}

        return apply

    def apply_generation(
        request_block: Payload,
        *,
        model: Optional[str],
    ) -> Payload:
        assert request_block is request
        calls.append(("generation", model))
        return {"last_transform": "generation", "generation": model}

    async def unused_async(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("unused runtime callback")

    runtime = google_adapter.Runtime(
        split_inline_context_and_prompt=transform("split"),
        compact_followup_contents=transform("followup_contents"),
        trim_followup_tools=transform("followup_tools"),
        compact_oversized_text_parts=transform("oversized_text"),
        apply_contents_window_policy=transform("contents_window"),
        repair_function_call_adjacency=transform("adjacency"),
        apply_generation_config_policy=apply_generation,
        build_request=unused_async,  # type: ignore[arg-type]
        load_access_token=unused_async,  # type: ignore[arg-type]
        get_project=unused_async,  # type: ignore[arg-type]
        prime_session=unused_async,  # type: ignore[arg-type]
        prepare_adapter_request=unused_async,  # type: ignore[arg-type]
        collect_model_response=unused_async,  # type: ignore[arg-type]
        translate_anthropic_response=lambda _response, **_kwargs: None,
        build_anthropic_response=lambda response: response,
    )
    request: Payload = {"contents": []}

    changes = google_adapter._apply_google_adapter_request_shape_policy(
        {"model": "gemini-test", "request": request},
        runtime=runtime,
    )

    assert calls == [
        ("split", None),
        ("followup_contents", None),
        ("followup_tools", None),
        ("oversized_text", None),
        ("contents_window", None),
        ("adjacency", None),
        ("generation", "gemini-test"),
    ]
    assert changes["last_transform"] == "generation"
    assert all(changes[name] is True for name, _model in calls[:-1])
    assert changes["generation"] == "gemini-test"


def test_google_request_shape_policy_ignores_missing_request_block() -> None:
    async def unused_async(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("unused runtime callback")

    def unused_sync(_payload: Payload) -> Payload:
        raise AssertionError("unused runtime callback")

    runtime = google_adapter.Runtime(
        split_inline_context_and_prompt=unused_sync,
        compact_followup_contents=unused_sync,
        trim_followup_tools=unused_sync,
        compact_oversized_text_parts=unused_sync,
        apply_contents_window_policy=unused_sync,
        repair_function_call_adjacency=unused_sync,
        apply_generation_config_policy=lambda _payload, **_kwargs: {},
        build_request=unused_async,  # type: ignore[arg-type]
        load_access_token=unused_async,  # type: ignore[arg-type]
        get_project=unused_async,  # type: ignore[arg-type]
        prime_session=unused_async,  # type: ignore[arg-type]
        prepare_adapter_request=unused_async,  # type: ignore[arg-type]
        collect_model_response=unused_async,  # type: ignore[arg-type]
        translate_anthropic_response=lambda _response, **_kwargs: None,
        build_anthropic_response=lambda response: response,
    )

    assert (
        google_adapter._apply_google_adapter_request_shape_policy(
            {"request": "not-an-object"},
            runtime=runtime,
        )
        == {}
    )


def test_antigravity_headers_endpoint_and_metadata_shaping() -> None:
    merge_calls: list[tuple[list[str], Payload]] = []
    observability_calls: list[tuple[object, Payload]] = []

    def merge_metadata(
        request_body: Payload,
        *,
        tags_to_add: list[str],
        extra_fields: Payload,
    ) -> Payload:
        merge_calls.append((tags_to_add, extra_fields))
        return {
            **request_body,
            "litellm_metadata": {
                "tags": tags_to_add,
                **extra_fields,
            },
        }

    def prepare_observability(
        *,
        request: object,
        request_body: Payload,
    ) -> Payload:
        observability_calls.append((request, request_body))
        return {**request_body, "observability_prepared": True}

    runtime = antigravity_adapter.Runtime(
        get_client_header=lambda: "antigravity-cli/test",
        merge_metadata=merge_metadata,
        prepare_observability=prepare_observability,
    )
    request = object()

    headers = antigravity_adapter._build_antigravity_native_headers(
        "token-1",
        runtime=runtime,
    )
    shaped = (
        antigravity_adapter._prepare_antigravity_request_body_for_passthrough(
            runtime=runtime,
            request=request,
            request_body={"project": "project-1"},
        )
    )

    assert headers == {
        "Authorization": "Bearer token-1",
        "Content-Type": "application/json",
        "User-Agent": "antigravity-cli/test",
        "x-goog-api-client": "antigravity-cli/test",
        "Accept": "application/json",
    }
    assert merge_calls == [
        (
            [
                "antigravity-code-assist",
                "route:antigravity_code_assist",
            ],
            {
                "client_name": "antigravity-cli",
                "antigravity_code_assist": True,
                "passthrough_route_family": "antigravity_code_assist",
            },
        )
    ]
    observed_body = {
        key: value
        for key, value in shaped.items()
        if key != "observability_prepared"
    }
    assert observability_calls == [(request, observed_body)]
    assert shaped["observability_prepared"] is True


@pytest.mark.parametrize(
    ("endpoint", "expected"),
    [
        ("", "/"),
        ("/", "/"),
        ("v1internal:loadCodeAssist", "/v1internal:loadCodeAssist"),
        ("/v1internal:streamGenerateContent?alt=sse", "/v1internal:streamGenerateContent"),
    ],
)
def test_antigravity_endpoint_normalization(
    endpoint: str,
    expected: str,
) -> None:
    assert (
        antigravity_adapter._normalize_antigravity_endpoint_for_target(endpoint)
        == expected
    )
