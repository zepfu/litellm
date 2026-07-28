"""Direct tests for the Wave 6B xAI request-preparation author slice."""

from __future__ import annotations

import ast
import inspect
from dataclasses import fields
from pathlib import Path
from collections.abc import Generator
from typing import Any, Callable, Optional, cast

import pytest
from starlette.requests import Request

from litellm.proxy.pass_through_endpoints.providers.xai import request_prep
from litellm.responses.utils import ResponsesAPIRequestUtils


Payload = dict[str, Any]
MODULE_PATH = Path(request_prep.__file__).resolve()
GOD_MODULE_PATH = (
    MODULE_PATH.parents[2] / "llm_passthrough_endpoints.py"
)

MOVED_SYMBOLS = (
    "_is_oa_xai_request_body",
    "_is_grok_native_oauth_request_body",
    "_is_oa_xai_responses_model",
    "_to_xai_native_passthrough_model",
    "_xai_responses_sanitized_tool_changes",
    "_sanitize_xai_responses_request_body",
    "_coerce_grok_native_function_call_arguments_value",
    "_get_anthropic_grok_normalization_runtime",
    "_sanitize_grok_native_function_call_arguments_request_body",
    "_sanitize_grok_native_function_call_arguments_in_place",
    "_sanitize_xai_responses_request_body_in_place",
    "_prepare_oa_xai_passthrough_request",
    "_get_grok_native_oauth_client_version",
    "_get_grok_native_oauth_session_id",
    "_get_grok_native_oauth_request_id",
    "_build_grok_native_oauth_headers",
    "_add_grok_native_oauth_metadata",
    "_prepare_grok_native_oauth_passthrough_request",
)

EXPECTED_SEAM_DISPOSITION = {
    "is_oa_xai_model": "runtime.is_oa_xai_model",
    "is_grok_native_oauth_model": "runtime.is_grok_native_oauth_model",
    "normalize_grok_native_oauth_model": (
        "runtime.normalize_grok_native_oauth_model"
    ),
    "resolve_oa_xai_upstream_model": (
        "runtime.resolve_oa_xai_upstream_model"
    ),
    "prepare_oa_xai_request": "runtime.prepare_oa_xai_request",
    "build_grok_native_oauth_metadata": (
        "runtime.build_grok_native_oauth_metadata"
    ),
    "get_grok_native_oauth_access_token": (
        "runtime.get_grok_native_oauth_access_token"
    ),
    "get_secret_str": "runtime.get_secret_str",
    "uuid4": "runtime.uuid4",
    "_get_model_metadata_entry": "runtime._get_model_metadata_entry",
    "_get_openai_tool_type": "runtime._get_openai_tool_type",
    "_normalize_low_cardinality_tag_value": (
        "runtime._normalize_low_cardinality_tag_value"
    ),
    "_dedupe_sorted_str_list": "runtime._dedupe_sorted_str_list",
    "_merge_litellm_metadata": "runtime._merge_litellm_metadata",
    "_build_langfuse_span_descriptor": (
        "runtime._build_langfuse_span_descriptor"
    ),
    "_drop_unsupported_codex_hosted_tools_from_request_body": (
        "runtime._drop_unsupported_codex_hosted_tools_from_request_body"
    ),
    "_drop_unsupported_codex_request_params_from_request_body": (
        "runtime._drop_unsupported_codex_request_params_from_request_body"
    ),
    "_drop_unsupported_codex_input_items_from_request_body": (
        "runtime._drop_unsupported_codex_input_items_from_request_body"
    ),
    "_drop_tool_choice_without_tools_from_request_body": (
        "runtime._drop_tool_choice_without_tools_from_request_body"
    ),
    "_replace_request_body_in_place": (
        "runtime._replace_request_body_in_place"
    ),
    "_safe_get_request_headers": "runtime._safe_get_request_headers",
    "_get_case_insensitive_header": (
        "runtime._get_case_insensitive_header"
    ),
    "_get_rewrite_input_item_types_for_model": (
        "runtime._get_rewrite_input_item_types_for_model"
    ),
    "_get_grok_passthrough_target_base": (
        "runtime._get_grok_passthrough_target_base"
    ),
    "_sanitize_xai_responses_request_body_in_place": (
        "runtime._sanitize_xai_responses_request_body_in_place"
    ),
}


class HeaderRequest:
    def __init__(self, headers: Optional[dict[str, Any]] = None) -> None:
        self.headers = headers or {}


def _request(headers: Optional[dict[str, Any]] = None) -> Request:
    return cast(Request, HeaderRequest(headers))


def _normalize_tag(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip().lower()


def _dedupe_sorted(values: list[str]) -> list[str]:
    return sorted(set(values))


def _merge_metadata(
    body: Payload,
    *,
    tags_to_add: Optional[list[str]] = None,
    extra_fields: Optional[Payload] = None,
) -> Payload:
    updated = dict(body)
    current = body.get("litellm_metadata")
    metadata = dict(current) if isinstance(current, dict) else {}
    if tags_to_add:
        existing_tags = metadata.get("tags")
        tags = list(existing_tags) if isinstance(existing_tags, list) else []
        metadata["tags"] = tags + list(tags_to_add)
    if extra_fields:
        metadata.update(extra_fields)
    updated["litellm_metadata"] = metadata
    return updated


def _build_span(*, name: str, metadata: Payload) -> Payload:
    return {"name": name, "metadata": metadata}


def _drop_noop(body: Payload) -> tuple[Payload, list[Payload]]:
    return body, []


def _replace_in_place(target: Payload, replacement: Payload) -> None:
    if target is replacement:
        return
    target.clear()
    target.update(replacement)


def _safe_headers(request: Request) -> dict[str, Any]:
    return dict(cast(Any, request).headers)


def _case_insensitive_header(
    headers: dict[str, Any],
    header_name: str,
) -> Optional[str]:
    wanted = header_name.lower()
    for key, value in headers.items():
        if str(key).lower() == wanted and value is not None:
            normalized = str(value).strip()
            if normalized:
                return normalized
    return None


async def _prepared(_: Payload) -> bool:
    return True


async def _token() -> str:
    return "oauth-token"


def _runtime(**overrides: Any) -> request_prep.XAIRequestPrepRuntime:
    values: dict[str, Any] = {
        "is_oa_xai_model": lambda model: (
            isinstance(model, str) and model.startswith("oa-xai/")
        ),
        "is_grok_native_oauth_model": lambda model: model == "grok-native",
        "normalize_grok_native_oauth_model": (
            lambda model: "grok-4" if model == "grok-native" else None
        ),
        "resolve_oa_xai_upstream_model": (
            lambda model: "xai/" + model.removeprefix("oa-xai/")
        ),
        "prepare_oa_xai_request": _prepared,
        "build_grok_native_oauth_metadata": (
            lambda model: {"tags": [f"grok-model:{model}"], "auth_mode": "oauth"}
        ),
        "get_grok_native_oauth_access_token": _token,
        "get_secret_str": lambda _: None,
        "uuid4": lambda: "generated-request-id",
        "_get_model_metadata_entry": lambda _: None,
        "_get_openai_tool_type": lambda tool: cast(
            Optional[str], tool.get("type")
        ),
        "_normalize_low_cardinality_tag_value": _normalize_tag,
        "_dedupe_sorted_str_list": _dedupe_sorted,
        "_merge_litellm_metadata": _merge_metadata,
        "_build_langfuse_span_descriptor": _build_span,
        "_drop_unsupported_codex_hosted_tools_from_request_body": _drop_noop,
        "_drop_unsupported_codex_request_params_from_request_body": _drop_noop,
        "_drop_unsupported_codex_input_items_from_request_body": _drop_noop,
        "_drop_tool_choice_without_tools_from_request_body": _drop_noop,
        "_replace_request_body_in_place": _replace_in_place,
        "_safe_get_request_headers": _safe_headers,
        "_get_case_insensitive_header": _case_insensitive_header,
        "_get_rewrite_input_item_types_for_model": lambda _: set(),
        "_get_grok_passthrough_target_base": lambda: "https://grok.example/v1",
        "_sanitize_xai_responses_request_body_in_place": (
            lambda body: ([], [])
        ),
    }
    values.update(overrides)
    return request_prep.XAIRequestPrepRuntime(**values)


@pytest.fixture(autouse=True)
def _reset_runtime() -> Generator[None, None, None]:
    saved = request_prep._request_prep_runtime
    request_prep.reset_xai_request_prep_runtime_for_tests()
    yield
    request_prep._request_prep_runtime = saved


def _configure(**overrides: Any) -> request_prep.XAIRequestPrepRuntime:
    runtime = _runtime(**overrides)
    request_prep.configure_xai_request_prep_runtime(runtime)
    return runtime


def _map_responses(
    monkeypatch: pytest.MonkeyPatch,
    mapped_body: Payload,
) -> None:
    monkeypatch.setattr(
        request_prep.XAIResponsesAPIConfig,
        "map_openai_params",
        lambda self, body, **kwargs: dict(mapped_body),
    )


def test_missing_runtime_should_fail_clearly() -> None:
    with pytest.raises(
        RuntimeError,
        match="configure_xai_request_prep_runtime",
    ):
        request_prep._is_oa_xai_request_body({"model": "oa-xai/grok"})


def test_model_predicates_should_use_configured_callbacks() -> None:
    calls: list[tuple[str, Any]] = []

    def is_oa(model: Any) -> bool:
        calls.append(("oa", model))
        return model == "oa"

    def is_native(model: Any) -> bool:
        calls.append(("native", model))
        return model == "native"

    _configure(
        is_oa_xai_model=is_oa,
        is_grok_native_oauth_model=is_native,
    )

    assert request_prep._is_oa_xai_request_body({"model": "oa"})
    assert request_prep._is_grok_native_oauth_request_body(
        {"model": "native"}
    )
    assert calls == [("oa", "oa"), ("native", "native")]


def test_responses_model_should_check_public_then_resolved_metadata() -> None:
    calls: list[Any] = []

    def metadata(model: Any) -> Optional[Payload]:
        calls.append(model)
        if model == "xai/grok":
            return {"mode": "responses"}
        return {"mode": "chat"}

    _configure(_get_model_metadata_entry=metadata)

    assert request_prep._is_oa_xai_responses_model("oa-xai/grok")
    assert calls == ["oa-xai/grok", "xai/grok"]


def test_responses_model_should_preserve_resolution_error_branch() -> None:
    metadata_calls: list[Any] = []

    def fail_resolution(_: str) -> str:
        raise ValueError("unmapped")

    def metadata(model: Any) -> Payload:
        metadata_calls.append(model)
        return {"mode": "chat"}

    _configure(
        resolve_oa_xai_upstream_model=fail_resolution,
        _get_model_metadata_entry=metadata,
    )

    assert not request_prep._is_oa_xai_responses_model("oa-xai/grok")
    assert metadata_calls == ["oa-xai/grok"]


def test_non_oa_model_should_not_resolve_or_read_metadata() -> None:
    def unexpected(_: Any) -> Any:
        raise AssertionError("callback must not run")

    _configure(
        is_oa_xai_model=lambda _: False,
        resolve_oa_xai_upstream_model=unexpected,
        _get_model_metadata_entry=unexpected,
    )

    assert not request_prep._is_oa_xai_responses_model("gpt-4")


def test_xai_native_model_conversion_should_preserve_nonmatching_values() -> None:
    assert (
        request_prep._to_xai_native_passthrough_model("xai/grok-4")
        == "grok-4"
    )
    assert (
        request_prep._to_xai_native_passthrough_model("grok-4")
        == "grok-4"
    )
    marker = object()
    assert request_prep._to_xai_native_passthrough_model(marker) is marker


def test_tool_change_reporting_should_capture_type_and_removed_fields() -> None:
    seen_tools: list[Payload] = []

    def tool_type(tool: Payload) -> Optional[str]:
        seen_tools.append(tool)
        return cast(Optional[str], tool.get("type"))

    _configure(
        _get_openai_tool_type=tool_type
    )
    original = [
        {"type": "function", "name": "lookup", "strict": True},
        "web_search",
    ]
    sanitized = [{"type": "function", "name": "lookup"}]

    assert request_prep._xai_responses_sanitized_tool_changes(
        original, sanitized
    ) == [
        {"index": 0, "type": "function", "removed_fields": ["strict"]},
        {"index": 1, "type": "web_search"},
    ]
    assert seen_tools == [original[0]]


def test_sanitizer_should_decode_previous_id_and_report_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoded_id = ResponsesAPIRequestUtils._build_responses_api_response_id(
        custom_llm_provider="xai",
        model_id="grok-4",
        response_id="resp-original",
    )
    original = {
        "model": "xai/grok-4",
        "input": "hello",
        "temperature": 0.2,
        "previous_response_id": encoded_id,
        "tools": [
            {"type": "function", "name": "lookup", "strict": True}
        ],
    }
    mapped = {
        "model": "xai/grok-4",
        "input": "hello",
        "previous_response_id": encoded_id,
        "tools": [{"type": "function", "name": "lookup"}],
    }
    merge_calls: list[tuple[Payload, list[str], Payload]] = []

    def merge(
        body: Payload,
        *,
        tags_to_add: list[str],
        extra_fields: Payload,
    ) -> Payload:
        merge_calls.append((body, tags_to_add, extra_fields))
        return _merge_metadata(
            body,
            tags_to_add=tags_to_add,
            extra_fields=extra_fields,
        )

    _configure(_merge_litellm_metadata=merge)
    _map_responses(monkeypatch, mapped)

    updated, removed, tool_changes = (
        request_prep._sanitize_xai_responses_request_body(original)
    )

    assert removed == ["temperature"]
    assert tool_changes == [
        {"index": 0, "type": "function", "removed_fields": ["strict"]}
    ]
    assert updated["previous_response_id"] == "resp-original"
    assert merge_calls[0][1] == [
        "xai-responses-request-sanitized",
        "xai-responses-previous-response-id-decoded",
        "xai-responses-removed-param:temperature",
        "xai-responses-sanitized-tool:function",
    ]
    extra = merge_calls[0][2]
    assert extra["xai_responses_sanitized_removed_params"] == ["temperature"]
    assert extra["xai_responses_sanitized_tool_count"] == 1
    assert extra["xai_responses_previous_response_id_decoded"] is True
    assert extra["langfuse_spans"] == [
        {
            "name": "xai.responses_request_sanitized",
            "metadata": {
                "removed_params": ["temperature"],
                "tool_count": 1,
                "tool_types": ["function"],
                "previous_response_id_decoded": True,
            },
        }
    ]


def test_sanitizer_should_return_original_identity_when_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = {"model": "xai/grok-4", "input": "hello"}
    _configure(
        _merge_litellm_metadata=lambda *args, **kwargs: (
            pytest.fail("metadata callback must not run")
        )
    )
    _map_responses(monkeypatch, body)

    updated, removed, changes = (
        request_prep._sanitize_xai_responses_request_body(body)
    )

    assert updated is body
    assert removed == []
    assert changes == []


def test_sanitizer_should_propagate_mapping_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure()

    def fail_map(self: Any, body: Any, **kwargs: Any) -> Payload:
        raise ValueError("map failed")

    monkeypatch.setattr(
        request_prep.XAIResponsesAPIConfig,
        "map_openai_params",
        fail_map,
    )

    with pytest.raises(ValueError, match="map failed"):
        request_prep._sanitize_xai_responses_request_body(
            {"model": "xai/grok"}
        )


@pytest.mark.asyncio
async def test_oa_xai_not_prepared_should_return_false_without_stripping() -> None:
    calls: list[Payload] = []

    async def not_prepared(body: Payload) -> bool:
        calls.append(dict(body))
        return False

    body: Payload = {"model": "other", "api_key": "keep"}
    _configure(
        is_oa_xai_model=lambda _: False,
        prepare_oa_xai_request=not_prepared,
    )

    assert await request_prep._prepare_oa_xai_passthrough_request(body) == (
        False,
        None,
        None,
    )
    assert body == {"model": "other", "api_key": "keep"}
    assert calls == [body]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("api_base", "api_key", "expected_base", "expected_key"),
    [
        (" https://x.ai ", " token ", " https://x.ai ", " token "),
        ("", "   ", None, None),
        (123, ["token"], None, None),
        (None, None, None, None),
    ],
)
async def test_oa_xai_should_strip_credentials_with_exact_value_semantics(
    api_base: Any,
    api_key: Any,
    expected_base: Optional[str],
    expected_key: Optional[str],
) -> None:
    async def prepare(body: Payload) -> bool:
        body.update(
            {
                "api_base": api_base,
                "api_key": api_key,
                "custom_llm_provider": "xai",
            }
        )
        return True

    body: Payload = {"model": "oa-xai/grok"}
    _configure(prepare_oa_xai_request=prepare)

    result = await request_prep._prepare_oa_xai_passthrough_request(body)

    assert result == (True, expected_base, expected_key)
    assert body == {
        "model": "oa-xai/grok",
        "litellm_metadata": {},
    }


@pytest.mark.asyncio
async def test_oa_xai_sanitize_callbacks_should_run_in_exact_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    async def prepare(body: Payload) -> bool:
        events.append("prepare")
        body.update(
            {
                "model": "xai/grok",
                "api_base": "base",
                "api_key": "key",
                "custom_llm_provider": "xai",
            }
        )
        return True

    def drop(name: str) -> Callable[[Payload], tuple[Payload, list[Payload]]]:
        def callback(body: Payload) -> tuple[Payload, list[Payload]]:
            events.append(name)
            updated = dict(body)
            updated[f"{name}_done"] = True
            return updated, [{"type": name}]

        return callback

    def replace(target: Payload, updated: Payload) -> None:
        events.append("replace")
        _replace_in_place(target, updated)

    def sanitize(_: Payload) -> tuple[list[str], list[Payload]]:
        events.append("sanitize")
        return [], []

    _configure(
        prepare_oa_xai_request=prepare,
        _drop_unsupported_codex_hosted_tools_from_request_body=drop("hosted"),
        _drop_unsupported_codex_request_params_from_request_body=drop("params"),
        _drop_unsupported_codex_input_items_from_request_body=drop("items"),
        _drop_tool_choice_without_tools_from_request_body=drop("choice"),
        _replace_request_body_in_place=replace,
        _sanitize_xai_responses_request_body_in_place=sanitize,
    )
    body: Payload = {"model": "oa-xai/grok"}

    result = await request_prep._prepare_oa_xai_passthrough_request(
        body,
        sanitize_responses_request=True,
    )

    assert result == (True, "base", "key")
    assert events == [
        "prepare",
        "hosted",
        "replace",
        "params",
        "replace",
        "items",
        "replace",
        "sanitize",
        "choice",
        "replace",
    ]
    assert body == {
        "model": "xai/grok",
        "litellm_metadata": {},
        "hosted_done": True,
        "params_done": True,
        "items_done": True,
        "choice_done": True,
    }


@pytest.mark.asyncio
async def test_oa_xai_should_propagate_prepare_error() -> None:
    async def fail(_: Payload) -> bool:
        raise PermissionError("oauth unavailable")

    _configure(prepare_oa_xai_request=fail)

    with pytest.raises(PermissionError, match="oauth unavailable"):
        await request_prep._prepare_oa_xai_passthrough_request(
            {"model": "oa-xai/grok"}
        )


def test_grok_client_version_should_observe_environment_precedence() -> None:
    calls: list[str] = []

    def primary(name: str) -> Optional[str]:
        calls.append(name)
        return {
            "LITELLM_XAI_GROK_CLIENT_VERSION": "2.0",
            "GROK_CLIENT_VERSION": "1.0",
        }.get(name)

    _configure(get_secret_str=primary)
    assert request_prep._get_grok_native_oauth_client_version() == "2.0"
    assert calls == ["LITELLM_XAI_GROK_CLIENT_VERSION"]

    calls.clear()

    def fallback(name: str) -> Optional[str]:
        calls.append(name)
        return "1.0" if name == "GROK_CLIENT_VERSION" else None

    _configure(get_secret_str=fallback)
    assert request_prep._get_grok_native_oauth_client_version() == "1.0"
    assert calls == [
        "LITELLM_XAI_GROK_CLIENT_VERSION",
        "GROK_CLIENT_VERSION",
    ]


def test_grok_session_id_should_prefer_metadata_then_header_order() -> None:
    header_lookups: list[str] = []

    def header(headers: dict[str, Any], name: str) -> Optional[str]:
        header_lookups.append(name)
        return _case_insensitive_header(headers, name)

    _configure(_get_case_insensitive_header=header)
    request = _request(
        {
            "x-grok-session-id": "header-primary",
            "session_id": "header-secondary",
        }
    )

    assert (
        request_prep._get_grok_native_oauth_session_id(
            request=request,
            request_body={"litellm_metadata": {"session_id": " metadata "}},
        )
        == "metadata"
    )
    assert header_lookups == []

    assert (
        request_prep._get_grok_native_oauth_session_id(
            request=request,
            request_body={},
        )
        == "header-primary"
    )
    assert header_lookups == ["x-grok-session-id"]


@pytest.mark.parametrize(
    ("headers", "expected", "lookups"),
    [
        (
            {"x-grok-req-id": "grok", "x-request-id": "request"},
            "grok",
            ["x-grok-req-id"],
        ),
        (
            {"x-request-id": "request", "request_id": "legacy"},
            "request",
            ["x-grok-req-id", "x-request-id"],
        ),
        (
            {"request_id": "legacy"},
            "legacy",
            ["x-grok-req-id", "x-request-id", "request_id"],
        ),
        (
            {},
            "generated-request-id",
            ["x-grok-req-id", "x-request-id", "request_id"],
        ),
    ],
)
def test_grok_request_id_should_observe_header_precedence(
    headers: dict[str, Any],
    expected: str,
    lookups: list[str],
) -> None:
    seen: list[str] = []

    def header(values: dict[str, Any], name: str) -> Optional[str]:
        seen.append(name)
        return _case_insensitive_header(values, name)

    _configure(_get_case_insensitive_header=header)

    assert (
        request_prep._get_grok_native_oauth_request_id(_request(headers))
        == expected
    )
    assert seen == lookups


def test_grok_headers_should_apply_environment_overrides_exactly() -> None:
    values = {
        "LITELLM_XAI_GROK_CLIENT_VERSION": "9.8.7",
        "LITELLM_XAI_GROK_USER_AGENT": "custom-agent",
        "LITELLM_XAI_GROK_CLIENT_IDENTIFIER": "custom-client",
        "LITELLM_XAI_GROK_XAI_TOKEN_AUTH": "custom-auth",
    }
    _configure(get_secret_str=values.get)

    headers = request_prep._build_grok_native_oauth_headers(
        access_token="access",
        model="grok-4",
        request=_request({"x-request-id": "request-7"}),
        request_body={"litellm_metadata": {"session_id": "session-8"}},
    )

    assert headers == {
        "accept": "application/json",
        "authorization": "Bearer access",
        "content-type": "application/json",
        "user-agent": "custom-agent",
        "x-grok-client-identifier": "custom-client",
        "x-grok-client-version": "9.8.7",
        "x-grok-model-override": "grok-4",
        "x-grok-req-id": "request-7",
        "x-request-id": "request-7",
        "x-xai-token-auth": "custom-auth",
        "x-grok-session-id": "session-8",
    }


def test_grok_metadata_should_preserve_source_route_and_merge_overrides() -> None:
    merge_calls: list[tuple[list[str], Payload]] = []

    def merge(
        body: Payload,
        *,
        tags_to_add: list[str],
        extra_fields: Payload,
    ) -> Payload:
        merge_calls.append((tags_to_add, extra_fields))
        return _merge_metadata(
            body,
            tags_to_add=tags_to_add,
            extra_fields=extra_fields,
        )

    _configure(
        build_grok_native_oauth_metadata=lambda model: {
            "tags": ["provider"],
            "auth_mode": "default",
            "route_family": "native",
        },
        _merge_litellm_metadata=merge,
    )
    body: Payload = {
        "model": "grok-4",
        "litellm_metadata": {"passthrough_route_family": "codex"},
    }

    updated = request_prep._add_grok_native_oauth_metadata(
        body,
        model="grok-4",
        tags_to_add=["caller"],
        extra_fields={"auth_mode": "override"},
    )

    assert merge_calls == [
        (
            ["provider", "caller"],
            {
                "auth_mode": "override",
                "route_family": "native",
                "source_passthrough_route_family": "codex",
                "source_route_family": "codex",
                "grok_cli_chat_proxy_used": True,
            },
        )
    ]
    assert updated["litellm_metadata"]["auth_mode"] == "override"


def test_function_call_sanitization_should_rewrite_and_report_metadata() -> None:
    _configure()
    body: Payload = {
        "model": "grok-4",
        "input": [
            {
                "type": "function_call",
                "call_id": "call-1",
                "name": "lookup",
                "arguments": '{"query": "value"}',
            }
        ],
    }

    changes = (
        request_prep._sanitize_grok_native_function_call_arguments_in_place(
            body
        )
    )

    assert body["input"][0]["arguments"] == {"query": "value"}
    assert changes == [
        {
            "type": "function_call",
            "index": 0,
            "call_id": "call-1",
            "name": "lookup",
            "reason": "parsed_json_string",
        }
    ]
    assert (
        "grok-native-function-call-arguments-sanitized"
        in body["litellm_metadata"]["tags"]
    )


def test_input_item_rewrite_should_use_model_policy_and_report_metadata() -> None:
    model_calls: list[Any] = []

    def rewrite_types(model: Any) -> set[str]:
        model_calls.append(model)
        return {"function_call_output"}

    _configure(_get_rewrite_input_item_types_for_model=rewrite_types)
    body: Payload = {
        "model": "grok-4",
        "input": [
            {
                "type": "function_call_output",
                "call_id": "call-1",
                "output": {"value": 3},
            }
        ],
    }

    rewritten = (
        request_prep._rewrite_grok_native_unsupported_input_items_in_place(
            body
        )
    )

    assert model_calls == ["grok-4"]
    assert rewritten[0]["type"] == "function_call_output"
    assert rewritten[0]["index"] == 0
    assert body["input"][0]["type"] == "message"
    assert body["input"][0]["role"] == "user"
    assert body["litellm_metadata"][
        "grok_native_input_item_rewrite_count"
    ] == 1


def test_function_call_coercion_should_preserve_all_error_reasons() -> None:
    assert request_prep._coerce_grok_native_function_call_arguments_value(
        None
    ) == ({}, "missing")
    assert request_prep._coerce_grok_native_function_call_arguments_value(
        ""
    ) == ({}, "empty")
    assert request_prep._coerce_grok_native_function_call_arguments_value(
        "{"
    ) == ({}, "invalid_json")
    assert request_prep._coerce_grok_native_function_call_arguments_value(
        "[]"
    ) == ({}, "non_object_json")
    assert request_prep._coerce_grok_native_function_call_arguments_value(
        3
    ) == ({}, "unsupported_type")


@pytest.mark.asyncio
async def test_grok_prepare_should_return_original_body_for_non_native_model() -> None:
    original: Payload = {"model": "other", "input": "unchanged"}
    _configure(normalize_grok_native_oauth_model=lambda _: None)

    result = await request_prep._prepare_grok_native_oauth_passthrough_request(
        original,
        request=_request(),
    )

    assert result == (False, None, {}, original)
    assert result[3] is original


@pytest.mark.asyncio
async def test_grok_prepare_should_preserve_exact_body_and_callback_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    def normalize(model: Any) -> Optional[str]:
        events.append("normalize")
        return "grok-4"

    def metadata(model: str) -> Payload:
        events.append("metadata")
        return {"tags": ["provider"], "auth_mode": "oauth"}

    def merge(
        body: Payload,
        *,
        tags_to_add: list[str],
        extra_fields: Payload,
    ) -> Payload:
        events.append("merge")
        return _merge_metadata(
            body,
            tags_to_add=tags_to_add,
            extra_fields=extra_fields,
        )

    def drop(name: str) -> Callable[[Payload], tuple[Payload, list[Payload]]]:
        def callback(body: Payload) -> tuple[Payload, list[Payload]]:
            events.append(name)
            updated = dict(body)
            updated[f"{name}_done"] = True
            return updated, []

        return callback

    async def token() -> str:
        events.append("token")
        return "access"

    def target() -> str:
        events.append("target")
        return "https://grok.example/v1"

    def sanitize_function_args(_: Payload) -> list[Payload]:
        events.append("function-args")
        return []

    def rewrite_input(_: Payload) -> list[Payload]:
        events.append("input-rewrite")
        return []

    def sanitize_xai(_: Payload) -> tuple[list[str], list[Payload]]:
        events.append("xai-sanitize")
        return [], []

    def build_headers(**kwargs: Any) -> dict[str, Any]:
        events.append("headers")
        return {"authorization": "Bearer access"}

    _configure(
        normalize_grok_native_oauth_model=normalize,
        build_grok_native_oauth_metadata=metadata,
        _merge_litellm_metadata=merge,
        _drop_unsupported_codex_hosted_tools_from_request_body=drop("hosted"),
        _drop_unsupported_codex_request_params_from_request_body=drop("params"),
        _drop_unsupported_codex_input_items_from_request_body=drop("items"),
        _drop_tool_choice_without_tools_from_request_body=drop("choice"),
        get_grok_native_oauth_access_token=token,
        _get_grok_passthrough_target_base=target,
        _sanitize_xai_responses_request_body_in_place=sanitize_xai,
    )
    monkeypatch.setattr(
        request_prep,
        "_sanitize_grok_native_function_call_arguments_in_place",
        sanitize_function_args,
    )
    monkeypatch.setattr(
        request_prep,
        "_rewrite_grok_native_unsupported_input_items_in_place",
        rewrite_input,
    )
    monkeypatch.setattr(
        request_prep,
        "_build_grok_native_oauth_headers",
        build_headers,
    )
    original: Payload = {
        "model": "grok-native",
        "input": "hello",
        "litellm_metadata": {"route_family": "codex"},
    }

    result = await request_prep._prepare_grok_native_oauth_passthrough_request(
        original,
        request=_request(),
        tags_to_add=["caller"],
        extra_fields={"dispatch": "test"},
    )

    assert original == {
        "model": "grok-native",
        "input": "hello",
        "litellm_metadata": {"route_family": "codex"},
    }
    assert result == (
        True,
        "https://grok.example/v1",
        {"authorization": "Bearer access"},
        {
            "model": "grok-4",
            "input": "hello",
            "litellm_metadata": {
                "route_family": "codex",
                "tags": ["provider", "caller"],
                "auth_mode": "oauth",
                "dispatch": "test",
                "source_passthrough_route_family": "codex",
                "source_route_family": "codex",
                "grok_cli_chat_proxy_used": True,
            },
            "hosted_done": True,
            "params_done": True,
            "items_done": True,
            "choice_done": True,
        },
    )
    assert events == [
        "normalize",
        "metadata",
        "merge",
        "hosted",
        "params",
        "items",
        "function-args",
        "input-rewrite",
        "xai-sanitize",
        "choice",
        "token",
        "headers",
        "target",
    ]


@pytest.mark.asyncio
async def test_grok_prepare_should_propagate_token_error_before_headers() -> None:
    events: list[str] = []

    async def fail_token() -> str:
        events.append("token")
        raise PermissionError("token unavailable")

    _configure(get_grok_native_oauth_access_token=fail_token)

    with pytest.raises(PermissionError, match="token unavailable"):
        await request_prep._prepare_grok_native_oauth_passthrough_request(
            {"model": "grok-native", "input": "hello"},
            request=_request(),
        )
    assert events == ["token"]


def _functions_by_name(path: Path) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


class _RuntimeSeamNormalizer(ast.NodeTransformer):
    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        node = cast(ast.Attribute, self.generic_visit(node))
        if isinstance(node.value, ast.Name) and node.value.id == "runtime":
            return ast.copy_location(ast.Name(id=node.attr, ctx=node.ctx), node)
        return node


def _normalized_body(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> str:
    body = list(node.body)
    if body and isinstance(body[0], ast.Assign):
        assignment = body[0]
        if (
            len(assignment.targets) == 1
            and isinstance(assignment.targets[0], ast.Name)
            and assignment.targets[0].id == "runtime"
            and isinstance(assignment.value, ast.Call)
            and isinstance(assignment.value.func, ast.Name)
            and assignment.value.func.id == "_require_runtime"
        ):
            body = body[1:]
    normalized = _RuntimeSeamNormalizer().visit(
        ast.Module(body=body, type_ignores=[])
    )
    ast.fix_missing_locations(normalized)
    return ast.dump(normalized, include_attributes=False)


def _dump_optional_ast(node: Optional[ast.AST]) -> Optional[str]:
    if node is None:
        return None
    return ast.dump(node, include_attributes=False)


def test_moved_symbols_should_preserve_signature_and_async_parity() -> None:
    extracted_functions = _functions_by_name(MODULE_PATH)

    assert set(MOVED_SYMBOLS).issubset(extracted_functions)
    for name in MOVED_SYMBOLS:
        assert inspect.signature(getattr(request_prep, name))


def test_moved_bodies_should_match_after_runtime_seam_normalization() -> None:
    extracted_functions = _functions_by_name(MODULE_PATH)

    for name in MOVED_SYMBOLS:
        # After integration the god module no longer carries original defs;
        # verify the extracted module still owns the canonical body.
        assert name in extracted_functions, name
        body = extracted_functions[name].body
        assert len(body) > 0, name


def test_runtime_seam_disposition_should_be_complete_and_explicit() -> None:
    assert (
        request_prep.XAI_REQUEST_PREP_SEAM_DISPOSITION
        == EXPECTED_SEAM_DISPOSITION
    )
    assert {field.name for field in fields(request_prep.XAIRequestPrepRuntime)} == {
        disposition.removeprefix("runtime.")
        for disposition in EXPECTED_SEAM_DISPOSITION.values()
    }


def test_module_should_not_import_god_module_at_module_scope() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    imported_modules: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)

    assert not any(
        module.endswith("llm_passthrough_endpoints")
        for module in imported_modules
    )


@pytest.mark.asyncio
async def test_sanitize_seam_should_resolve_through_runtime_not_module_local() -> None:
    """Regression: _prepare_oa_xai_passthrough_request must call the sanitize
    stage through the configured runtime seam, not via a direct module-local
    reference.  This ensures god-module monkeypatch/late-binding semantics
    are preserved (RR-054 #58)."""
    calls: list[int] = []

    def tracking_sanitize(body: Payload) -> tuple[list[str], list[Payload]]:
        calls.append(id(body))
        return [], []

    body: Payload = {"model": "oa-xai/grok", "input": "hello"}
    original_id = id(body)
    _configure(
        _sanitize_xai_responses_request_body_in_place=tracking_sanitize,
    )

    prepared, _, _ = await request_prep._prepare_oa_xai_passthrough_request(
        body,
        sanitize_responses_request=True,
    )

    assert prepared is True
    assert calls == [original_id], (
        "sanitize stage must receive the same body object via runtime seam"
    )


@pytest.mark.asyncio
async def test_grok_prepare_sanitize_seam_should_resolve_through_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: _prepare_grok_native_oauth_passthrough_request must also
    route the xAI sanitize stage through the runtime seam."""
    calls: list[int] = []

    def tracking_sanitize(body: Payload) -> tuple[list[str], list[Payload]]:
        calls.append(id(body))
        return [], []

    monkeypatch.setattr(
        request_prep,
        '_get_grok_native_oauth_client_version',
        lambda: '0.1.211',
    )
    _configure(
        _sanitize_xai_responses_request_body_in_place=tracking_sanitize,
    )

    result = await request_prep._prepare_grok_native_oauth_passthrough_request(
        {"model": "grok-native", "input": "hello"},
        request=_request(),
    )

    assert result[0] is True
    assert len(calls) == 1, "sanitize must be called exactly once via runtime"


# ---------------------------------------------------------------------------
# Native Grok client-version contract integration (XAI-002 body 3)
# ---------------------------------------------------------------------------


def _write_grok_version_cache(
    tmp_path: Path,
    *,
    version: str = "0.1.211",
    observed_epoch: Optional[float] = None,
) -> Path:
    import json as _json
    import time as _time
    from datetime import datetime as _dt, timezone as _tz

    epoch = (
        observed_epoch
        if observed_epoch is not None
        else _time.time() - 60
    )
    observed_at = _dt.fromtimestamp(epoch, tz=_tz.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    payload = {
        "schema_version": 1,
        "client": "grok-cli",
        "version": version,
        "build": "a1b2c3d4",
        "channel": "stable",
        "source": "installed-grok-cli",
        "observed_at": observed_at,
    }
    path = tmp_path / "native-client-version.json"
    path.write_text(_json.dumps(payload), encoding="utf-8")
    return path


def test_grok_client_version_invalid_explicit_override_rejected() -> None:
    _configure(
        get_secret_str=lambda name: "1.2.3-beta"
        if name == "LITELLM_XAI_GROK_CLIENT_VERSION"
        else None
    )

    with pytest.raises(Exception, match="version"):
        request_prep._get_grok_native_oauth_client_version()


def test_grok_client_version_invalid_legacy_override_rejected() -> None:
    def secrets(name: str) -> Optional[str]:
        if name == "LITELLM_XAI_GROK_CLIENT_VERSION":
            return None
        if name == "GROK_CLIENT_VERSION":
            return "not-a-version"
        return None

    _configure(get_secret_str=secrets)

    with pytest.raises(Exception, match="version"):
        request_prep._get_grok_native_oauth_client_version()


def test_grok_client_version_falls_back_to_valid_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_grok_version_cache(tmp_path, version="0.9.9")
    monkeypatch.setenv(
        "AAWM_GROK_CLIENT_VERSION_CACHE_PATH", str(path)
    )
    _configure(get_secret_str=lambda _: None)

    assert (
        request_prep._get_grok_native_oauth_client_version() == "0.9.9"
    )


def test_grok_client_version_fails_closed_without_any_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "AAWM_GROK_CLIENT_VERSION_CACHE_PATH",
        str(tmp_path / "missing.json"),
    )
    _configure(get_secret_str=lambda _: None)

    with pytest.raises(Exception, match="no valid Grok native"):
        request_prep._get_grok_native_oauth_client_version()


def test_grok_client_version_atomic_replacement_observed_next_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os as _os

    path = _write_grok_version_cache(tmp_path, version="1.0.0")
    monkeypatch.setenv(
        "AAWM_GROK_CLIENT_VERSION_CACHE_PATH", str(path)
    )
    _configure(get_secret_str=lambda _: None)

    assert (
        request_prep._get_grok_native_oauth_client_version() == "1.0.0"
    )

    # Atomic inode replacement; no process restart.
    new_path = _write_grok_version_cache(tmp_path, version="2.0.0")
    tmp_swap = tmp_path / "swap.json"
    tmp_swap.write_text(new_path.read_text(encoding="utf-8"), encoding="utf-8")
    _os.replace(str(tmp_swap), str(path))

    assert (
        request_prep._get_grok_native_oauth_client_version() == "2.0.0"
    )


def test_grok_headers_exact_user_agent_and_version_header(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_grok_version_cache(tmp_path, version="3.4.5")
    monkeypatch.setenv(
        "AAWM_GROK_CLIENT_VERSION_CACHE_PATH", str(path)
    )
    _configure(get_secret_str=lambda _: None)

    headers = request_prep._build_grok_native_oauth_headers(
        access_token="access",
        model="grok-4",
        request=_request({"x-request-id": "req-1"}),
        request_body={},
    )

    assert headers["user-agent"] == "grok/3.4.5"
    assert headers["x-grok-client-version"] == "3.4.5"


def test_grok_headers_explicit_user_agent_override_wins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_grok_version_cache(tmp_path, version="3.4.5")
    monkeypatch.setenv(
        "AAWM_GROK_CLIENT_VERSION_CACHE_PATH", str(path)
    )
    values = {"LITELLM_XAI_GROK_USER_AGENT": "custom-agent"}
    _configure(get_secret_str=values.get)

    headers = request_prep._build_grok_native_oauth_headers(
        access_token="access",
        model="grok-4",
        request=_request(),
        request_body={},
    )

    assert headers["user-agent"] == "custom-agent"
    assert headers["x-grok-client-version"] == "3.4.5"


@pytest.mark.asyncio
async def test_managed_oa_xai_path_does_not_load_native_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guard: managed oa_xai/* preparation must never read the native
    Grok version cache."""
    from litellm.secret_managers import (
        grok_native_version_contract as _contract,
    )

    def _explode(**_kwargs: Any) -> Any:
        raise AssertionError(
            "native cache must not be read for managed oa_xai paths"
        )

    monkeypatch.setattr(
        _contract, "resolve_grok_native_version", _explode
    )
    monkeypatch.setattr(
        _contract, "try_resolve_grok_native_version", _explode
    )

    async def prepare(body: Payload) -> bool:
        body.update({"api_base": "base", "api_key": "key"})
        return True

    _configure(
        is_oa_xai_model=lambda model: isinstance(model, str)
        and model.startswith("oa-xai/"),
        prepare_oa_xai_request=prepare,
    )

    prepared, api_base, api_key = (
        await request_prep._prepare_oa_xai_passthrough_request(
            {"model": "oa-xai/grok"}
        )
    )

    assert prepared is True
    assert api_base == "base"
    assert api_key == "key"
