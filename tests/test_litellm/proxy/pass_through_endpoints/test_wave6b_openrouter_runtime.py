"""Direct parity tests for the Wave 6B OpenRouter runtime extraction."""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable, Generator, Iterable, Mapping
from types import SimpleNamespace
from typing import Any, Optional

import pytest
from fastapi import HTTPException

from litellm.llms.anthropic.experimental_pass_through.providers.openrouter import (
    retry_transport,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
    MonotonicCooldownMap,
)
from litellm.proxy.pass_through_endpoints.providers.openrouter import (
    runtime as openrouter_runtime,
)


@pytest.fixture(autouse=True)
def _restore_openrouter_runtime() -> Generator[None, None, None]:
    saved = openrouter_runtime._runtime
    yield
    openrouter_runtime._runtime = saved


class ProviderError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        detail: object = None,
        headers: Optional[Mapping[str, object]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.detail = detail
        self.headers = dict(headers or {})


def _embedded_json_candidates(value: object) -> Iterable[object]:
    if not isinstance(value, str):
        return []
    start = value.find("{")
    end = value.rfind("}")
    if start < 0 or end < start:
        return []
    return [value[start : end + 1]]


def _parse_json_candidates(values: Iterable[object]) -> Iterable[object]:
    parsed: list[object] = []
    for value in values:
        if not isinstance(value, str):
            continue
        try:
            parsed.append(json.loads(value))
        except json.JSONDecodeError:
            continue
    return parsed


def _header_value(
    headers: Mapping[str, object],
    header_name: str,
) -> Optional[str]:
    lowered_name = header_name.lower()
    for name, value in headers.items():
        if name.lower() == lowered_name:
            return str(value)
    return None


def _merge_litellm_metadata(
    request_body: dict[str, Any],
    *,
    tags_to_add: Optional[list[str]] = None,
    extra_fields: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    updated_body = dict(request_body)
    metadata = dict(updated_body.get("litellm_metadata") or {})
    tags = list(metadata.get("tags") or [])
    for tag in tags_to_add or []:
        if tag not in tags:
            tags.append(tag)
    metadata["tags"] = tags
    metadata.update(extra_fields or {})
    updated_body["litellm_metadata"] = metadata
    return updated_body


def _chat_message_role(message: Any) -> Optional[str]:
    if isinstance(message, dict):
        role = message.get("role")
    else:
        role = getattr(message, "role", None)
    return role if isinstance(role, str) else None


def _chat_message_tool_call_ids(message: Any) -> list[str]:
    if isinstance(message, dict):
        tool_calls = message.get("tool_calls")
    else:
        tool_calls = getattr(message, "tool_calls", None)
    if not isinstance(tool_calls, list):
        return []
    result: list[str] = []
    for tool_call in tool_calls:
        if isinstance(tool_call, dict):
            tool_call_id = tool_call.get("id")
        else:
            tool_call_id = getattr(tool_call, "id", None)
        if isinstance(tool_call_id, str):
            result.append(tool_call_id)
    return result


def _chat_message_tool_result_id(message: Any) -> Optional[str]:
    if isinstance(message, dict):
        value = message.get("tool_call_id")
    else:
        value = getattr(message, "tool_call_id", None)
    return value if isinstance(value, str) and value else None


def _is_empty_text_content(content: Any) -> bool:
    if content is None:
        return True
    if isinstance(content, str):
        return not content.strip()
    if isinstance(content, list):
        return not content
    return False


def _configure_runtime(
    *,
    env: Optional[dict[str, str]] = None,
    secrets: Optional[dict[str, str]] = None,
    pass_through_request: Optional[
        Callable[..., Awaitable[object]]
    ] = None,
    sleeps: Optional[list[float]] = None,
) -> retry_transport.Runtime:
    environment = env or {}
    secret_values = secrets or {}
    recorded_sleeps = sleeps if sleeps is not None else []
    async def default_pass_through_request(**_kwargs: object) -> object:
        return object()

    async def fake_sleep(seconds: float) -> None:
        recorded_sleeps.append(seconds)

    async def wait_for_cooldown(
        rate_limit_keys: str,
        *,
        adapter_model: Optional[str] = None,
        use_alias_candidate_probe: bool = False,
    ) -> None:
        await openrouter_runtime._wait_for_openrouter_adapter_cooldown_if_needed(
            rate_limit_keys,
            adapter_model=adapter_model,
            use_alias_candidate_probe=use_alias_candidate_probe,
        )

    async def set_cooldown_callback(
        rate_limit_keys: str,
        wait_seconds: float,
    ) -> None:
        await openrouter_runtime._set_openrouter_adapter_cooldown(
            rate_limit_keys,
            wait_seconds,
        )

    async def maybe_raise_failure_circuit_open_callback(
        adapter_model: Optional[str],
    ) -> None:
        await openrouter_runtime._maybe_raise_openrouter_adapter_failure_circuit_open(
            adapter_model
        )

    async def open_failure_circuit_callback(
        adapter_model: Optional[str],
        *,
        exc: object,
    ) -> None:
        await openrouter_runtime._openrouter_adapter_open_failure_circuit(
            adapter_model,
            exc=exc,
        )

    def clear_failure_circuit_callback(
        adapter_model: Optional[str],
    ) -> None:
        openrouter_runtime._clear_openrouter_adapter_failure_circuit(
            adapter_model
        )

    async def no_alias_cooldown(
        _adapter_model: Optional[str],
        *,
        use_alias_candidate_probe: bool,
    ) -> None:
        _ = use_alias_candidate_probe

    def clean(value: Optional[str]) -> Optional[str]:
        if value is None or not str(value).strip():
            return None
        return str(value).strip()

    retry_runtime = retry_transport.Runtime(
        rate_limit=MonotonicCooldownMap(),
        failure_circuit_until_monotonic_by_key={},
        clean_secret_string=clean,
        extract_embedded_json_payload_candidates=_embedded_json_candidates,
        parse_json_payloads_from_text_candidates=_parse_json_candidates,
        extract_upstream_headers=lambda exc: getattr(exc, "headers", {}),
        parse_retry_after_seconds_from_headers=lambda headers: (
            float(value)
            if (value := _header_value(headers, "Retry-After")) is not None
            else None
        ),
        get_header_value=_header_value,
        parse_reset_wait_seconds_from_headers=lambda headers: (
            float(value)
            if (value := _header_value(headers, "X-Test-Reset-Wait"))
            is not None
            else None
        ),
        raise_candidate_unavailable=lambda message: (_ for _ in ()).throw(
            RuntimeError(message)
        ),
        maybe_raise_alias_probe_cooldown=no_alias_cooldown,
        get_completion_model=lambda model: (
            model.removeprefix("openrouter/") if model is not None else None
        ),
        pass_through_request=(
            pass_through_request or default_pass_through_request
        ),
        wait_for_cooldown=wait_for_cooldown,
        set_cooldown_callback=set_cooldown_callback,
        maybe_raise_failure_circuit_open_callback=(
            maybe_raise_failure_circuit_open_callback
        ),
        open_failure_circuit_callback=open_failure_circuit_callback,
        clear_failure_circuit_callback=clear_failure_circuit_callback,
        log_debug=lambda *_args, **_kwargs: None,
        log_warning=lambda *_args, **_kwargs: None,
        getenv=environment.get,
        sleep=fake_sleep,
        monotonic=time.monotonic,
    )
    openrouter_runtime.configure_openrouter_runtime(
        openrouter_runtime.Runtime(
            retry_transport_runtime=retry_runtime,
            clean_secret_string=clean,
            get_first_secret_value=lambda names: next(
                (
                    cleaned
                    for name in names
                    if (cleaned := clean(secret_values.get(name))) is not None
                ),
                None,
            ),
            getenv=environment.get,
            get_secret_str=secret_values.get,
            sanitize_opencode_zen_completion_messages=lambda kwargs: (
                kwargs,
                {},
            ),
            chat_message_role=_chat_message_role,
            chat_message_tool_call_ids=_chat_message_tool_call_ids,
            chat_message_tool_result_id=_chat_message_tool_result_id,
            is_empty_text_content=_is_empty_text_content,
            merge_litellm_metadata=_merge_litellm_metadata,
            build_langfuse_span_descriptor=lambda **kwargs: kwargs,
        )
    )
    return retry_runtime


@pytest.mark.asyncio
async def test_openrouter_runtime_success_and_target_configuration() -> None:
    _configure_runtime(
        env={
            "OPENROUTER_API_BASE": "https://router.example/api/v1/",
        },
        secrets={
            "AAWM_OPENROUTER_API_KEY": "or-key",
            "OR_SITE_URL": "https://client.example",
            "OR_APP_NAME": "AAWM",
        },
    )

    async def operation() -> dict[str, bool]:
        return {"ok": True}

    result = (
        await openrouter_runtime._perform_openrouter_completion_adapter_operation(
            adapter_model="openrouter/example/model",
            operation=operation,
        )
    )

    assert result == {"ok": True}
    assert openrouter_runtime._get_openrouter_api_key() == "or-key"
    assert (
        openrouter_runtime._get_anthropic_adapter_openrouter_api_key()
        == "or-key"
    )
    assert (
        openrouter_runtime._get_openrouter_target_base()
        == "https://router.example/api"
    )
    assert (
        openrouter_runtime._get_anthropic_adapter_openrouter_target_base()
        == "https://router.example/api"
    )
    assert openrouter_runtime._build_openrouter_default_headers() == {
        "HTTP-Referer": "https://client.example",
        "X-Title": "AAWM",
    }


@pytest.mark.asyncio
async def test_openrouter_runtime_retry_cooldown_and_failure_circuit() -> None:
    calls: list[dict[str, object]] = []
    sleeps: list[float] = []

    async def pass_through_request(**kwargs: object) -> object:
        calls.append(kwargs)
        if len(calls) == 1:
            raise ProviderError("rate limited", status_code=429)
        return {"ok": True}

    retry_runtime = _configure_runtime(
        env={
            "AAWM_OPENROUTER_ADAPTER_MAX_RETRIES": "1",
            "AAWM_OPENROUTER_ADAPTER_BACKOFF_SECONDS": "1",
            "AAWM_OPENROUTER_ADAPTER_POST_FAILURE_COOLDOWN_SECONDS": "5",
        },
        pass_through_request=pass_through_request,
        sleeps=sleeps,
    )

    result = (
        await openrouter_runtime._perform_openrouter_adapter_pass_through_request(
            adapter_model="openrouter/example",
            request=object(),
            target="https://router.example/v1/responses",
            custom_headers={},
            user_api_key_dict=object(),
            custom_body={"model": "example"},
        )
    )

    assert result == {"ok": True}
    assert len(calls) == 2
    assert len(sleeps) == 1
    assert 0 < sleeps[0] <= 1
    assert all(
        call["retryable_upstream_status_codes"] == [429, 500, 502, 503, 504]
        for call in calls
    )
    assert all(call["caller_managed_hidden_retry"] is True for call in calls)

    model = "openrouter/example"
    await openrouter_runtime._set_openrouter_adapter_cooldown(model, 10)
    active = (
        await openrouter_runtime._get_openrouter_adapter_active_cooldown_seconds(
            model
        )
    )
    assert 0 < active <= 10

    exc = ProviderError(
        "rate limited",
        status_code=429,
        detail={"error": {"metadata": {"retry_after_seconds": 20}}},
    )
    await openrouter_runtime._openrouter_adapter_open_failure_circuit(
        model,
        exc=exc,
    )
    assert model in retry_runtime.failure_circuit_until_monotonic_by_key
    with pytest.raises(HTTPException, match="temporarily cooling down"):
        await openrouter_runtime._maybe_raise_openrouter_adapter_failure_circuit_open(
            model
        )
    openrouter_runtime._clear_openrouter_adapter_failure_circuit(model)
    assert model not in retry_runtime.failure_circuit_until_monotonic_by_key


def test_openrouter_runtime_error_payload_and_header_extraction() -> None:
    _configure_runtime()
    exc = ProviderError(
        "wrapped OpenRouter error",
        status_code=429,
        detail=(
            'prefix {"error":{"message":"Provider returned an error",'
            '"metadata":{"provider_name":"Example","raw":"ERROR",'
            '"retry_after_seconds":4,"headers":{"X-Meta":"yes"}}}} suffix'
        ),
        headers={"Retry-After": "9", "X-Upstream": "present"},
    )

    payload = openrouter_runtime._extract_openrouter_adapter_error_payload(exc)
    assert payload is not None
    assert (
        openrouter_runtime._extract_openrouter_adapter_exception_status_code(
            exc
        )
        == 429
    )
    assert (
        openrouter_runtime._extract_openrouter_adapter_provider_name(exc)
        == "Example"
    )
    assert openrouter_runtime._extract_openrouter_adapter_raw_message(exc) == (
        "ERROR"
    )
    assert (
        openrouter_runtime._extract_openrouter_adapter_retry_after_seconds(exc)
        == 4.0
    )
    assert openrouter_runtime._is_openrouter_adapter_provider_raw_error(exc)
    assert openrouter_runtime._extract_openrouter_adapter_error_headers(exc) == {
        "Retry-After": "9",
        "X-Upstream": "present",
        "X-Meta": "yes",
    }


def test_openrouter_runtime_message_sanitization_and_metadata_application() -> None:
    _configure_runtime()
    object_message = SimpleNamespace(
        role="assistant",
        content=None,
        tool_calls=[
            SimpleNamespace(
                id="call_object",
                type="function",
                function=SimpleNamespace(
                    name="Read",
                    arguments={"path": "b.txt"},
                ),
            )
        ],
    )
    completion_kwargs = {
        "messages": [
            {"role": "user", "content": "start"},
            {"role": "assistant", "content": ""},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_dict",
                        "type": "function",
                        "function": {
                            "name": "Read",
                            "arguments": {"path": "a.txt"},
                        },
                    },
                    {
                        "id": "call_array",
                        "type": "function",
                        "function": {
                            "name": "Batch",
                            "arguments": ["a", "b"],
                        },
                    },
                    {
                        "id": "call_scalar",
                        "type": "function",
                        "function": {"name": "Count", "arguments": 7},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_dict", "content": "ok"},
            object_message,
            SimpleNamespace(
                role="tool",
                tool_call_id="call_object",
                content="ok",
            ),
            {"role": "user", "content": []},
        ]
    }

    request_body, updated_kwargs, metadata = (
        openrouter_runtime._apply_openrouter_completion_message_sanitization(
            request_body={"model": "example"},
            completion_kwargs=completion_kwargs,
            litellm_metadata={"tags": ["existing"]},
            span_name="openrouter.sanitize",
            tag="openrouter-sanitized",
        )
    )

    assert len(updated_kwargs["messages"]) == 5
    dict_tool_calls = updated_kwargs["messages"][1]["tool_calls"]
    assert json.loads(dict_tool_calls[0]["function"]["arguments"]) == {
        "path": "a.txt"
    }
    assert json.loads(dict_tool_calls[1]["function"]["arguments"]) == [
        "a",
        "b",
    ]
    assert json.loads(dict_tool_calls[2]["function"]["arguments"]) == 7
    updated_object = updated_kwargs["messages"][3]
    assert json.loads(
        updated_object.tool_calls[0].function.arguments
    ) == {"path": "b.txt"}
    assert object_message.tool_calls[0].function.arguments == {"path": "b.txt"}
    assert metadata["openrouter_chat_message_shape_removed_empty_message_count"] == 2
    assert metadata["openrouter_chat_tool_arguments_normalized_count"] == 4
    assert metadata["openrouter_chat_tool_arguments_object_count"] == 2
    assert metadata["openrouter_chat_tool_arguments_array_count"] == 1
    assert metadata["openrouter_chat_tool_arguments_scalar_count"] == 1
    assert metadata["tags"] == ["existing", "openrouter-sanitized"]
    assert request_body["litellm_metadata"] == metadata
    assert updated_kwargs["metadata"] == metadata
