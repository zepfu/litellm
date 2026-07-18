"""Behavioral parity tests for OpenRouter-owned retry and transport logic."""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from typing import Optional

import pytest
from fastapi import HTTPException

from litellm.llms.anthropic.experimental_pass_through.providers.openrouter import (
    retry_transport,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
    MonotonicCooldownMap,
)


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


class CandidateUnavailable(Exception):
    pass


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


def _retry_after(headers: Mapping[str, object]) -> Optional[float]:
    value = _header_value(headers, "Retry-After")
    return float(value) if value is not None else None


def _reset_wait(headers: Mapping[str, object]) -> Optional[float]:
    value = _header_value(headers, "X-Test-Reset-Wait")
    return float(value) if value is not None else None


def _raise_candidate_unavailable(message: str) -> None:
    raise CandidateUnavailable(message)


async def _no_alias_cooldown(
    _adapter_model: Optional[str],
    *,
    use_alias_candidate_probe: bool,
) -> None:
    _ = use_alias_candidate_probe


def _runtime(
    *,
    env: Optional[dict[str, str]] = None,
    pass_through_request: Optional[
        Callable[..., Awaitable[object]]
    ] = None,
    sleeps: Optional[list[float]] = None,
) -> retry_transport.Runtime:
    environment = env or {}
    recorded_sleeps = sleeps if sleeps is not None else []
    runtime_holder: dict[str, retry_transport.Runtime] = {}

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
        await retry_transport.wait_for_cooldown_if_needed(
            runtime_holder["runtime"],
            rate_limit_keys,
            adapter_model=adapter_model,
            use_alias_candidate_probe=use_alias_candidate_probe,
        )

    async def set_cooldown_callback(
        rate_limit_keys: str,
        wait_seconds: float,
    ) -> None:
        await retry_transport.set_cooldown(
            runtime_holder["runtime"],
            rate_limit_keys,
            wait_seconds,
        )

    async def maybe_raise_failure_circuit_open_callback(
        adapter_model: Optional[str],
    ) -> None:
        await retry_transport.maybe_raise_failure_circuit_open(
            runtime_holder["runtime"],
            adapter_model,
        )

    async def open_failure_circuit_callback(
        adapter_model: Optional[str],
        *,
        exc: object,
    ) -> None:
        await retry_transport.open_failure_circuit(
            runtime_holder["runtime"],
            adapter_model,
            exc=exc,
        )

    def clear_failure_circuit_callback(adapter_model: Optional[str]) -> None:
        retry_transport.clear_failure_circuit(
            runtime_holder["runtime"],
            adapter_model,
        )

    runtime = retry_transport.Runtime(
        rate_limit=MonotonicCooldownMap(),
        failure_circuit_until_monotonic_by_key={},
        clean_secret_string=lambda value: (
            str(value).strip() if value is not None and str(value).strip() else None
        ),
        extract_embedded_json_payload_candidates=_embedded_json_candidates,
        parse_json_payloads_from_text_candidates=_parse_json_candidates,
        extract_upstream_headers=lambda exc: getattr(exc, "headers", {}),
        parse_retry_after_seconds_from_headers=_retry_after,
        get_header_value=_header_value,
        parse_reset_wait_seconds_from_headers=_reset_wait,
        raise_candidate_unavailable=_raise_candidate_unavailable,
        maybe_raise_alias_probe_cooldown=_no_alias_cooldown,
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
    runtime_holder["runtime"] = runtime
    return runtime


def test_rr054_openrouter_error_payload_and_metadata_extraction() -> None:
    runtime = _runtime()
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

    payload = retry_transport.extract_error_payload(runtime, exc)
    assert payload is not None
    assert retry_transport.extract_exception_status_code(runtime, exc) == 429
    assert retry_transport.extract_provider_name(runtime, exc) == "Example"
    assert retry_transport.extract_raw_message(runtime, exc) == "ERROR"
    assert retry_transport.extract_retry_after_seconds(runtime, exc) == 4.0
    assert retry_transport.is_provider_raw_error(runtime, exc) is True
    assert retry_transport.extract_error_headers(runtime, exc) == {
        "Retry-After": "9",
        "X-Upstream": "present",
        "X-Meta": "yes",
    }


def test_rr054_openrouter_no_endpoint_error_is_alias_probe_only() -> None:
    runtime = _runtime()
    exc = ProviderError(
        "OpenRouter request failed",
        status_code=404,
        detail={"error": {"message": "No endpoints found for this model"}},
    )

    assert retry_transport.is_no_endpoint_candidate_error(runtime, exc)
    retry_transport.maybe_raise_alias_probe_no_endpoint_unavailable(
        runtime,
        exc,
        adapter_model="openrouter/example",
        use_alias_candidate_probe=False,
    )
    with pytest.raises(CandidateUnavailable, match="no available endpoints"):
        retry_transport.maybe_raise_alias_probe_no_endpoint_unavailable(
            runtime,
            exc,
            adapter_model="openrouter/example",
            use_alias_candidate_probe=True,
        )


def test_rr054_openrouter_retry_wait_prefers_provider_reset_metadata() -> None:
    runtime = _runtime(env={"AAWM_OPENROUTER_ADAPTER_BACKOFF_SECONDS": "2,10"})
    metadata_retry = ProviderError(
        "rate limited",
        status_code=429,
        detail={
            "error": {
                "metadata": {
                    "retry_after_seconds": 4,
                }
            }
        },
    )
    header_reset = ProviderError(
        "rate limited",
        status_code=429,
        headers={
            "X-RateLimit-Remaining": "0",
            "X-Test-Reset-Wait": "7",
        },
    )

    assert retry_transport.get_retry_wait_seconds(runtime, metadata_retry, 1) == 5.0
    assert retry_transport.get_retry_wait_seconds(runtime, header_reset, 1) == 8.0
    assert retry_transport.get_retry_wait_seconds(runtime, header_reset, 2) == 10.0


def test_rr054_openrouter_long_window_rate_limit_classification() -> None:
    runtime = _runtime()
    metadata_long = ProviderError(
        "rate limited",
        status_code=429,
        detail={"error": {"metadata": {"retry_after_seconds": 31}}},
    )
    metadata_short = ProviderError(
        "rate limited",
        status_code=429,
        detail={"error": {"metadata": {"retry_after_seconds": 30}}},
    )
    header_long = ProviderError(
        "rate limited",
        status_code=429,
        headers={
            "X-RateLimit-Remaining": "0",
            "X-Test-Reset-Wait": "31",
        },
    )

    assert retry_transport.is_long_window_rate_limit(
        runtime,
        metadata_long,
        hidden_retry_budget_seconds=5,
    )
    assert not retry_transport.is_long_window_rate_limit(
        runtime,
        metadata_short,
        hidden_retry_budget_seconds=5,
    )
    assert retry_transport.is_long_window_rate_limit(
        runtime,
        header_long,
        hidden_retry_budget_seconds=5,
    )


@pytest.mark.asyncio
async def test_rr054_openrouter_cooldown_and_failure_circuit_state() -> None:
    runtime = _runtime(
        env={"AAWM_OPENROUTER_ADAPTER_POST_FAILURE_COOLDOWN_SECONDS": "5"}
    )
    model = "openrouter/example"

    await retry_transport.set_cooldown(runtime, model, 10)
    active = await retry_transport.get_active_cooldown_seconds(runtime, model)
    assert 0 < active <= 10

    exc = ProviderError(
        "rate limited",
        status_code=429,
        detail={"error": {"metadata": {"retry_after_seconds": 20}}},
    )
    await retry_transport.open_failure_circuit(runtime, model, exc=exc)
    circuit_until = runtime.failure_circuit_until_monotonic_by_key[model]
    assert 19 < circuit_until - time.monotonic() <= 20

    with pytest.raises(HTTPException) as raised:
        await retry_transport.maybe_raise_failure_circuit_open(runtime, model)
    assert raised.value.status_code == 429
    assert "temporarily cooling down" in str(raised.value.detail)

    retry_transport.clear_failure_circuit(runtime, model)
    assert model not in runtime.failure_circuit_until_monotonic_by_key


@pytest.mark.asyncio
async def test_rr054_openrouter_passthrough_retry_forwards_transport_policy() -> None:
    calls: list[dict[str, object]] = []
    sleeps: list[float] = []

    async def fake_pass_through_request(**kwargs: object) -> object:
        calls.append(kwargs)
        if len(calls) == 1:
            raise ProviderError("rate limited", status_code=429)
        return {"ok": True}

    runtime = _runtime(
        env={
            "AAWM_OPENROUTER_ADAPTER_MAX_RETRIES": "1",
            "AAWM_OPENROUTER_ADAPTER_BACKOFF_SECONDS": "1",
        },
        pass_through_request=fake_pass_through_request,
        sleeps=sleeps,
    )

    result = await retry_transport.perform_pass_through_request(
        runtime,
        adapter_model="openrouter/example",
        request=object(),
        target="https://openrouter.example/v1/responses",
        custom_headers={},
        user_api_key_dict=object(),
        custom_body={"model": "example"},
    )

    assert result == {"ok": True}
    assert len(calls) == 2
    for call in calls:
        assert call["custom_body"] == {"model": "example"}
        assert call["target"] == "https://openrouter.example/v1/responses"
        assert call["retryable_upstream_status_codes"] == [429, 500, 502, 503, 504]
        assert call["caller_managed_hidden_retry"] is True
    assert len(sleeps) == 1
    assert 0 < sleeps[0] <= 1
