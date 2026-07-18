"""RR-054 #12 behavioral retry/cooldown parity tests.

Covers:
- OpenRouter completion vs pass-through shared retry-loop behavior
- Codex/Anthropic auto-agent cooldown scope semantics via shared applicator
- Google distinct multi-budget policy (capacity vs generic rate-limit vs transient)
- Shared non-negative env parsers used by adapter retry knobs

No production code is modified by this suite; failures surface live divergences.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request
from starlette.responses import Response

from litellm.proxy._types import ProxyException
from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import retry as ar_retry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_request(session_id: str = "rr054-12-session") -> MagicMock:
    request = MagicMock(spec=Request)
    request.method = "POST"
    request.headers = {"session_id": session_id}
    request.query_params = {}
    request.state = MagicMock()
    # Ensure request-local maps start empty and are real dict/set containers.
    request.state.aawm_alias_request_local_cooldown_until = {}
    request.state.aawm_alias_request_local_excluded_keys = set()
    return request


class _FakeDurableCache:
    def __init__(self) -> None:
        self.redis_cache = self
        self.store: dict[str, Any] = {}
        self.set_calls: list[dict[str, Any]] = []

    async def async_get_cache(self, key: str, **_: Any) -> Any:
        return self.store.get(key)

    async def async_set_cache(self, key: str, value: Any, **kwargs: Any) -> None:
        self.set_calls.append({"key": key, "value": value, "kwargs": kwargs})
        self.store[key] = value


def _openrouter_429(
    *,
    model: str = "google/gemma-4-31b-it:free",
    raw: str | None = None,
    retry_after_seconds: float | None = None,
    reset_wait_header: str | None = None,
) -> ProxyException:
    raw_message = raw or f"{model} is temporarily rate-limited upstream."
    metadata: dict[str, Any] = {
        "raw": raw_message,
        "provider_name": "Stealth",
        "is_byok": False,
    }
    if retry_after_seconds is not None:
        metadata["retry_after_seconds"] = retry_after_seconds
    if reset_wait_header is not None:
        metadata["headers"] = {
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": reset_wait_header,
        }
    exc = ProxyException(
        message="Provider returned error",
        type="None",
        param="None",
        code=429,
    )
    exc.detail = {
        "error": {
            "message": "Provider returned error",
            "code": 429,
            "metadata": metadata,
        },
        "user_id": "user_test",
    }
    return exc


def _google_capacity_429() -> ProxyException:
    exc = ProxyException(
        message="No capacity available for model gemini-3.1-pro-preview",
        type="None",
        param="None",
        code=429,
    )
    exc.detail = (
        '429: b\'{\n  "error": {\n    "code": 429,\n'
        '    "message": "No capacity available for model gemini-3.1-pro-preview",\n'
        '    "status": "RESOURCE_EXHAUSTED",\n'
        '    "details": [\n      {\n'
        '        "@type": "type.googleapis.com/google.rpc.ErrorInfo",\n'
        '        "reason": "MODEL_CAPACITY_EXHAUSTED",\n'
        '        "domain": "cloudcode-pa.googleapis.com",\n'
        '        "metadata": {"model": "gemini-3.1-pro-preview"}\n'
        "      }\n    ]\n  }\n}\n'"
    )
    return exc


def _google_generic_429(*, reset_after_s: int = 4) -> ProxyException:
    exc = ProxyException(
        message=f"You have exhausted your capacity on this model. Your quota will reset after {reset_after_s}s.",
        type="None",
        param="None",
        code=429,
    )
    exc.detail = (
        f'b\'{{"error":{{"message":"quota reset after {reset_after_s}s"}}}}\''
    )
    return exc


def _google_transient_503() -> ProxyException:
    exc = ProxyException(
        message="upstream unavailable",
        type="None",
        param="None",
        code=503,
    )
    exc.detail = {"error": {"message": "upstream unavailable", "code": 503}}
    return exc


# ---------------------------------------------------------------------------
# OpenRouter completion / pass-through parity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rr054_12_openrouter_completion_and_passthrough_share_retry_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both entry points must retry the same short 429 once under identical env."""
    monkeypatch.setenv("AAWM_OPENROUTER_ADAPTER_MAX_RETRIES", "1")
    monkeypatch.setenv("AAWM_OPENROUTER_ADAPTER_BACKOFF_SECONDS", "2")
    monkeypatch.setenv("AAWM_OPENROUTER_ADAPTER_HIDDEN_RETRY_BUDGET_SECONDS", "0")
    monkeypatch.setenv("AAWM_OPENROUTER_ADAPTER_POST_FAILURE_COOLDOWN_SECONDS", "0")
    model = "google/gemma-4-31b-it:free"
    first = _openrouter_429(model=model)
    second = _openrouter_429(model=model)
    successful = Response(content=b'{"ok":true}', media_type="application/json")

    completion_set = AsyncMock()
    passthrough_set = AsyncMock()
    wait_mock = AsyncMock()

    # Completion path: caller-supplied operation fails once then succeeds.
    operation = AsyncMock(side_effect=[first, "completion-ok"])
    with patch.object(
        lpe, "_wait_for_openrouter_adapter_cooldown_if_needed", new=wait_mock
    ), patch.object(
        lpe, "_set_openrouter_adapter_cooldown", new=completion_set
    ), patch.object(
        lpe, "_maybe_raise_openrouter_adapter_failure_circuit_open", new=AsyncMock()
    ):
        completion_result = await lpe._perform_openrouter_completion_adapter_operation(
            adapter_model=model,
            operation=operation,
        )

    assert completion_result == "completion-ok"
    assert operation.await_count == 2
    assert [c.args for c in completion_set.await_args_list] == [(model, 2.0)]

    # Pass-through path: pass_through_request fails once then succeeds.
    with patch.object(
        lpe, "pass_through_request", new=AsyncMock(side_effect=[second, successful])
    ) as mock_pt, patch.object(
        lpe, "_wait_for_openrouter_adapter_cooldown_if_needed", new=wait_mock
    ), patch.object(
        lpe, "_set_openrouter_adapter_cooldown", new=passthrough_set
    ), patch.object(
        lpe, "_maybe_raise_openrouter_adapter_failure_circuit_open", new=AsyncMock()
    ):
        pt_result = await lpe._perform_openrouter_adapter_pass_through_request(
            adapter_model=model,
            request=MagicMock(),
            target="https://openrouter.ai/api/v1/chat/completions",
        )

    assert pt_result is successful
    assert mock_pt.await_count == 2
    assert [c.args for c in passthrough_set.await_args_list] == [(model, 2.0)]

    # Parity: same attempt count and same cooldown key/wait as completion path.
    assert operation.await_count == mock_pt.await_count
    assert completion_set.await_args_list == passthrough_set.await_args_list


@pytest.mark.asyncio
async def test_rr054_12_openrouter_completion_and_passthrough_exhaust_identically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When retries are exhausted, both paths raise and open the failure circuit."""
    monkeypatch.setenv("AAWM_OPENROUTER_ADAPTER_MAX_RETRIES", "1")
    monkeypatch.setenv("AAWM_OPENROUTER_ADAPTER_BACKOFF_SECONDS", "3")
    monkeypatch.setenv("AAWM_OPENROUTER_ADAPTER_HIDDEN_RETRY_BUDGET_SECONDS", "0")
    monkeypatch.setenv("AAWM_OPENROUTER_ADAPTER_POST_FAILURE_COOLDOWN_SECONDS", "60")
    model = "qwen/qwen3-coder:free"
    err_a = _openrouter_429(model=model)
    err_b = _openrouter_429(model=model)
    err_c = _openrouter_429(model=model)
    err_d = _openrouter_429(model=model)

    completion_set = AsyncMock()
    completion_circuit = AsyncMock()
    with patch.object(
        lpe, "_wait_for_openrouter_adapter_cooldown_if_needed", new=AsyncMock()
    ), patch.object(
        lpe, "_set_openrouter_adapter_cooldown", new=completion_set
    ), patch.object(
        lpe, "_maybe_raise_openrouter_adapter_failure_circuit_open", new=AsyncMock()
    ), patch.object(
        lpe, "_openrouter_adapter_open_failure_circuit", new=completion_circuit
    ):
        with pytest.raises(ProxyException):
            await lpe._perform_openrouter_completion_adapter_operation(
                adapter_model=model,
                operation=AsyncMock(side_effect=[err_a, err_b]),
            )

    passthrough_set = AsyncMock()
    passthrough_circuit = AsyncMock()
    with patch.object(
        lpe, "pass_through_request", new=AsyncMock(side_effect=[err_c, err_d])
    ) as mock_pt, patch.object(
        lpe, "_wait_for_openrouter_adapter_cooldown_if_needed", new=AsyncMock()
    ), patch.object(
        lpe, "_set_openrouter_adapter_cooldown", new=passthrough_set
    ), patch.object(
        lpe, "_maybe_raise_openrouter_adapter_failure_circuit_open", new=AsyncMock()
    ), patch.object(
        lpe, "_openrouter_adapter_open_failure_circuit", new=passthrough_circuit
    ):
        with pytest.raises(ProxyException):
            await lpe._perform_openrouter_adapter_pass_through_request(
                adapter_model=model,
                request=MagicMock(),
            )

    # Shared loop: one backoff cooldown on the first 429; circuit open on terminal.
    assert [c.args for c in completion_set.await_args_list] == [(model, 3.0)]
    assert [c.args for c in passthrough_set.await_args_list] == [(model, 3.0)]
    assert completion_circuit.await_count == 1
    assert passthrough_circuit.await_count == 1
    assert mock_pt.await_count == 2


@pytest.mark.asyncio
async def test_rr054_12_openrouter_passthrough_forwards_shared_retryable_status_codes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass-through wrapper must always advertise the shared 429/5xx set."""
    monkeypatch.setenv("AAWM_OPENROUTER_ADAPTER_MAX_RETRIES", "0")
    successful = Response(content=b'{"ok":true}', media_type="application/json")
    with patch.object(
        lpe, "pass_through_request", new=AsyncMock(return_value=successful)
    ) as mock_pt, patch.object(
        lpe, "_wait_for_openrouter_adapter_cooldown_if_needed", new=AsyncMock()
    ), patch.object(
        lpe, "_maybe_raise_openrouter_adapter_failure_circuit_open", new=AsyncMock()
    ):
        result = await lpe._perform_openrouter_adapter_pass_through_request(
            adapter_model="openrouter/free",
            request=MagicMock(),
            target="https://openrouter.ai/api/v1/chat/completions",
        )

    assert result is successful
    kwargs = mock_pt.await_args.kwargs
    assert kwargs["caller_managed_hidden_retry"] is True
    assert kwargs["retryable_upstream_status_codes"] == [429, 500, 502, 503, 504]


# ---------------------------------------------------------------------------
# Codex / Anthropic cooldown semantics (shared applicator)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rr054_12_codex_and_anthropic_candidate_scope_use_family_setters() -> None:
    """Durable candidate-scope cooldowns must call the matching family setter."""
    request = _minimal_request()
    candidate = {
        "provider": "openrouter",
        "model": "openrouter/cohere/north-mini-code:free",
        "route_family": "codex_openrouter_completion_adapter",
        "last_resort": False,
    }
    cooldown_key = "openrouter:openrouter/cohere/north-mini-code:free:openrouter"
    codex_setter = AsyncMock()
    anthropic_setter = AsyncMock()

    with patch.object(
        lpe, "_set_codex_auto_agent_cooldown", new=codex_setter
    ), patch.object(
        lpe, "_set_anthropic_auto_agent_cooldown", new=anthropic_setter
    ):
        codex_scope = await lpe._apply_codex_auto_agent_alias_cooldown(
            request=request,
            candidate=candidate,
            lane_key="openrouter",
            selected_cooldown_key=cooldown_key,
            cooldown_seconds=30.0,
            error_class="provider_terminal_error",
        )
        anthropic_scope = await lpe._apply_anthropic_auto_agent_alias_cooldown(
            request=request,
            candidate=candidate,
            lane_key="openrouter",
            selected_cooldown_key=cooldown_key,
            cooldown_seconds=30.0,
            error_class="provider_terminal_error",
        )

    assert codex_scope == "candidate"
    assert anthropic_scope == "candidate"
    codex_setter.assert_awaited_once_with(cooldown_key, 30.0)
    anthropic_setter.assert_awaited_once_with(cooldown_key, 30.0)


@pytest.mark.asyncio
async def test_rr054_12_codex_and_anthropic_xai_candidate_unavailable_is_request_local() -> None:
    """xAI candidate_unavailable must stay request-local for both families."""
    request_codex = _minimal_request("codex-xai")
    request_anth = _minimal_request("anth-xai")
    candidate = {
        "provider": "xai",
        "model": "grok-composer-2.5-fast",
        "route_family": "codex_grok_native_responses_adapter",
        "last_resort": False,
    }
    # Anthropic-family twin of the same xAI native route shape.
    anth_candidate = {
        **candidate,
        "route_family": "anthropic_grok_native_responses_adapter",
    }
    cooldown_key = "xai:grok-composer-2.5-fast:xai_grok_native"
    codex_setter = AsyncMock()
    anthropic_setter = AsyncMock()

    with patch.object(
        lpe, "_set_codex_auto_agent_cooldown", new=codex_setter
    ), patch.object(
        lpe, "_set_anthropic_auto_agent_cooldown", new=anthropic_setter
    ):
        codex_scope = await lpe._apply_codex_auto_agent_alias_cooldown(
            request=request_codex,
            candidate=candidate,
            lane_key="xai_grok_native",
            selected_cooldown_key=cooldown_key,
            cooldown_seconds=10800.0,
            error_class="candidate_unavailable",
        )
        anthropic_scope = await lpe._apply_anthropic_auto_agent_alias_cooldown(
            request=request_anth,
            candidate=anth_candidate,
            lane_key="xai_grok_native",
            selected_cooldown_key=cooldown_key,
            cooldown_seconds=10800.0,
            error_class="candidate_unavailable",
        )

    assert codex_scope == "request_local"
    assert anthropic_scope == "request_local"
    codex_setter.assert_not_awaited()
    anthropic_setter.assert_not_awaited()
    # Request-local exclusion must be recorded so the candidate is skipped.
    assert lpe._get_codex_auto_agent_request_local_excluded_keys(request_codex)
    assert lpe._get_codex_auto_agent_request_local_excluded_keys(request_anth)


@pytest.mark.asyncio
async def test_rr054_12_codex_and_anthropic_native_grok_45_unavailable_scope_none() -> None:
    """Native Grok 4.5 candidate_unavailable must be cooldown_scope=none for both."""
    request_codex = _minimal_request("codex-g45")
    request_anth = _minimal_request("anth-g45")
    codex_candidate = {
        "provider": "xai",
        "model": "xai/grok-4.5",
        "route_family": "codex_grok_native_responses_adapter",
        "last_resort": False,
    }
    anth_candidate = {
        "provider": "xai",
        "model": "xai/grok-4.5",
        "route_family": "anthropic_grok_native_responses_adapter",
        "last_resort": False,
    }
    cooldown_key = "xai:xai/grok-4.5:xai_grok_native"
    codex_setter = AsyncMock()
    anthropic_setter = AsyncMock()

    with patch.object(
        lpe, "_set_codex_auto_agent_cooldown", new=codex_setter
    ), patch.object(
        lpe, "_set_anthropic_auto_agent_cooldown", new=anthropic_setter
    ):
        codex_scope = await lpe._apply_codex_auto_agent_alias_cooldown(
            request=request_codex,
            candidate=codex_candidate,
            lane_key="xai_grok_native",
            selected_cooldown_key=cooldown_key,
            cooldown_seconds=3600.0,
            error_class="candidate_unavailable",
        )
        anthropic_scope = await lpe._apply_anthropic_auto_agent_alias_cooldown(
            request=request_anth,
            candidate=anth_candidate,
            lane_key="xai_grok_native",
            selected_cooldown_key=cooldown_key,
            cooldown_seconds=3600.0,
            error_class="candidate_unavailable",
        )

    assert codex_scope == "none"
    assert anthropic_scope == "none"
    codex_setter.assert_not_awaited()
    anthropic_setter.assert_not_awaited()
    assert not lpe._get_codex_auto_agent_request_local_excluded_keys(request_codex)
    assert not lpe._get_codex_auto_agent_request_local_excluded_keys(request_anth)


@pytest.mark.asyncio
async def test_rr054_12_codex_and_anthropic_setters_write_distinct_alias_families() -> None:
    """Family setters must durable-write under their own alias_family labels."""
    dual = _FakeDurableCache()
    cooldown_key = "rr054-12-family-key"
    lpe._codex_auto_agent_cooldown_until_monotonic_by_key.pop(cooldown_key, None)
    lpe._anthropic_auto_agent_cooldown_until_monotonic_by_key.pop(cooldown_key, None)

    with patch.object(
        lpe, "_get_aawm_alias_routing_dual_cache", return_value=dual
    ), patch(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable.get_aawm_alias_routing_dual_cache",
        return_value=dual,
    ):
        await lpe._set_codex_auto_agent_cooldown(cooldown_key, 12.0)
        await lpe._set_anthropic_auto_agent_cooldown(cooldown_key, 12.0)

    families = []
    for call in dual.set_calls:
        key = str(call["key"])
        if "codex" in key and "cooldown" in key:
            families.append("codex")
        if "anthropic" in key and "cooldown" in key:
            families.append("anthropic")
    # If durable keys encode alias_family, both families must appear.
    # Fall back to payload inspection when key layout differs.
    if not families:
        # Inspect write path args via monkeypatched write helper if set_calls
        # used opaque keys — re-run with write spy for deterministic parity.
        write_spy = AsyncMock(return_value=True)
        with patch.object(lpe, "_write_aawm_alias_routing_durable_payload", new=write_spy):
            await lpe._set_codex_auto_agent_cooldown(cooldown_key + "-b", 9.0)
            await lpe._set_anthropic_auto_agent_cooldown(cooldown_key + "-b", 9.0)
        assert write_spy.await_count == 2
        families = [
            call.kwargs.get("alias_family") or call.args[0]
            for call in write_spy.await_args_list
        ]
        # Normalize positional vs keyword forms.
        normalized = []
        for item in families:
            if isinstance(item, str):
                normalized.append(item)
        families = normalized or [
            c.kwargs["alias_family"] for c in write_spy.await_args_list
        ]

    assert "codex" in families
    assert "anthropic" in families

    lpe._codex_auto_agent_cooldown_until_monotonic_by_key.pop(cooldown_key, None)
    lpe._anthropic_auto_agent_cooldown_until_monotonic_by_key.pop(cooldown_key, None)
    lpe._codex_auto_agent_cooldown_until_monotonic_by_key.pop(cooldown_key + "-b", None)
    lpe._anthropic_auto_agent_cooldown_until_monotonic_by_key.pop(
        cooldown_key + "-b", None
    )


# ---------------------------------------------------------------------------
# Google distinct multi-budget policy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rr054_12_google_capacity_uses_capacity_budget_not_generic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MODEL_CAPACITY_EXHAUSTED must honor capacity_max_retries, not max_retries."""
    # Generic budget would stop at 1 attempt (max_retries=0); capacity allows more.
    monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_MAX_RETRIES", "0")
    monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_MODEL_CAPACITY_MAX_RETRIES", "2")
    monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_MODEL_CAPACITY_BACKOFF_SECONDS", "5,15,30")
    monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_HIDDEN_RETRY_BUDGET_SECONDS", "0")

    capacity_err = _google_capacity_429()
    successful = Response(content=b'{"ok":true}', media_type="application/json")
    set_cooldown = AsyncMock()

    with patch.object(
        lpe,
        "pass_through_request",
        new=AsyncMock(side_effect=[capacity_err, capacity_err, successful]),
    ) as mock_pt, patch.object(
        lpe, "_wait_for_google_adapter_cooldown_if_needed", new=AsyncMock()
    ), patch.object(
        lpe, "_set_google_adapter_cooldown", new=set_cooldown
    ):
        result = await lpe._perform_google_adapter_pass_through_request(
            request=MagicMock(),
            google_adapter_rate_limit_key="lane-capacity",
        )

    assert result is successful
    # capacity_max_retries=2 => total_attempts=3, success on 3rd.
    assert mock_pt.await_count == 3
    # Capacity path uses schedule backoff (+1s buffer on set).
    assert [c.args[1] for c in set_cooldown.await_args_list] == [6.0, 16.0]


@pytest.mark.asyncio
async def test_rr054_12_google_generic_rate_limit_uses_max_retries_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-capacity 429s must stop at max_retries+1 and not use capacity budget."""
    monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_MAX_RETRIES", "1")
    monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_MODEL_CAPACITY_MAX_RETRIES", "5")
    monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_HIDDEN_RETRY_BUDGET_SECONDS", "0")

    err = _google_generic_429(reset_after_s=4)
    set_cooldown = AsyncMock()

    with patch.object(
        lpe, "pass_through_request", new=AsyncMock(side_effect=[err, err])
    ) as mock_pt, patch.object(
        lpe, "_wait_for_google_adapter_cooldown_if_needed", new=AsyncMock()
    ), patch.object(
        lpe, "_set_google_adapter_cooldown", new=set_cooldown
    ):
        with pytest.raises(ProxyException):
            await lpe._perform_google_adapter_pass_through_request(
                request=MagicMock(),
                google_adapter_rate_limit_key="lane-generic",
            )

    # max_retries=1 => 2 attempts only (capacity budget must not extend this).
    assert mock_pt.await_count == 2
    assert set_cooldown.await_count == 1


@pytest.mark.asyncio
async def test_rr054_12_google_transient_uses_distinct_transient_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transient 503s must sleep via transient backoff and not set rate-limit cooldown."""
    monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_MAX_RETRIES", "0")
    monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_MODEL_CAPACITY_MAX_RETRIES", "0")
    monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_HIDDEN_RETRY_BUDGET_SECONDS", "0")

    transient = _google_transient_503()
    successful = Response(content=b'{"ok":true}', media_type="application/json")
    set_cooldown = AsyncMock()
    sleep_mock = AsyncMock()

    with patch.object(
        lpe, "pass_through_request", new=AsyncMock(side_effect=[transient, successful])
    ) as mock_pt, patch.object(
        lpe, "_wait_for_google_adapter_cooldown_if_needed", new=AsyncMock()
    ), patch.object(
        lpe, "_set_google_adapter_cooldown", new=set_cooldown
    ), patch.object(
        lpe.asyncio, "sleep", new=sleep_mock
    ):
        result = await lpe._perform_google_adapter_pass_through_request(
            request=MagicMock(),
            google_adapter_rate_limit_key="lane-transient",
        )

    assert result is successful
    assert mock_pt.await_count == 2
    set_cooldown.assert_not_awaited()
    sleep_mock.assert_awaited()
    # Transient schedule comes from shared passthrough hidden-retry wait helper.
    waited = sleep_mock.await_args.args[0]
    assert waited == lpe._get_google_adapter_transient_backoff_seconds(1)


@pytest.mark.asyncio
async def test_rr054_12_google_does_not_collapse_capacity_into_openrouter_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Google must keep multi-budget knobs; OpenRouter has a single attempt budget."""
    # Prove Google capacity backoff schedule is independent of OpenRouter schedule.
    monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_MODEL_CAPACITY_BACKOFF_SECONDS", "7,11")
    monkeypatch.setenv("AAWM_OPENROUTER_ADAPTER_BACKOFF_SECONDS", "2,10,20")
    assert lpe._get_google_adapter_capacity_backoff_seconds(1) == 7.0
    assert lpe._get_google_adapter_capacity_backoff_seconds(2) == 11.0
    assert lpe._get_openrouter_adapter_backoff_seconds(1) == 2.0
    assert lpe._get_openrouter_adapter_backoff_seconds(2) == 10.0
    # Google exposes three distinct budget helpers; OpenRouter does not.
    assert callable(lpe._get_google_adapter_max_retries)
    assert callable(lpe._get_google_adapter_model_capacity_max_retries)
    assert callable(lpe._get_google_adapter_transient_retry_max_attempts)
    assert not hasattr(lpe, "_get_openrouter_adapter_model_capacity_max_retries")
    assert not hasattr(lpe, "_get_openrouter_adapter_transient_retry_max_attempts")


# ---------------------------------------------------------------------------
# Shared retry env parsing
# ---------------------------------------------------------------------------


def test_rr054_12_shared_env_parsers_defaults_and_clamps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RR054_12_FLOAT", raising=False)
    monkeypatch.delenv("RR054_12_INT", raising=False)
    assert ar_retry.parse_non_negative_float_env("RR054_12_FLOAT", default=1.5) == 1.5
    assert ar_retry.parse_non_negative_int_env("RR054_12_INT", default=3) == 3

    monkeypatch.setenv("RR054_12_FLOAT", "not-a-number")
    monkeypatch.setenv("RR054_12_INT", "nope")
    assert ar_retry.parse_non_negative_float_env("RR054_12_FLOAT", default=2.5) == 2.5
    assert ar_retry.parse_non_negative_int_env("RR054_12_INT", default=4) == 4

    monkeypatch.setenv("RR054_12_FLOAT", "-3.0")
    monkeypatch.setenv("RR054_12_INT", "-2")
    assert ar_retry.parse_non_negative_float_env("RR054_12_FLOAT", default=9.0) == 0.0
    assert ar_retry.parse_non_negative_int_env("RR054_12_INT", default=9) == 0

    monkeypatch.setenv("RR054_12_FLOAT", "12.5")
    monkeypatch.setenv("RR054_12_INT", "8")
    assert (
        ar_retry.parse_non_negative_float_env(
            "RR054_12_FLOAT", default=0.0, maximum=10.0
        )
        == 10.0
    )
    assert (
        ar_retry.parse_non_negative_int_env("RR054_12_INT", default=0, maximum=5) == 5
    )
    assert ar_retry.parse_non_negative_float_env("RR054_12_FLOAT", default=0.0) == 12.5
    assert ar_retry.parse_non_negative_int_env("RR054_12_INT", default=0) == 8


def test_rr054_12_adapter_knobs_use_shared_env_parsers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AAWM_OPENROUTER_ADAPTER_MAX_RETRIES", "5")
    monkeypatch.setenv("AAWM_OPENROUTER_ADAPTER_HIDDEN_RETRY_BUDGET_SECONDS", "12.5")
    monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_MAX_RETRIES", "4")
    monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_HIDDEN_RETRY_BUDGET_SECONDS", "9.25")

    assert lpe._get_openrouter_adapter_max_retries() == 5
    assert lpe._get_openrouter_adapter_hidden_retry_budget_seconds() == 12.5
    assert lpe._get_google_adapter_max_retries() == 4
    assert lpe._get_google_adapter_hidden_retry_budget_seconds() == 9.25

    monkeypatch.setenv("AAWM_OPENROUTER_ADAPTER_MAX_RETRIES", "-1")
    monkeypatch.setenv("AAWM_GOOGLE_ADAPTER_HIDDEN_RETRY_BUDGET_SECONDS", "bad")
    assert lpe._get_openrouter_adapter_max_retries() == 0
    assert lpe._get_google_adapter_hidden_retry_budget_seconds() == 0.0


def test_rr054_12_projected_hidden_retry_within_budget_helper() -> None:
    projected, within = ar_retry.projected_hidden_retry_within_budget(
        accumulated_hidden_wait_seconds=2.0,
        next_wait_seconds=3.0,
        hidden_retry_budget_seconds=10.0,
    )
    assert projected == 5.0
    assert within is True

    projected, within = ar_retry.projected_hidden_retry_within_budget(
        accumulated_hidden_wait_seconds=8.0,
        next_wait_seconds=3.0,
        hidden_retry_budget_seconds=10.0,
    )
    assert projected == 11.0
    assert within is False

    # Budget disabled (0) never counts as within.
    projected, within = ar_retry.projected_hidden_retry_within_budget(
        accumulated_hidden_wait_seconds=0.0,
        next_wait_seconds=1.0,
        hidden_retry_budget_seconds=0.0,
    )
    assert projected == 1.0
    assert within is False


@pytest.mark.asyncio
async def test_rr054_12_openrouter_and_google_wait_helpers_use_shared_map_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both provider wait helpers must route through wait_for_monotonic_cooldown_map."""
    shared = AsyncMock(return_value=0.0)
    with patch.object(ar_retry, "wait_for_monotonic_cooldown_map", new=shared), patch.object(
        lpe._aawm_alias_retry, "wait_for_monotonic_cooldown_map", new=shared
    ):
        await lpe._wait_for_openrouter_adapter_cooldown_if_needed("or-key")
        await lpe._wait_for_google_adapter_cooldown_if_needed("g-key")

    assert shared.await_count == 2
    labels = []
    for call in shared.await_args_list:
        labels.append(call.kwargs.get("log_label") or call.args[2] if len(call.args) > 2 else call.kwargs.get("log_label"))
    # Accept kwargs form primarily.
    kwargs_labels = [c.kwargs.get("log_label") for c in shared.await_args_list]
    assert "OpenRouter adapter" in kwargs_labels
    assert "Google adapter" in kwargs_labels
