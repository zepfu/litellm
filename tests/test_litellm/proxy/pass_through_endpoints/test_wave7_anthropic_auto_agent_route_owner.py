"""Wave 7 ownership tests: anthropic_auto_agent_route module.

Validates the extracted ``anthropic_auto_agent_route`` module structure,
runtime bundle completeness, seam disposition, and behavioral/fail-closed
coverage for both ``handle_auto_agent_alias_route`` (legacy facade) and
``handle_anthropic_auto_agent_alias_route`` (production wrapper).

Behavioral risk coverage (xAI-identified):
- Anthropic family-state selection and cooldown-capture fail-closed path
- Service perform-function binding under partial runtime
- Legacy nonlocal state round-trip through cooldown publication

Write scope: this file only.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime import (
    anthropic_auto_agent_route as mod,
)
from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.anthropic_auto_agent_route import (
    AnthropicAutoAgentRouteRuntime,
    ANTHROPIC_AUTO_AGENT_ROUTE_SEAM_DISPOSITION,
    handle_anthropic_auto_agent_alias_route,
    handle_auto_agent_alias_route,
)

MODULE_PATH = Path(mod.__file__).resolve()

# ---------------------------------------------------------------------------
# Structural: module exists and has no god-module import at module scope
# ---------------------------------------------------------------------------


class TestModuleStructure:
    """Module-level structural contracts."""

    def test_module_file_exists(self) -> None:
        assert MODULE_PATH.exists()

    def test_no_god_module_import_at_module_scope(self) -> None:
        """The module must not import llm_passthrough_endpoints at top level."""
        tree = ast.parse(MODULE_PATH.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "llm_passthrough_endpoints" not in alias.name
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert "llm_passthrough_endpoints" not in node.module

    def test_public_functions_exist(self) -> None:
        assert callable(handle_auto_agent_alias_route)
        assert callable(handle_anthropic_auto_agent_alias_route)

    def test_functions_are_async(self) -> None:
        assert inspect.iscoroutinefunction(handle_auto_agent_alias_route)
        assert inspect.iscoroutinefunction(handle_anthropic_auto_agent_alias_route)


# ---------------------------------------------------------------------------
# Runtime bundle: completeness and frozen dataclass contract
# ---------------------------------------------------------------------------

EXPECTED_RUNTIME_FIELDS = frozenset({
    "handle_alias_route",
    "resolve_cooldown_publication",
    "anthropic_family_state",
    "codex_family_state",
    "normalize_alias_model",
    "default_alias_model",
    "perform_candidate_request",
    "select_candidate",
    "publish_cooldown_memory",
    "persist_cooldown_durable",
    "set_session_affinity",
    "add_alias_metadata",
    "raise_redispatch_required",
    "get_candidates_for_alias",
    "get_active_cooldown_state",
})


class TestRuntimeBundle:
    """AnthropicAutoAgentRouteRuntime dataclass contract."""

    def test_is_frozen_dataclass(self) -> None:
        assert dataclasses.is_dataclass(AnthropicAutoAgentRouteRuntime)
        params = AnthropicAutoAgentRouteRuntime.__dataclass_params__  # type: ignore[attr-defined]
        assert params.frozen is True

    def test_field_names_match_expected(self) -> None:
        actual = {f.name for f in dataclasses.fields(AnthropicAutoAgentRouteRuntime)}
        assert actual == EXPECTED_RUNTIME_FIELDS

    def test_seam_disposition_covers_all_fields(self) -> None:
        field_names = {f.name for f in dataclasses.fields(AnthropicAutoAgentRouteRuntime)}
        assert set(ANTHROPIC_AUTO_AGENT_ROUTE_SEAM_DISPOSITION.keys()) == field_names

    def test_seam_disposition_values_are_runtime_prefixed(self) -> None:
        for key, value in ANTHROPIC_AUTO_AGENT_ROUTE_SEAM_DISPOSITION.items():
            assert value == f"runtime.{key}", f"Disposition mismatch for {key}"


# ---------------------------------------------------------------------------
# Behavioral: legacy facade delegation
# ---------------------------------------------------------------------------


def _make_runtime(**overrides: Any) -> AnthropicAutoAgentRouteRuntime:
    """Build a runtime with all-async-mock callbacks for testing."""
    defaults: dict[str, Any] = {
        "handle_alias_route": AsyncMock(return_value=MagicMock(name="response")),
        "resolve_cooldown_publication": MagicMock(),
        "anthropic_family_state": MagicMock(),
        "codex_family_state": MagicMock(),
        "normalize_alias_model": MagicMock(return_value=None),
        "default_alias_model": "claude-sonnet-4-20250514",
        "perform_candidate_request": AsyncMock(),
        "select_candidate": AsyncMock(),
        "publish_cooldown_memory": MagicMock(),
        "persist_cooldown_durable": AsyncMock(),
        "set_session_affinity": AsyncMock(),
        "add_alias_metadata": MagicMock(),
        "raise_redispatch_required": MagicMock(),
        "get_candidates_for_alias": MagicMock(return_value=[{"id": "c1"}]),
        "get_active_cooldown_state": AsyncMock(return_value=(0.0, "none")),
    }
    defaults.update(overrides)
    return AnthropicAutoAgentRouteRuntime(**defaults)


class TestLegacyFacadeDelegation:
    """handle_auto_agent_alias_route delegates correctly to runtime."""

    @pytest.mark.asyncio
    async def test_delegates_to_handle_alias_route(self) -> None:
        runtime = _make_runtime()
        request = MagicMock()
        await handle_auto_agent_alias_route(
            runtime,
            alias_family="anthropic_auto_agent",
            alias_model="claude-sonnet-4-20250514",
            request=request,
            prepared_request_body={"model": "claude-sonnet-4-20250514"},
            max_candidate_attempts=3,
            select_candidate_fn=AsyncMock(),
            add_alias_metadata_fn=MagicMock(),
            perform_candidate_request_fn=AsyncMock(),
            get_active_cooldown_state_fn=AsyncMock(return_value=(0.0, "none")),
            set_session_affinity_fn=AsyncMock(),
            apply_cooldown_fn=AsyncMock(),
            raise_redispatch_required_fn=MagicMock(),
            attempts_metadata_key="anthropic_auto_agent_attempts",
            skipped_candidates_metadata_key="anthropic_auto_agent_skipped_candidates",
            no_candidate_detail="No candidates.",
            log_label="Anthropic",
        )
        runtime.handle_alias_route.assert_awaited_once()  # type: ignore[union-attr]
        call_kwargs = runtime.handle_alias_route.call_args  # type: ignore[union-attr]
        assert call_kwargs.kwargs["alias_family"] == "anthropic_auto_agent"
        assert call_kwargs.kwargs["log_label"] == "Anthropic"

# ---------------------------------------------------------------------------
# Behavioral: family-state selection and cooldown-capture fail-closed
# ---------------------------------------------------------------------------


class TestFamilyStateSelectionAndCooldownCapture:
    """Verify family_state drives memory publish and cooldown queries,
    and that the legacy persist path fails closed when resolve_publication
    has not captured a request."""

    @pytest.mark.asyncio
    async def test_anthropic_family_memory_publish_uses_anthropic_state(self) -> None:
        """publish_cooldown_memory_fn writes to anthropic_family_state for
        alias_family='anthropic_auto_agent'."""
        runtime = _make_runtime()
        captured_services: list[Any] = []

        async def _grab(services: Any, **kw: Any) -> MagicMock:
            captured_services.append(services)
            return MagicMock()

        object.__setattr__(runtime, "handle_alias_route", _grab)

        await handle_auto_agent_alias_route(
            runtime,
            alias_family="anthropic_auto_agent",
            alias_model="m",
            request=MagicMock(),
            prepared_request_body={},
            max_candidate_attempts=1,
            select_candidate_fn=AsyncMock(),
            add_alias_metadata_fn=MagicMock(),
            perform_candidate_request_fn=AsyncMock(),
            get_active_cooldown_state_fn=AsyncMock(return_value=(0.0, "none")),
            set_session_affinity_fn=AsyncMock(),
            apply_cooldown_fn=AsyncMock(),
            raise_redispatch_required_fn=MagicMock(),
            attempts_metadata_key="k",
            skipped_candidates_metadata_key="sk",
            no_candidate_detail="none",
            log_label="Test",
        )
        services = captured_services[0]
        # Invoke the memory publisher
        services.publish_cooldown_memory_fn(keys=["lane:a", "lane:b"], seconds=30.0)
        runtime.anthropic_family_state.set_cooldown_memory.assert_any_call("lane:a", 30.0)  # type: ignore[union-attr]
        runtime.anthropic_family_state.set_cooldown_memory.assert_any_call("lane:b", 30.0)  # type: ignore[union-attr]
        runtime.codex_family_state.set_cooldown_memory.assert_not_called()  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_codex_family_memory_publish_uses_codex_state(self) -> None:
        """publish_cooldown_memory_fn writes to codex_family_state for
        alias_family='codex_auto_agent'."""
        runtime = _make_runtime()
        captured_services: list[Any] = []

        async def _grab(services: Any, **kw: Any) -> MagicMock:
            captured_services.append(services)
            return MagicMock()

        object.__setattr__(runtime, "handle_alias_route", _grab)

        await handle_auto_agent_alias_route(
            runtime,
            alias_family="codex_auto_agent",
            alias_model="m",
            request=MagicMock(),
            prepared_request_body={},
            max_candidate_attempts=1,
            select_candidate_fn=AsyncMock(),
            add_alias_metadata_fn=MagicMock(),
            perform_candidate_request_fn=AsyncMock(),
            get_active_cooldown_state_fn=AsyncMock(return_value=(0.0, "none")),
            set_session_affinity_fn=AsyncMock(),
            apply_cooldown_fn=AsyncMock(),
            raise_redispatch_required_fn=MagicMock(),
            attempts_metadata_key="k",
            skipped_candidates_metadata_key="sk",
            no_candidate_detail="none",
            log_label="Test",
        )
        services = captured_services[0]
        services.publish_cooldown_memory_fn(keys=["cx:1"], seconds=10.0)
        runtime.codex_family_state.set_cooldown_memory.assert_called_once_with("cx:1", 10.0)  # type: ignore[union-attr]
        runtime.anthropic_family_state.set_cooldown_memory.assert_not_called()  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_cooldown_state_memory_short_circuits_fn(self) -> None:
        """When family_state memory reports >0, the injected
        get_active_cooldown_state_fn is NOT called."""
        runtime = _make_runtime()
        runtime.anthropic_family_state.get_memory_cooldown_remaining.return_value = 42.0  # type: ignore[union-attr]
        injected_fn = AsyncMock(return_value=(0.0, "redis"))
        captured_kwargs: list[dict[str, Any]] = []

        async def _grab(services: Any, **kw: Any) -> MagicMock:
            captured_kwargs.append(kw)
            return MagicMock()

        object.__setattr__(runtime, "handle_alias_route", _grab)

        await handle_auto_agent_alias_route(
            runtime,
            alias_family="anthropic_auto_agent",
            alias_model="m",
            request=MagicMock(),
            prepared_request_body={},
            max_candidate_attempts=1,
            select_candidate_fn=AsyncMock(),
            add_alias_metadata_fn=MagicMock(),
            perform_candidate_request_fn=AsyncMock(),
            get_active_cooldown_state_fn=injected_fn,
            set_session_affinity_fn=AsyncMock(),
            apply_cooldown_fn=AsyncMock(),
            raise_redispatch_required_fn=MagicMock(),
            attempts_metadata_key="k",
            skipped_candidates_metadata_key="sk",
            no_candidate_detail="none",
            log_label="Test",
        )
        # The wrapped cooldown fn was passed as get_active_cooldown_state_fn kwarg
        wrapped_fn = captured_kwargs[0]["get_active_cooldown_state_fn"]
        result = await wrapped_fn("some_key")
        assert result == (42.0, "memory")
        injected_fn.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_codex_cooldown_state_memory_short_circuits_fn(self) -> None:
        """Codex aliases query Codex family memory and skip the fallback."""
        fallback_fn = AsyncMock(return_value=(0.0, "redis"))

        async def _exercise(services: Any, **kwargs: Any) -> tuple[float, str]:
            return await kwargs["get_active_cooldown_state_fn"]("codex:key")

        runtime = _make_runtime(handle_alias_route=_exercise)
        runtime.codex_family_state.get_memory_cooldown_remaining.return_value = 17.0  # type: ignore[union-attr]

        result = await handle_auto_agent_alias_route(
            runtime,
            alias_family="codex_auto_agent",
            alias_model="m",
            request=MagicMock(),
            prepared_request_body={},
            max_candidate_attempts=1,
            select_candidate_fn=AsyncMock(),
            add_alias_metadata_fn=MagicMock(),
            perform_candidate_request_fn=AsyncMock(),
            get_active_cooldown_state_fn=fallback_fn,
            set_session_affinity_fn=AsyncMock(),
            apply_cooldown_fn=AsyncMock(),
            raise_redispatch_required_fn=MagicMock(),
            attempts_metadata_key="k",
            skipped_candidates_metadata_key="sk",
            no_candidate_detail="none",
            log_label="Codex",
        )

        assert result == (17.0, "memory")
        runtime.codex_family_state.get_memory_cooldown_remaining.assert_called_once_with(  # type: ignore[union-attr]
            "codex:key"
        )
        runtime.anthropic_family_state.get_memory_cooldown_remaining.assert_not_called()  # type: ignore[union-attr]
        fallback_fn.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cooldown_state_falls_through_to_injected_fn(self) -> None:
        """When family_state memory reports 0, the injected fn is called."""
        runtime = _make_runtime()
        runtime.anthropic_family_state.get_memory_cooldown_remaining.return_value = 0.0  # type: ignore[union-attr]
        injected_fn = AsyncMock(return_value=(99.0, "redis"))
        captured_kwargs: list[dict[str, Any]] = []

        async def _grab(services: Any, **kw: Any) -> MagicMock:
            captured_kwargs.append(kw)
            return MagicMock()

        object.__setattr__(runtime, "handle_alias_route", _grab)

        await handle_auto_agent_alias_route(
            runtime,
            alias_family="anthropic_auto_agent",
            alias_model="m",
            request=MagicMock(),
            prepared_request_body={},
            max_candidate_attempts=1,
            select_candidate_fn=AsyncMock(),
            add_alias_metadata_fn=MagicMock(),
            perform_candidate_request_fn=AsyncMock(),
            get_active_cooldown_state_fn=injected_fn,
            set_session_affinity_fn=AsyncMock(),
            apply_cooldown_fn=AsyncMock(),
            raise_redispatch_required_fn=MagicMock(),
            attempts_metadata_key="k",
            skipped_candidates_metadata_key="sk",
            no_candidate_detail="none",
            log_label="Test",
        )
        wrapped_fn = captured_kwargs[0]["get_active_cooldown_state_fn"]
        result = await wrapped_fn("key_x")
        assert result == (99.0, "redis")
        injected_fn.assert_awaited_once_with("key_x")

    @pytest.mark.asyncio
    async def test_legacy_persist_fails_closed_without_resolve(self) -> None:
        """_legacy_persist raises RuntimeError if resolve_publication was
        never called (request not captured)."""
        runtime = _make_runtime()
        captured_services: list[Any] = []

        async def _grab(services: Any, **kw: Any) -> MagicMock:
            captured_services.append(services)
            return MagicMock()

        object.__setattr__(runtime, "handle_alias_route", _grab)

        await handle_auto_agent_alias_route(
            runtime,
            alias_family="anthropic_auto_agent",
            alias_model="m",
            request=MagicMock(),
            prepared_request_body={},
            max_candidate_attempts=1,
            select_candidate_fn=AsyncMock(),
            add_alias_metadata_fn=MagicMock(),
            perform_candidate_request_fn=AsyncMock(),
            get_active_cooldown_state_fn=AsyncMock(return_value=(0.0, "none")),
            set_session_affinity_fn=AsyncMock(),
            apply_cooldown_fn=AsyncMock(),
            raise_redispatch_required_fn=MagicMock(),
            attempts_metadata_key="k",
            skipped_candidates_metadata_key="sk",
            no_candidate_detail="none",
            log_label="Test",
        )
        services = captured_services[0]
        # persist_cooldown_fn is _legacy_persist; calling it without prior
        # resolve_publication must raise RuntimeError (fail-closed)
        with pytest.raises(RuntimeError, match="did not capture a request"):
            await services.persist_cooldown_fn(keys=["k"], seconds=5.0)

    @pytest.mark.asyncio
    async def test_legacy_resolve_publication_captures_all_state(self) -> None:
        """_legacy_resolve_publication round-trips all nonlocal state to the
        underlying runtime.resolve_cooldown_publication and makes persist
        callable."""
        runtime = _make_runtime()
        apply_mock = AsyncMock(return_value="applied")
        captured_services: list[Any] = []

        async def _grab(services: Any, **kw: Any) -> MagicMock:
            captured_services.append(services)
            return MagicMock()

        object.__setattr__(runtime, "handle_alias_route", _grab)

        req = MagicMock(name="the_request")
        await handle_auto_agent_alias_route(
            runtime,
            alias_family="anthropic_auto_agent",
            alias_model="m",
            request=req,
            prepared_request_body={},
            max_candidate_attempts=1,
            select_candidate_fn=AsyncMock(),
            add_alias_metadata_fn=MagicMock(),
            perform_candidate_request_fn=AsyncMock(),
            get_active_cooldown_state_fn=AsyncMock(return_value=(0.0, "none")),
            set_session_affinity_fn=AsyncMock(),
            apply_cooldown_fn=apply_mock,
            raise_redispatch_required_fn=MagicMock(),
            attempts_metadata_key="k",
            skipped_candidates_metadata_key="sk",
            no_candidate_detail="none",
            log_label="Test",
        )
        services = captured_services[0]
        # Call resolve_publication to capture state
        services.resolve_cooldown_publication_fn(
            request=req,
            candidate={"provider": "anthropic"},
            lane_key="lane:1",
            selected_cooldown_key="cd:1",
            cooldown_seconds=60.0,
            error_class="RateLimitError",
            grok_account_quota_exhausted=True,
            kimi_failure_metadata={"k": "v"},
            is_basic_pilot_lane=True,
        )
        runtime.resolve_cooldown_publication.assert_called_once_with(  # type: ignore[union-attr]
            request=req,
            candidate={"provider": "anthropic"},
            lane_key="lane:1",
            selected_cooldown_key="cd:1",
            cooldown_seconds=60.0,
            error_class="RateLimitError",
            grok_account_quota_exhausted=True,
            kimi_failure_metadata={"k": "v"},
            is_basic_pilot_lane=True,
        )
        # Now persist should succeed and forward captured state to apply_cooldown_fn
        await services.persist_cooldown_fn(keys=["cd:1"], seconds=60.0)
        apply_mock.assert_awaited_once()
        call_kw = apply_mock.call_args.kwargs
        assert call_kw["request"] is req
        assert call_kw["candidate"] == {"provider": "anthropic"}
        assert call_kw["lane_key"] == "lane:1"
        assert call_kw["selected_cooldown_key"] == "cd:1"
        assert call_kw["cooldown_seconds"] == 60.0
        assert call_kw["error_class"] == "RateLimitError"
        assert call_kw["grok_account_quota_exhausted"] is True
        assert call_kw["kimi_failure_metadata"] == {"k": "v"}
        assert call_kw["is_basic_pilot_lane"] is True


# ---------------------------------------------------------------------------
# Behavioral: production wrapper perform-function binding
# ---------------------------------------------------------------------------


class TestPerformFunctionBinding:
    """Verify the production wrapper's _perform_candidate_request closure
    binds all call-site parameters correctly."""

    @pytest.mark.asyncio
    async def test_perform_closure_forwards_all_kwargs(self) -> None:
        """The closure passed as perform_candidate_request_fn must forward
        endpoint, request, fastapi_response, user_api_key_dict, target_url,
        and custom_headers from the call site."""
        captured: dict[str, Any] = {}

        async def _capture_perform(**kwargs: Any) -> MagicMock:
            captured.update(kwargs)
            return _sentinel_resp

        runtime = _make_runtime(perform_candidate_request=_capture_perform)
        captured_services: list[Any] = []
        _sentinel_resp = MagicMock()

        async def _grab(services: Any, **kw: Any) -> MagicMock:
            captured_services.append(services)
            return MagicMock()

        object.__setattr__(runtime, "handle_alias_route", _grab)

        req = MagicMock(name="req")
        resp = MagicMock(name="resp")
        uak = MagicMock(name="uak")
        headers = {"x-custom": "val"}

        await handle_anthropic_auto_agent_alias_route(
            runtime,
            endpoint="/v1/messages",
            request=req,
            fastapi_response=resp,
            user_api_key_dict=uak,
            prepared_request_body={"model": "claude-sonnet-4-20250514"},
            target_url="https://api.anthropic.com/v1/messages",
            custom_headers=headers,
        )
        services = captured_services[0]
        candidate = {"provider": "anthropic", "model": "claude-sonnet-4-20250514"}
        candidate_body = {"model": "claude-sonnet-4-20250514", "messages": []}
        result = await services.perform_candidate_request_fn(
            candidate=candidate, candidate_body=candidate_body
        )
        assert captured["endpoint"] == "/v1/messages"
        assert captured["request"] is req
        assert captured["fastapi_response"] is resp
        assert captured["user_api_key_dict"] is uak
        assert captured["candidate"] is candidate
        assert captured["candidate_body"] is candidate_body
        assert captured["target_url"] == "https://api.anthropic.com/v1/messages"
        assert captured["custom_headers"] is headers
        assert result is _sentinel_resp

    @pytest.mark.asyncio
    async def test_perform_closure_returns_exact_response(self) -> None:
        """The closure must return the exact object from
        runtime.perform_candidate_request."""
        sentinel = MagicMock(name="exact_response_sentinel")

        async def _return_sentinel(**kwargs: Any) -> MagicMock:
            return sentinel

        runtime = _make_runtime(perform_candidate_request=_return_sentinel)
        captured_services: list[Any] = []

        async def _grab(services: Any, **kw: Any) -> MagicMock:
            captured_services.append(services)
            return MagicMock()

        object.__setattr__(runtime, "handle_alias_route", _grab)

        await handle_anthropic_auto_agent_alias_route(
            runtime,
            endpoint="/v1/messages",
            request=MagicMock(),
            fastapi_response=MagicMock(),
            user_api_key_dict=MagicMock(),
            prepared_request_body={},
            target_url="https://api.anthropic.com",
            custom_headers={},
        )
        services = captured_services[0]
        result = await services.perform_candidate_request_fn(
            candidate={}, candidate_body={}
        )
        assert result is sentinel


# ---------------------------------------------------------------------------
# Behavioral: production wrapper delegation
# ---------------------------------------------------------------------------


class TestProductionWrapperDelegation:
    """handle_anthropic_auto_agent_alias_route delegates correctly."""

    @pytest.mark.asyncio
    async def test_delegates_with_correct_alias_family(self) -> None:
        runtime = _make_runtime()
        await handle_anthropic_auto_agent_alias_route(
            runtime,
            endpoint="/v1/messages",
            request=MagicMock(),
            fastapi_response=MagicMock(),
            user_api_key_dict=MagicMock(),
            prepared_request_body={"model": "claude-sonnet-4-20250514"},
            target_url="https://api.anthropic.com",
            custom_headers={},
        )
        runtime.handle_alias_route.assert_awaited_once()  # type: ignore[union-attr]
        call_kwargs = runtime.handle_alias_route.call_args.kwargs  # type: ignore[union-attr]
        assert call_kwargs["alias_family"] == "anthropic_auto_agent"
        assert call_kwargs["log_label"] == "Anthropic"
        assert call_kwargs["attempts_metadata_key"] == "anthropic_auto_agent_attempts"
        assert (
            call_kwargs["skipped_candidates_metadata_key"]
            == "anthropic_auto_agent_skipped_candidates"
        )

    @pytest.mark.asyncio
    async def test_alias_model_normalization_fallback(self) -> None:
        """When normalize returns None, default_alias_model is used."""
        runtime = _make_runtime(
            normalize_alias_model=MagicMock(return_value=None),
            default_alias_model="claude-fallback",
        )
        await handle_anthropic_auto_agent_alias_route(
            runtime,
            endpoint="/v1/messages",
            request=MagicMock(),
            fastapi_response=MagicMock(),
            user_api_key_dict=MagicMock(),
            prepared_request_body={"model": "unknown"},
            target_url="https://api.anthropic.com",
            custom_headers={},
        )
        call_kwargs = runtime.handle_alias_route.call_args.kwargs  # type: ignore[union-attr]
        assert call_kwargs["alias_model"] == "claude-fallback"

    @pytest.mark.asyncio
    async def test_alias_model_normalization_hit(self) -> None:
        """When normalize returns a model, it is used directly."""
        runtime = _make_runtime(
            normalize_alias_model=MagicMock(return_value="claude-opus-4-20250514"),
        )
        await handle_anthropic_auto_agent_alias_route(
            runtime,
            endpoint="/v1/messages",
            request=MagicMock(),
            fastapi_response=MagicMock(),
            user_api_key_dict=MagicMock(),
            prepared_request_body={"model": "opus"},
            target_url="https://api.anthropic.com",
            custom_headers={},
        )
        call_kwargs = runtime.handle_alias_route.call_args.kwargs  # type: ignore[union-attr]
        assert call_kwargs["alias_model"] == "claude-opus-4-20250514"

    @pytest.mark.asyncio
    async def test_max_candidate_attempts_from_candidates_list(self) -> None:
        """max_candidate_attempts equals len(get_candidates_for_alias(alias))."""
        get_candidates = MagicMock(return_value=[{"a": 1}, {"b": 2}, {"c": 3}])
        runtime = _make_runtime(get_candidates_for_alias=get_candidates)
        request = MagicMock()
        body: dict[str, Any] = {}
        await handle_anthropic_auto_agent_alias_route(
            runtime,
            endpoint="/v1/messages",
            request=request,
            fastapi_response=MagicMock(),
            user_api_key_dict=MagicMock(),
            prepared_request_body=body,
            target_url="https://api.anthropic.com",
            custom_headers={},
        )
        call_kwargs = runtime.handle_alias_route.call_args.kwargs  # type: ignore[union-attr]
        assert call_kwargs["max_candidate_attempts"] == 3
        get_candidates.assert_called_once_with(
            "claude-sonnet-4-20250514",
            request=request,
            request_body=body,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("alias_model", "expected_attempts"),
        (
            ("sota-moonshot", 1),
            ("sota-alibaba", 2),
            ("work", 6),
        ),
    )
    async def test_installed_runtime_counts_checked_in_anthropic_alias_candidates(
        self,
        alias_model: str,
        expected_attempts: int,
    ) -> None:
        """The production runtime uses the request-aware snapshot projection."""
        from starlette.requests import Request
        from starlette.responses import Response

        from litellm.proxy.pass_through_endpoints import (
            llm_passthrough_endpoints as lpe,
        )
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_snapshot import (
            active_routing_snapshot_holder,
        )
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_startup import (
            DEFAULT_CONFIG_DIR,
            compile_directory,
        )

        snapshot = compile_directory(DEFAULT_CONFIG_DIR)
        previous_snapshot = active_routing_snapshot_holder.swap(snapshot)
        handle_alias_route = AsyncMock(return_value=Response(status_code=204))
        request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/anthropic/v1/messages",
                "query_string": b"",
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"x-litellm-end-user-id", b"adapter-harness-tenant"),
                    (b"langfuse_trace_user_id", b"adapter-harness-tenant"),
                    (b"langfuse_trace_name", b"claude-code"),
                    (b"x-aawm-tenant-id", b"adapter-harness-tenant"),
                ],
            }
        )
        body = {
            "model": alias_model,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "offline harness probe"}],
            "tools": [
                {
                    "name": "Read",
                    "description": "offline",
                    "input_schema": {"type": "object"},
                }
            ],
        }
        runtime = dataclasses.replace(
            lpe._ANTHROPIC_AUTO_AGENT_ROUTE_RUNTIME,
            handle_alias_route=handle_alias_route,
        )

        try:
            await handle_anthropic_auto_agent_alias_route(
                runtime,
                endpoint="/v1/messages",
                request=request,
                fastapi_response=Response(),
                user_api_key_dict=MagicMock(),
                prepared_request_body=body,
                target_url="http://unused.invalid",
                custom_headers={},
            )
        finally:
            active_routing_snapshot_holder.swap(previous_snapshot)

        assert (
            handle_alias_route.await_args.kwargs["max_candidate_attempts"]
            == expected_attempts
        )

    @pytest.mark.asyncio
    async def test_services_bundle_wired_from_runtime(self) -> None:
        """The AliasRouteServices is assembled from runtime fields."""
        runtime = _make_runtime()
        await handle_anthropic_auto_agent_alias_route(
            runtime,
            endpoint="/v1/messages",
            request=MagicMock(),
            fastapi_response=MagicMock(),
            user_api_key_dict=MagicMock(),
            prepared_request_body={},
            target_url="https://api.anthropic.com",
            custom_headers={},
        )
        services = runtime.handle_alias_route.call_args.args[0]  # type: ignore[union-attr]
        assert services.select_candidate_fn is runtime.select_candidate
        assert services.resolve_cooldown_publication_fn is runtime.resolve_cooldown_publication
        assert services.publish_cooldown_memory_fn is runtime.publish_cooldown_memory
        assert services.persist_cooldown_fn is runtime.persist_cooldown_durable
        assert services.set_session_affinity_fn is runtime.set_session_affinity
        assert services.add_alias_metadata_fn is runtime.add_alias_metadata
        assert services.raise_redispatch_fn is runtime.raise_redispatch_required

    @pytest.mark.asyncio
    async def test_no_candidate_detail_string(self) -> None:
        runtime = _make_runtime()
        await handle_anthropic_auto_agent_alias_route(
            runtime,
            endpoint="/v1/messages",
            request=MagicMock(),
            fastapi_response=MagicMock(),
            user_api_key_dict=MagicMock(),
            prepared_request_body={},
            target_url="https://api.anthropic.com",
            custom_headers={},
        )
        call_kwargs = runtime.handle_alias_route.call_args.kwargs  # type: ignore[union-attr]
        assert (
            call_kwargs["no_candidate_detail"]
            == "No Anthropic auto-agent alias candidates were available."
        )
