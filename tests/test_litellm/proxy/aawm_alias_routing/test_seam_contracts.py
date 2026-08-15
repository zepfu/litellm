"""Wave 0/2 guardrail: seam-contract tests for the alias-routing retry loop.

These tests turn the ``b9c97f9540``-class silent stub-rot failure mode (a
production applicator gaining a new required/optional kwarg that a stub
mirror silently falls behind on) into named, collection-time-stable
assertions, per
``.analysis/plan-godmodule-decomposition-r3-remediation-2026-07-23.md``
Wave 0 (kwarg/key contracts + state reset) and Wave 2 (the
``AliasRouteServices`` typed seam bundle that replaces the type-erased
``Callable[..., ...]`` seams the candidate_loop extraction introduces).

``test_reset_alias_routing_state_for_tests_clears_everything`` is RED until
the Wave-0 engineer adds ``reset_alias_routing_state_for_tests()`` -- that is
expected and intentional; do not add the helper here.

``test_alias_route_services_signature_contracts`` is RED until the Wave-2
engineer creates
``litellm.proxy.pass_through_endpoints.aawm_alias_routing.interfaces``
(the ``AliasRouteServices`` frozen dataclass bundling the typed
``SelectCandidateFn`` / ``PerformCandidateRequestFn`` /
``ResolveCooldownPublicationFn`` / ``PublishCooldownMemoryFn`` /
``PersistCooldownFn`` / ``SetSessionAffinityFn`` / ``AddAliasMetadataFn`` /
``RaiseRedispatchFn`` protocols) -- that is expected and intentional; do not
create the module here.
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
from types import SimpleNamespace
from typing import Any, Callable, Optional, Sequence
from unittest.mock import MagicMock

import pytest
from fastapi import Request
from starlette.responses import Response

from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    candidate_loop,
    classification,
    config_compiler as compiler,
    selection,
    snapshot_select,
)
from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime import (
    codex_auto_agent_route,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.interfaces import (
    AliasRouteServices,
    CooldownPublicationPlan,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
    alias_routing_state,
)

# The exact key set the loop consumes off the ``select_candidate_fn`` return
# value (``llm_passthrough_endpoints.py:22143-22187``).
_SELECT_CANDIDATE_REQUIRED_KEYS = {
    "candidate",
    "lane_key",
    "cooldown_key",
    "session_key",
    "selection_reason",
    "skipped",
    "in_flight_session",
}


@pytest.mark.asyncio
async def test_select_candidate_fn_returns_required_selection_keys() -> None:
    """``_select_codex_auto_agent_candidate`` returns exactly the keys the loop consumes.

    Drives the session-affinity branch (continuation state + a matching
    session-affinity record) because that is the branch that returns
    ``in_flight_session`` -- the plain first-available branch omits it.
    """
    raw_yaml = """
defaults: {}
aliases:
  - name: seam-contract
    candidates:
      - provider: openrouter
        model: openrouter/seam-contract-model
        route_family: codex_openrouter_completion_adapter
        priority: 900
"""
    snapshot = compiler.compile_yaml(raw_yaml)
    previous_snapshot = snapshot_select.get_active_routing_snapshot()
    snapshot_select.set_active_routing_snapshot(snapshot)
    request = MagicMock(spec=Request)
    request.method = "POST"
    request.headers = {"session_id": "seam-contract-session"}
    request.query_params = {}
    request.state = SimpleNamespace()
    request.state.aawm_alias_request_local_cooldown_until = {}
    request.state.aawm_alias_request_local_excluded_keys = set()
    request_body = {
        "model": "seam-contract",
        "previous_response_id": "resp_seam_contract",
    }
    session_key = lpe._resolve_codex_auto_agent_session_key(
        request,
        request_body,
        alias_model="seam-contract",
    )
    assert session_key is not None
    previous_affinity = alias_routing_state.codex.session_affinity_by_key.get(session_key)
    alias_routing_state.codex.session_affinity_by_key[session_key] = {
        "provider": "openrouter",
        "model": "openrouter/seam-contract-model",
        "route_family": "codex_openrouter_completion_adapter",
        "last_resort": False,
        "expires_at_monotonic": __import__("time").monotonic() + 3600.0,
    }
    try:
        selection_result = await selection._select_codex_auto_agent_candidate(
            request=request,
            request_body=request_body,
        )
    finally:
        snapshot_select.set_active_routing_snapshot(previous_snapshot)
        if previous_affinity is None:
            alias_routing_state.codex.session_affinity_by_key.pop(session_key, None)
        else:
            alias_routing_state.codex.session_affinity_by_key[session_key] = previous_affinity

    assert selection_result.get("selection_reason") == "session_affinity"
    assert _SELECT_CANDIDATE_REQUIRED_KEYS <= set(selection_result.keys()), (
        "_select_codex_auto_agent_candidate no longer returns every key the "
        f"retry loop consumes: missing {_SELECT_CANDIDATE_REQUIRED_KEYS - set(selection_result.keys())}"
    )


def test_reset_alias_routing_state_for_tests_clears_everything() -> None:
    """RED until Wave-0 engineer adds ``reset_alias_routing_state_for_tests()``.

    Once added, the helper must clear: both family (codex/anthropic)
    cooldown/negative/affinity/evidence maps, ``candidate_probe_locks``, the
    alias-scoped Codex failure-evidence state,
    ``_round_robin_cursor_by_alias``, and the active routing snapshot (set to
    ``None``).
    """
    reset_fn = getattr(lpe, "reset_alias_routing_state_for_tests", None)
    assert reset_fn is not None, (
        "reset_alias_routing_state_for_tests() does not exist yet -- expected "
        "RED until the Wave-0 engineer lands it."
    )

    # Seed every piece of state the helper is required to clear.
    alias_routing_state.codex.cooldown_until_monotonic_by_key["seed"] = 1.0
    alias_routing_state.codex.cooldown_negative_until_monotonic_by_key["seed"] = 1.0
    alias_routing_state.codex.session_affinity_by_key["seed"] = {"provider": "p", "model": "m"}
    alias_routing_state.codex.evidence_events_by_key["seed"] = [1.0]
    alias_routing_state.anthropic.cooldown_until_monotonic_by_key["seed"] = 1.0
    alias_routing_state.anthropic.cooldown_negative_until_monotonic_by_key["seed"] = 1.0
    alias_routing_state.anthropic.session_affinity_by_key["seed"] = {"provider": "p", "model": "m"}
    alias_routing_state.anthropic.evidence_events_by_key["seed"] = [1.0]
    alias_routing_state.candidate_probe_locks["seed"] = asyncio.Lock()
    evidence = classification.classify_failure(
        status_code=429,
        provider="openrouter",
        message="rate limited",
    )
    alias_routing_state.codex_failure_evidence_gate.record(
        canonical_alias="reset-alias",
        cooldown_key="seed",
        event=evidence,
    )
    alias_routing_state.round_robin_cursor[("reset-alias", "seed")] = 1

    raw_yaml = """
defaults: {}
aliases:
  - name: reset-alias
    candidates:
      - provider: openrouter
        model: openrouter/reset-helper-model
        route_family: codex_openrouter_completion_adapter
        priority: 900
"""
    snapshot_select.set_active_routing_snapshot(compiler.compile_yaml(raw_yaml))

    reset_fn()

    assert alias_routing_state.codex.cooldown_until_monotonic_by_key == {}
    assert alias_routing_state.codex.cooldown_negative_until_monotonic_by_key == {}
    assert alias_routing_state.codex.session_affinity_by_key == {}
    assert alias_routing_state.codex.evidence_events_by_key == {}
    assert alias_routing_state.anthropic.cooldown_until_monotonic_by_key == {}
    assert alias_routing_state.anthropic.cooldown_negative_until_monotonic_by_key == {}
    assert alias_routing_state.anthropic.session_affinity_by_key == {}
    assert alias_routing_state.anthropic.evidence_events_by_key == {}
    assert alias_routing_state.candidate_probe_locks == {}
    assert (
        alias_routing_state.codex_failure_evidence_gate.gate_for_alias(
            canonical_alias="reset-alias"
        )
        is None
    )
    assert alias_routing_state.round_robin_cursor == {}
    assert snapshot_select.get_active_routing_snapshot() is None


# ---------------------------------------------------------------------------
# Wave 2: AliasRouteServices typed seam contract (RED until interfaces.py lands)
# ---------------------------------------------------------------------------

# The exact protocol attribute names ``AliasRouteServices`` must bundle, per
# the Wave-2 Source Spec. Encoded explicitly so a future rename/removal on
# any one callback is a named, readable failure.
_ALIAS_ROUTE_SERVICES_CALLBACK_FIELDS = (
    "select_candidate_fn",
    "perform_candidate_request_fn",
    "resolve_cooldown_publication_fn",
    "publish_cooldown_memory_fn",
    "persist_cooldown_fn",
    "set_session_affinity_fn",
    "add_alias_metadata_fn",
    "raise_redispatch_fn",
)

# The exact required keyword names ``PublishCooldownMemoryFn`` must declare
# (Wave-2 Source Spec: ``(*, keys: Sequence[str], seconds: float) -> None``).
_PUBLISH_COOLDOWN_MEMORY_FN_REQUIRED_KWARGS = ("keys", "seconds")


_CALLBACK_PARAMETER_KINDS: dict[str, dict[str, inspect._ParameterKind]] = {
    "select_candidate_fn": {
        "request": inspect.Parameter.KEYWORD_ONLY,
        "request_body": inspect.Parameter.KEYWORD_ONLY,
    },
    "perform_candidate_request_fn": {
        "candidate": inspect.Parameter.KEYWORD_ONLY,
        "candidate_body": inspect.Parameter.KEYWORD_ONLY,
    },
    "resolve_cooldown_publication_fn": {
        "request": inspect.Parameter.KEYWORD_ONLY,
        "candidate": inspect.Parameter.KEYWORD_ONLY,
        "lane_key": inspect.Parameter.KEYWORD_ONLY,
        "selected_cooldown_key": inspect.Parameter.KEYWORD_ONLY,
        "cooldown_seconds": inspect.Parameter.KEYWORD_ONLY,
        "error_class": inspect.Parameter.KEYWORD_ONLY,
        "grok_account_quota_exhausted": inspect.Parameter.KEYWORD_ONLY,
        "kimi_failure_metadata": inspect.Parameter.KEYWORD_ONLY,
        "codex_failure_evidence_alias": inspect.Parameter.KEYWORD_ONLY,
    },
    "publish_cooldown_memory_fn": {
        "keys": inspect.Parameter.KEYWORD_ONLY,
        "seconds": inspect.Parameter.KEYWORD_ONLY,
        "allow_ttl_shrink": inspect.Parameter.KEYWORD_ONLY,
    },
    "persist_cooldown_fn": {
        "keys": inspect.Parameter.KEYWORD_ONLY,
        "seconds": inspect.Parameter.KEYWORD_ONLY,
        "allow_ttl_shrink": inspect.Parameter.KEYWORD_ONLY,
    },
    "set_session_affinity_fn": {
        "session_key": inspect.Parameter.POSITIONAL_OR_KEYWORD,
        "candidate": inspect.Parameter.POSITIONAL_OR_KEYWORD,
    },
    "add_alias_metadata_fn": {
        "request_body": inspect.Parameter.POSITIONAL_OR_KEYWORD,
        "request": inspect.Parameter.KEYWORD_ONLY,
        "selection": inspect.Parameter.KEYWORD_ONLY,
        "attempts": inspect.Parameter.KEYWORD_ONLY,
    },
    "raise_redispatch_fn": {
        "candidate": inspect.Parameter.KEYWORD_ONLY,
        "lane_key": inspect.Parameter.KEYWORD_ONLY,
        "cooldown_seconds": inspect.Parameter.KEYWORD_ONLY,
        "error_tokens": inspect.Parameter.KEYWORD_ONLY,
        "alias_model": inspect.Parameter.KEYWORD_ONLY,
        "error_class": inspect.Parameter.KEYWORD_ONLY,
        "cooldown_scope": inspect.Parameter.KEYWORD_ONLY,
        "error_status_code": inspect.Parameter.KEYWORD_ONLY,
        "error_type": inspect.Parameter.KEYWORD_ONLY,
        "error_code": inspect.Parameter.KEYWORD_ONLY,
        "retry_after_seconds": inspect.Parameter.KEYWORD_ONLY,
        "failure_phase": inspect.Parameter.KEYWORD_ONLY,
        "attempted_provider_call": inspect.Parameter.KEYWORD_ONLY,
        "audit_events": inspect.Parameter.KEYWORD_ONLY,
        "attempts": inspect.Parameter.KEYWORD_ONLY,
        "skipped_candidates": inspect.Parameter.KEYWORD_ONLY,
    },
}

_CALLBACK_COROUTINE_STATUS = {
    "select_candidate_fn": True,
    "perform_candidate_request_fn": True,
    "resolve_cooldown_publication_fn": False,
    "publish_cooldown_memory_fn": False,
    "persist_cooldown_fn": True,
    "set_session_affinity_fn": True,
    "add_alias_metadata_fn": False,
    "raise_redispatch_fn": False,
}

# ---------------------------------------------------------------------------
# Explicit field-to-production-target mapping for DI forwarding validation.
#
# The god module builds each runtime bundle with forwarding lambdas of the
# form ``lambda *args, **kwargs: <lpe_global>(*args, **kwargs)``.  These
# erase the typed signature and coroutine flag of the real target.  The
# maps below name the ``lpe`` module global each DI-wrapped field delegates
# to, so the test can (a) behaviorally prove forwarding with a sentinel and
# (b) validate the real target's signature directly.
#
# Fields absent from these maps (``perform_candidate_request_fn``) are typed
# closures built inside the handler and are validated on the captured
# callback itself.
# ---------------------------------------------------------------------------

_CODEX_DI_FORWARDING_TARGETS: dict[str, str] = {
    "select_candidate_fn": "_select_codex_auto_agent_candidate",
    "resolve_cooldown_publication_fn": "_resolve_auto_agent_cooldown_publication_plan",
    "publish_cooldown_memory_fn": "_publish_codex_cooldown_memory",
    "persist_cooldown_fn": "_persist_codex_cooldown_durable",
    "set_session_affinity_fn": "_set_codex_auto_agent_session_affinity",
    "add_alias_metadata_fn": "_add_codex_auto_agent_alias_metadata",
    "raise_redispatch_fn": "_raise_codex_auto_agent_redispatch_required",
}

_ANTHROPIC_DI_FORWARDING_TARGETS: dict[str, str] = {
    "select_candidate_fn": "_select_anthropic_auto_agent_candidate",
    "resolve_cooldown_publication_fn": "_resolve_auto_agent_cooldown_publication_plan",
    "publish_cooldown_memory_fn": "_publish_anthropic_cooldown_memory",
    "persist_cooldown_fn": "_persist_anthropic_cooldown_durable",
    "set_session_affinity_fn": "_set_anthropic_auto_agent_session_affinity",
    "add_alias_metadata_fn": "_add_anthropic_auto_agent_alias_metadata",
    "raise_redispatch_fn": "_raise_anthropic_auto_agent_redispatch_required",
}

_DI_FORWARDING_TARGETS_BY_FAMILY: dict[str, dict[str, str]] = {
    "codex_auto_agent": _CODEX_DI_FORWARDING_TARGETS,
    "anthropic_auto_agent": _ANTHROPIC_DI_FORWARDING_TARGETS,
}


def _signature_contract_request() -> MagicMock:
    request = MagicMock(spec=Request)
    request.method = "POST"
    request.headers = {"session_id": "signature-contract"}
    request.query_params = {}
    request.url = MagicMock()
    request.scope = {
        "path": "/openai_passthrough/v1/responses",
        "query_string": b"",
        "parsed_body": None,
    }
    request.state = MagicMock()
    request.state.aawm_alias_request_local_cooldown_until = {}
    request.state.aawm_alias_request_local_excluded_keys = set()
    return request


@pytest.mark.asyncio
async def test_alias_route_services_signature_contracts(  # noqa: PLR0915
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inspect every callback actually bound by both production wrappers."""
    captured: dict[str, AliasRouteServices] = {}

    async def _capture_services(
        services: AliasRouteServices,
        *,
        alias_family: str,
        alias_model: str,
        request: Request,
        prepared_request_body: dict[str, Any],
        max_candidate_attempts: int,
        get_active_cooldown_state_fn: object,
        attempts_metadata_key: str,
        skipped_candidates_metadata_key: str,
        no_candidate_detail: str,
        log_label: str,
    ) -> Response:
        captured[alias_family] = services
        return Response(content="{}")

    monkeypatch.setattr(
        candidate_loop,
        "handle_alias_route",
        _capture_services,
    )
    # The Codex facade calls handle_alias_route via a from-import in
    # codex_auto_agent_route, so patch that module's local binding too.
    monkeypatch.setattr(
        codex_auto_agent_route,
        "handle_alias_route",
        _capture_services,
    )
    # The facade wrappers resolve these globals at call time.
    monkeypatch.setattr(
        lpe,
        "_resolve_aawm_alias_selection_enumeration",
        lambda request, alias_model, *, ingress, client_product_label=None: MagicMock(
            candidates=({},)
        ),
    )

    request = _signature_contract_request()
    # Exercise the actual facade wrappers that bind the service bundle.
    await lpe._handle_codex_auto_agent_alias_route(
        endpoint="/v1/responses",
        request=request,
        fastapi_response=MagicMock(spec=Response),
        user_api_key_dict=MagicMock(),
        prepared_request_body={"model": "seam-contract"},
        target_url="https://chatgpt.com/backend-api/codex/responses",
        api_key=None,
        forward_headers=True,
        canonical_alias="seam-contract",
    )
    await lpe._handle_anthropic_auto_agent_alias_route(
        endpoint="/v1/messages",
        request=request,
        fastapi_response=MagicMock(spec=Response),
        user_api_key_dict=MagicMock(),
        prepared_request_body={"model": "seam-contract"},
        target_url="https://api.anthropic.com/v1/messages",
        custom_headers={},
        canonical_alias="seam-contract",
    )

    field_names = {field.name for field in dataclasses.fields(AliasRouteServices)}
    missing_fields = set(_ALIAS_ROUTE_SERVICES_CALLBACK_FIELDS) - field_names
    assert not missing_fields, f"AliasRouteServices is missing typed callback fields: {missing_fields}"
    assert set(captured) == {"codex_auto_agent", "anthropic_auto_agent"}
    for alias_family, services in captured.items():
        di_targets = _DI_FORWARDING_TARGETS_BY_FAMILY[alias_family]
        # Capture every real production target up front, before any
        # monkeypatching, so Phase 2 always validates the genuine callable
        # even when a prior Phase 1 sentinel replaced the lpe global.
        real_targets: dict[str, Callable[..., object]] = {}
        for fn_name in _CALLBACK_PARAMETER_KINDS:
            if fn_name in di_targets:
                real_targets[fn_name] = getattr(lpe, di_targets[fn_name])
            else:
                real_targets[fn_name] = getattr(services, fn_name)
        for field_name, expected_parameters in _CALLBACK_PARAMETER_KINDS.items():
            callback = getattr(services, field_name)
            is_async = _CALLBACK_COROUTINE_STATUS[field_name]

            # --- Phase 1: behavioral DI forwarding validation ----------
            # Replace the underlying lpe global with a recording sentinel,
            # then invoke the captured DI wrapper and assert it forwards
            # positional/keyword args exactly and propagates the return
            # value (or awaitable for async targets).  A synchronous
            # black-hole ``(*args, **kwargs)`` wrapper that swallows the
            # call will fail here because the sentinel is never invoked.
            if field_name in di_targets:
                lpe_attr = di_targets[field_name]
                original_target = getattr(lpe, lpe_attr)
                sentinel_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
                sentinel_return = object()

                if is_async:

                    async def _async_sentinel(
                        *args: Any,
                        _calls: list = sentinel_calls,
                        _ret: object = sentinel_return,
                        **kwargs: Any,
                    ) -> object:
                        _calls.append((args, kwargs))
                        return _ret

                    monkeypatch.setattr(lpe, lpe_attr, _async_sentinel)
                else:

                    def _sync_sentinel(
                        *args: Any,
                        _calls: list = sentinel_calls,
                        _ret: object = sentinel_return,
                        **kwargs: Any,
                    ) -> object:
                        _calls.append((args, kwargs))
                        return _ret

                    monkeypatch.setattr(lpe, lpe_attr, _sync_sentinel)

                pos_a, pos_b = object(), object()
                kw_a, kw_b = object(), object()
                result = callback(pos_a, pos_b, _fwd_a=kw_a, _fwd_b=kw_b)

                if is_async:
                    assert inspect.isawaitable(result), (
                        f"{alias_family}.{field_name}: DI wrapper did not "
                        f"return an awaitable for async target {lpe_attr!r}"
                    )
                    result = await result

                assert result is sentinel_return, (
                    f"{alias_family}.{field_name}: DI wrapper did not "
                    f"propagate the return value of {lpe_attr!r}"
                )
                assert len(sentinel_calls) == 1, (
                    f"{alias_family}.{field_name}: DI wrapper did not "
                    f"forward to {lpe_attr!r} "
                    f"(sentinel called {len(sentinel_calls)} times)"
                )
                forwarded_args, forwarded_kwargs = sentinel_calls[0]
                assert forwarded_args == (pos_a, pos_b), (
                    f"{alias_family}.{field_name}: positional args not "
                    f"forwarded exactly: {forwarded_args!r}"
                )
                assert forwarded_kwargs == {"_fwd_a": kw_a, "_fwd_b": kw_b}, (
                    f"{alias_family}.{field_name}: keyword args not "
                    f"forwarded exactly: {forwarded_kwargs!r}"
                )

                # Restore the original target so subsequent iterations
                # and Phase 2 validation see the genuine callable.
                monkeypatch.setattr(lpe, lpe_attr, original_target)

            # --- Phase 2: real-target signature validation -------------
            # Validate the actual production target (not the DI wrapper)
            # against the expected parameter names, kinds, and coroutine
            # status.  An incompatible target (e.g. one requiring
            # ``wrong_required_name``) fails here.
            signature = inspect.signature(real_targets[field_name])
            actual_parameters = signature.parameters
            variadic_parameters = [
                parameter.name
                for parameter in actual_parameters.values()
                if parameter.kind
                in {
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                }
            ]
            assert not variadic_parameters, (
                f"{alias_family}.{field_name} must not declare variadic "
                f"parameters: {variadic_parameters}"
            )
            unexpected_required = [
                parameter.name
                for parameter in actual_parameters.values()
                if parameter.name not in expected_parameters
                and parameter.default is inspect.Parameter.empty
            ]
            assert not unexpected_required, (
                f"{alias_family}.{field_name} has unexpected required "
                f"parameters: {unexpected_required}"
            )
            assert tuple(actual_parameters) == tuple(expected_parameters), (
                f"{alias_family}.{field_name} parameters "
                f"{tuple(actual_parameters)} do not exactly match "
                f"{tuple(expected_parameters)}"
            )
            for parameter_name, expected_kind in expected_parameters.items():
                parameter = actual_parameters.get(parameter_name)
                assert parameter is not None, (
                    f"{alias_family}.{field_name} is missing {parameter_name!r}: "
                    f"{signature}"
                )
                assert parameter.kind is expected_kind, (
                    f"{alias_family}.{field_name}.{parameter_name} has "
                    f"{parameter.kind}, expected {expected_kind}"
                )
            assert (
                inspect.iscoroutinefunction(real_targets[field_name]) is is_async
            ), (
                f"{alias_family}.{field_name} coroutine status does not match "
                f"the production contract"
            )


async def _typed_select_candidate(
    *,
    request: Request,
    request_body: dict[str, Any],
) -> dict[str, Any]:
    return {}


async def _typed_perform_candidate_request(
    *,
    candidate: dict[str, Any],
    candidate_body: dict[str, Any],
) -> Response:
    return Response(content="{}")


def _typed_resolve_cooldown_publication(
    *,
    request: Optional[Request],
    candidate: dict[str, Any],
    lane_key: Optional[str],
    selected_cooldown_key: str,
    cooldown_seconds: float,
    error_class: Optional[str],
    grok_account_quota_exhausted: bool = False,
    kimi_failure_metadata: Optional[dict[str, Any]] = None,
    codex_failure_evidence_alias: Optional[str] = None,
) -> CooldownPublicationPlan:
    return CooldownPublicationPlan()


def _typed_publish_cooldown_memory(
    *,
    keys: Sequence[str],
    seconds: float,
    allow_ttl_shrink: bool = False,
) -> None:
    return None


async def _typed_persist_cooldown(
    *,
    keys: Sequence[str],
    seconds: float,
    allow_ttl_shrink: bool = False,
) -> None:
    return None


async def _typed_set_session_affinity(
    session_key: Optional[str],
    candidate: dict[str, Any],
) -> object:
    return None


def _typed_add_alias_metadata(
    request_body: dict[str, Any],
    *,
    request: Request,
    selection: dict[str, Any],
    attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    return request_body


def _typed_raise_redispatch(
    *,
    candidate: dict[str, Any],
    lane_key: Optional[str],
    cooldown_seconds: float,
    error_tokens: set[str],
    alias_model: str,
    error_class: str,
    cooldown_scope: Optional[str],
    error_status_code: Optional[int] = None,
    error_type: Optional[str] = None,
    error_code: Optional[str] = None,
    retry_after_seconds: Optional[float] = None,
    failure_phase: Optional[str] = None,
    attempted_provider_call: Optional[bool] = None,
    audit_events: Optional[list[Any]] = None,
    attempts: Optional[list[Any]] = None,
    skipped_candidates: Optional[list[Any]] = None,
) -> None:
    return None


def _build_typed_alias_route_services_fixture() -> AliasRouteServices:
    return AliasRouteServices(
        select_candidate_fn=_typed_select_candidate,
        perform_candidate_request_fn=_typed_perform_candidate_request,
        resolve_cooldown_publication_fn=_typed_resolve_cooldown_publication,
        publish_cooldown_memory_fn=_typed_publish_cooldown_memory,
        persist_cooldown_fn=_typed_persist_cooldown,
        set_session_affinity_fn=_typed_set_session_affinity,
        add_alias_metadata_fn=_typed_add_alias_metadata,
        raise_redispatch_fn=_typed_raise_redispatch,
    )


def test_alias_route_services_typed_assignment_fixture() -> None:
    services: AliasRouteServices = _build_typed_alias_route_services_fixture()
    assert services.publish_cooldown_memory_fn is _typed_publish_cooldown_memory


def test_candidate_selection_bridge_carries_reasoning_effort() -> None:
    """CFG-006: the typed bridge carries candidate reasoning_effort verbatim."""
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.interfaces import (
        CandidateSelection,
    )

    selection = CandidateSelection.from_legacy_dict(
        {
            "candidate": {
                "provider": "openai",
                "model": "gpt-5.6-luna",
                "route_family": "codex_responses",
                "last_resort": False,
                "reasoning_effort": "low",
            },
            "lane_key": "openai:gpt-5.6-luna",
            "cooldown_key": "cd:openai:gpt-5.6-luna",
        }
    )
    assert selection.candidate.reasoning_effort == "low"

    unset = CandidateSelection.from_legacy_dict(
        {
            "candidate": {
                "provider": "openrouter",
                "model": "openrouter/cohere/north-mini-code:free",
                "route_family": "codex_openrouter_completion_adapter",
            },
            "lane_key": "k",
            "cooldown_key": "ck",
        }
    )
    assert unset.candidate.reasoning_effort is None
