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
from typing import Any, Optional, Sequence
from unittest.mock import MagicMock

import pytest
from fastapi import Request
from starlette.responses import Response

from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    classification,
    config_compiler as compiler,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.interfaces import (
    AliasRouteServices,
    CooldownPublicationPlan,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
    alias_routing_state,
)

# The exact kwarg set passed at the loop's ``apply_cooldown_fn`` call site
# (``llm_passthrough_endpoints.py:22223-22233``). Encoded explicitly here so
# a future signature drift on either applicator is a named, readable failure
# rather than a silent ``**_kwargs`` swallow.
_APPLY_COOLDOWN_CALL_SITE_KWARGS = [
    "request",
    "candidate",
    "lane_key",
    "selected_cooldown_key",
    "cooldown_seconds",
    "error_class",
    "grok_account_quota_exhausted",
    "kimi_failure_metadata",
    "is_read_pilot_lane",
]

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


def _accepts_all_kwargs(fn: object, kwarg_names: list[str]) -> bool:
    signature = inspect.signature(fn)  # type: ignore[arg-type]
    parameters = signature.parameters
    has_var_keyword = any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
    if has_var_keyword:
        return True
    return all(name in parameters for name in kwarg_names)


def test_apply_cooldown_fn_call_site_kwargs_match_applicators() -> None:
    """Both production ``apply_cooldown_fn`` applicators accept the full call-site kwarg set."""
    assert _accepts_all_kwargs(
        lpe._set_codex_auto_agent_candidate_cooldowns,
        _APPLY_COOLDOWN_CALL_SITE_KWARGS,
    ), (
        "_set_codex_auto_agent_candidate_cooldowns no longer accepts every "
        f"kwarg the loop passes at its call site: {_APPLY_COOLDOWN_CALL_SITE_KWARGS}"
    )
    assert _accepts_all_kwargs(
        lpe._apply_anthropic_auto_agent_alias_cooldown,
        _APPLY_COOLDOWN_CALL_SITE_KWARGS,
    ), (
        "_apply_anthropic_auto_agent_alias_cooldown no longer accepts every "
        f"kwarg the loop passes at its call site: {_APPLY_COOLDOWN_CALL_SITE_KWARGS}"
    )


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
  - name: read
    candidates:
      - provider: openrouter
        model: openrouter/seam-contract-model
        route_family: codex_openrouter_completion_adapter
        priority: 900
"""
    snapshot = compiler.compile_yaml(raw_yaml)
    previous_snapshot = lpe.get_active_routing_snapshot()
    lpe.set_active_routing_snapshot(snapshot)
    session_key = "read:seam-contract-session:session:seam-contract-session"
    previous_affinity = lpe._codex_auto_agent_session_affinity_by_key.get(session_key)
    lpe._codex_auto_agent_session_affinity_by_key[session_key] = {
        "provider": "openrouter",
        "model": "openrouter/seam-contract-model",
        "route_family": "codex_openrouter_completion_adapter",
        "last_resort": False,
        "expires_at_monotonic": __import__("time").monotonic() + 3600.0,
    }
    try:
        from unittest.mock import MagicMock

        from fastapi import Request

        request = MagicMock(spec=Request)
        request.method = "POST"
        request.headers = {"session_id": "seam-contract-session"}
        request.query_params = {}
        request.state = MagicMock()
        request.state.aawm_alias_request_local_cooldown_until = {}
        request.state.aawm_alias_request_local_excluded_keys = set()

        selection = await lpe._select_codex_auto_agent_candidate(
            request=request,
            request_body={"model": "read", "previous_response_id": "resp_seam_contract"},
        )
    finally:
        lpe.set_active_routing_snapshot(previous_snapshot)
        if previous_affinity is None:
            lpe._codex_auto_agent_session_affinity_by_key.pop(session_key, None)
        else:
            lpe._codex_auto_agent_session_affinity_by_key[session_key] = previous_affinity

    assert selection.get("selection_reason") == "session_affinity"
    assert _SELECT_CANDIDATE_REQUIRED_KEYS <= set(selection.keys()), (
        "_select_codex_auto_agent_candidate no longer returns every key the "
        f"retry loop consumes: missing {_SELECT_CANDIDATE_REQUIRED_KEYS - set(selection.keys())}"
    )


def test_reset_alias_routing_state_for_tests_clears_everything() -> None:
    """RED until Wave-0 engineer adds ``reset_alias_routing_state_for_tests()``.

    Once added, the helper must clear: both family (codex/anthropic)
    cooldown/negative/affinity/evidence maps, ``candidate_probe_locks``, the
    read-pilot gate's ``_key_state`` + ``_family_state.evidence_events_by_key``,
    ``_round_robin_cursor_by_alias``, and the active routing snapshot (set to
    ``None``).
    """
    reset_fn = getattr(lpe, "reset_alias_routing_state_for_tests", None)
    assert reset_fn is not None, (
        "reset_alias_routing_state_for_tests() does not exist yet -- expected "
        "RED until the Wave-0 engineer lands it."
    )

    # Seed every piece of state the helper is required to clear.
    lpe._codex_auto_agent_cooldown_until_monotonic_by_key["seed"] = 1.0
    lpe._codex_auto_agent_cooldown_negative_until_monotonic_by_key["seed"] = 1.0
    lpe._codex_auto_agent_session_affinity_by_key["seed"] = {"provider": "p", "model": "m"}
    alias_routing_state.codex.evidence_events_by_key["seed"] = [1.0]
    lpe._anthropic_auto_agent_cooldown_until_monotonic_by_key["seed"] = 1.0
    lpe._anthropic_auto_agent_cooldown_negative_until_monotonic_by_key["seed"] = 1.0
    lpe._anthropic_auto_agent_session_affinity_by_key["seed"] = {"provider": "p", "model": "m"}
    alias_routing_state.anthropic.evidence_events_by_key["seed"] = [1.0]
    alias_routing_state.candidate_probe_locks["seed"] = asyncio.Lock()
    evidence = classification.classify_failure(
        status_code=429,
        provider="openrouter",
        message="rate limited",
    )
    lpe._read_pilot_cooldown_gate.record(
        cooldown_key="seed",
        event=evidence,
    )
    lpe._round_robin_cursor_by_alias["seed"] = 1

    raw_yaml = """
defaults: {}
aliases:
  - name: read
    candidates:
      - provider: openrouter
        model: openrouter/reset-helper-model
        route_family: codex_openrouter_completion_adapter
        priority: 900
"""
    lpe.set_active_routing_snapshot(compiler.compile_yaml(raw_yaml))

    reset_fn()

    assert lpe._codex_auto_agent_cooldown_until_monotonic_by_key == {}
    assert lpe._codex_auto_agent_cooldown_negative_until_monotonic_by_key == {}
    assert lpe._codex_auto_agent_session_affinity_by_key == {}
    assert alias_routing_state.codex.evidence_events_by_key == {}
    assert lpe._anthropic_auto_agent_cooldown_until_monotonic_by_key == {}
    assert lpe._anthropic_auto_agent_cooldown_negative_until_monotonic_by_key == {}
    assert lpe._anthropic_auto_agent_session_affinity_by_key == {}
    assert alias_routing_state.anthropic.evidence_events_by_key == {}
    assert alias_routing_state.candidate_probe_locks == {}
    assert lpe._read_pilot_cooldown_gate._key_state == {}
    assert lpe._read_pilot_cooldown_gate._family_state.evidence_events_by_key == {}
    assert lpe._round_robin_cursor_by_alias == {}
    assert lpe.get_active_routing_snapshot() is None


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
        "is_read_pilot_lane": inspect.Parameter.KEYWORD_ONLY,
    },
    "publish_cooldown_memory_fn": {
        "keys": inspect.Parameter.KEYWORD_ONLY,
        "seconds": inspect.Parameter.KEYWORD_ONLY,
    },
    "persist_cooldown_fn": {
        "keys": inspect.Parameter.KEYWORD_ONLY,
        "seconds": inspect.Parameter.KEYWORD_ONLY,
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
async def test_alias_route_services_signature_contracts(
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
        lpe._aawm_alias_candidate_loop,
        "handle_alias_route",
        _capture_services,
    )
    monkeypatch.setattr(
        lpe,
        "_resolve_aawm_alias_selection_enumeration",
        lambda request, alias_model, *, client_product_label=None: MagicMock(
            candidates=({},)
        ),
    )
    monkeypatch.setattr(
        lpe,
        "_get_anthropic_auto_agent_candidates_for_alias",
        lambda alias_model: ({},),
    )

    request = _signature_contract_request()
    await lpe._handle_codex_auto_agent_alias_route(
        endpoint="/v1/responses",
        request=request,
        fastapi_response=MagicMock(spec=Response),
        user_api_key_dict=MagicMock(),
        prepared_request_body={"model": "aawm-low"},
        target_url="https://chatgpt.com/backend-api/codex/responses",
        api_key=None,
        forward_headers=True,
    )
    await lpe._handle_anthropic_auto_agent_alias_route(
        endpoint="/v1/messages",
        request=request,
        fastapi_response=MagicMock(spec=Response),
        user_api_key_dict=MagicMock(),
        prepared_request_body={"model": "aawm-low-anthropic"},
        target_url="https://api.anthropic.com/v1/messages",
        custom_headers={},
    )

    field_names = {field.name for field in dataclasses.fields(AliasRouteServices)}
    missing_fields = set(_ALIAS_ROUTE_SERVICES_CALLBACK_FIELDS) - field_names
    assert not missing_fields, f"AliasRouteServices is missing typed callback fields: {missing_fields}"
    assert set(captured) == {"codex_auto_agent", "anthropic_auto_agent"}
    for alias_family, services in captured.items():
        for field_name, expected_parameters in _CALLBACK_PARAMETER_KINDS.items():
            callback = getattr(services, field_name)
            signature = inspect.signature(callback)
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
                inspect.iscoroutinefunction(callback)
                is _CALLBACK_COROUTINE_STATUS[field_name]
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
    is_read_pilot_lane: bool = False,
) -> CooldownPublicationPlan:
    return CooldownPublicationPlan()


def _typed_publish_cooldown_memory(
    *,
    keys: Sequence[str],
    seconds: float,
) -> None:
    return None


async def _typed_persist_cooldown(
    *,
    keys: Sequence[str],
    seconds: float,
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
