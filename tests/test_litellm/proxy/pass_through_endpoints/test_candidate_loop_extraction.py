"""Wave 2 structural pin: the alias candidate retry loop lives in the package.

Extends the ``test_rr054_structural_extraction.py`` AST-ownership discipline to
the Wave-2 candidate-loop extraction
(``.analysis/plan-godmodule-decomposition-r3-remediation-2026-07-23.md``):

- the loop body (``handle_alias_route``) is defined in
  ``aawm_alias_routing/candidate_loop.py`` and must NOT be re-defined as a
  function body in the god-module
- ``_handle_auto_agent_alias_route`` remains on the god-module (39 test files
  depend on the name) but is now a THIN façade that delegates to
  ``candidate_loop.handle_alias_route``
- the typed seam bundle (``AliasRouteServices``) is defined in
  ``aawm_alias_routing/interfaces.py`` and must not be re-defined on the
  god-module

Write-only surface: this file. No production edits.
"""

from __future__ import annotations

import asyncio
import ast
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

from fastapi import HTTPException
import pytest
from starlette.requests import Request

from litellm.proxy.pass_through_endpoints import aawm_alias_routing as package
from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import candidate_loop
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import session_affinity
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import snapshot_select
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_startup import (
    DEFAULT_CONFIG_DIR,
    compile_directory,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
    AliasRoutingStateManager,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.policy import (
    CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_ACCOUNT_QUOTA_COOLDOWN_KEY,
    CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_LANE_KEY,
    CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER,
    CODEX_AUTO_AGENT_ZAI_CODING_PLAN_PROVIDER,
)
from litellm.proxy._types import ProxyException

PACKAGE_DIR = Path(package.__file__).resolve().parent
GOD_PATH = Path(lpe.__file__).resolve()
CANDIDATE_LOOP_PATH = PACKAGE_DIR / "candidate_loop.py"
INTERFACES_PATH = PACKAGE_DIR / "interfaces.py"


class _StructuredWeeklyExhaustion(Exception):
    def __init__(self) -> None:
        super().__init__("Token Plan exhausted")
        self.status_code = 429
        self._aawm_provider_returned = True
        self.detail = {
            "error": {
                "type": "insufficient_quota",
                "code": "token_plan_quota_exhausted",
                "message": (
                    "Your token-plan 1-week quota has been exhausted. "
                    "The quota will reset at 08-27 12:04:00 UTC."
                ),
            }
        }


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _parse(path: Path) -> ast.AST:
    return ast.parse(_read(path), filename=str(path))


def _top_level_function_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    assert isinstance(tree, ast.Module)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
    return names


def _function_node(tree: ast.AST, name: str) -> ast.AST | None:
    assert isinstance(tree, ast.Module)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def _call_attr_names(fn_node: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(fn_node):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            names.add(func.id)
            continue
        if isinstance(func, ast.Attribute):
            parts: list[str] = []
            cur: ast.AST = func
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
            names.add(".".join(reversed(parts)))
    return names


def test_candidate_loop_body_is_defined_in_package() -> None:
    """``handle_alias_route`` is a real function body in ``candidate_loop.py``."""
    assert CANDIDATE_LOOP_PATH.is_file(), CANDIDATE_LOOP_PATH
    loop_source = _read(CANDIDATE_LOOP_PATH)
    tree = _parse(CANDIDATE_LOOP_PATH)
    assert "handle_alias_route" in _top_level_function_names(tree), "candidate_loop.py must define handle_alias_route"
    # Substance markers proving the R3-1 widened-lock body lives here.
    for marker in (
        "async def handle_alias_route",
        "candidate_probe_lock",
        "probe_lock.release()",
        "CooldownPublicationPlan",
    ):
        assert marker in loop_source, f"candidate_loop.py missing loop-body marker {marker!r}"


def test_god_module_does_not_define_loop_body() -> None:
    """The loop body must NOT be re-defined as a function in the god-module."""
    god_tree = _parse(GOD_PATH)
    god_fns = _top_level_function_names(god_tree)
    assert "handle_alias_route" not in god_fns, (
        "god-module re-defines the candidate loop body (handle_alias_route); "
        "it must delegate to aawm_alias_routing.candidate_loop instead"
    )


def test_god_facade_is_thin_delegate_to_candidate_loop() -> None:
    """``_handle_auto_agent_alias_route`` stays on the god-module but delegates."""
    god_source = _read(GOD_PATH)
    assert (
        "_handle_auto_agent_alias_route = partial(" in god_source
    ), "the legacy loop entrypoint name must be preserved (39 test files depend on it)"
    assert (
        "_aawm_anthropic_auto_agent_route.handle_auto_agent_alias_route"
        in god_source
    )
    assert "_aawm_alias_candidate_loop.handle_alias_route" in god_source, (
        "_handle_auto_agent_alias_route must delegate through the route runtime "
        "to candidate_loop.handle_alias_route"
    )


def test_alias_route_services_defined_in_interfaces_not_god() -> None:
    """``AliasRouteServices`` is owned by ``interfaces.py``, not the god-module."""
    assert INTERFACES_PATH.is_file(), INTERFACES_PATH
    iface_tree = _parse(INTERFACES_PATH)
    iface_names = {
        node.name
        for node in iface_tree.body  # type: ignore[union-attr]
        if isinstance(node, ast.ClassDef)
    }
    assert "AliasRouteServices" in iface_names, "interfaces.py must define AliasRouteServices"
    god_tree = _parse(GOD_PATH)
    god_classes = {
        node.name
        for node in god_tree.body  # type: ignore[union-attr]
        if isinstance(node, ast.ClassDef)
    }
    assert (
        "AliasRouteServices" not in god_classes
    ), "god-module must not redefine AliasRouteServices; it imports it from interfaces"


@pytest.mark.asyncio
async def test_candidate_loop_resolves_generic_status_helper_from_live_host() -> None:
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/responses",
            "headers": [],
            "query_string": b"",
            "server": ("testserver", 80),
            "client": ("testclient", 123),
            "scheme": "http",
        }
    )
    services = SimpleNamespace(
        select_candidate_fn=None,
        perform_candidate_request_fn=None,
        resolve_cooldown_publication_fn=None,
        publish_cooldown_memory_fn=None,
        persist_cooldown_fn=None,
        set_session_affinity_fn=lambda *args, **kwargs: None,
        add_alias_metadata_fn=lambda body, **kwargs: body,
        raise_redispatch_fn=None,
    )

    with pytest.raises(lpe.HTTPException) as exc_info:
        await candidate_loop.handle_alias_route(
            services,
            alias_family="codex",
            alias_model="aawm-test",
            request=request,
            prepared_request_body={},
            max_candidate_attempts=0,
            get_active_cooldown_state_fn=None,
            attempts_metadata_key="attempts",
            skipped_candidates_metadata_key="skipped",
            no_candidate_detail="no candidates",
            log_label="test",
        )

    assert exc_info.value.status_code == 429
    assert (
        lpe._extract_adapter_exception_status_code
        is package.error_signals._extract_adapter_exception_status_code
    )


@pytest.mark.asyncio
async def test_candidate_loop_records_in_flight_pinned_cooldown_without_no_candidate_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = SimpleNamespace(state=SimpleNamespace())
    body = {"model": "basic", "input": "hello", "stream": False}
    detail = {
        "error": {
            "type": "invalid_request_error",
            "code": "aawm_codex_auto_agent_in_flight_provider_cooling_down",
            "message": "pinned session target is cooling down",
        },
        "candidate": {
            "provider": "openai",
            "model": "gpt-5.5-codex",
            "route_family": "codex_responses",
            "lane_key": "codex-oauth:pinned",
            "cooldown_key": "openai:pinned",
        },
        "redispatch_required": True,
    }
    original_exc = HTTPException(
        status_code=429,
        detail=detail,
        headers={"Retry-After": "9"},
    )
    emitted: list[dict[str, Any]] = []
    persisted: list[list[dict[str, Any]]] = []

    async def _select(**_kwargs: Any) -> dict[str, Any]:
        raise original_exc

    async def _perform(**_kwargs: Any) -> object:
        raise AssertionError("pinned cooldown must stop before provider I/O")

    async def _noop_async(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(
        candidate_loop,
        "_session_affinity_mod",
        lambda: SimpleNamespace(
            is_replay_safe_session_owner_redispatch_body=lambda _body: False
        ),
    )
    monkeypatch.setattr(
        lpe,
        "_emit_auto_agent_alias_route_event",
        lambda event, **_kwargs: emitted.append(event),
    )
    monkeypatch.setattr(
        lpe,
        "_persist_auto_agent_alias_audit_only_events_best_effort",
        lambda events, *, request_body=None: persisted.append(events),
    )
    monkeypatch.setattr(
        lpe,
        "_emit_auto_agent_alias_no_candidate_event",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("pinned cooldown is not all-candidates exhaustion")
        ),
    )

    services = SimpleNamespace(
        select_candidate_fn=_select,
        perform_candidate_request_fn=_perform,
        resolve_cooldown_publication_fn=None,
        publish_cooldown_memory_fn=None,
        persist_cooldown_fn=None,
        set_session_affinity_fn=_noop_async,
        add_alias_metadata_fn=lambda request_body, **_kwargs: request_body,
        raise_redispatch_fn=None,
    )

    with pytest.raises(HTTPException) as caught:
        await candidate_loop.handle_alias_route(
            services,
            alias_family="codex_auto_agent",
            alias_model="basic",
            request=request,
            prepared_request_body=body,
            max_candidate_attempts=1,
            get_active_cooldown_state_fn=None,
            attempts_metadata_key="codex_auto_agent_attempts",
            skipped_candidates_metadata_key="codex_auto_agent_skipped_candidates",
            no_candidate_detail="no candidates",
            log_label="Codex",
        )

    assert caught.value is original_exc
    assert caught.value.detail is detail
    assert caught.value.headers == {"Retry-After": "9"}
    assert len(emitted) == 1
    event = emitted[0]
    assert event["event_type"] == "in_flight_pinned_session_cooldown"
    assert event["candidate_status"] == "pinned_session_cooldown"
    assert event["failure_phase"] == "session_affinity_cooldown"
    assert (
        event["error_code"]
        == "aawm_codex_auto_agent_in_flight_provider_cooling_down"
    )
    assert event["error_type"] == "invalid_request_error"
    assert event["error_status_code"] == 429
    assert event["attempted_provider_call"] is False
    assert event["redispatch_required"] is True
    assert event["fallback_result"] != "no_candidate_available"
    assert len(persisted) == 1
    assert persisted[0][-1] == event


@pytest.mark.asyncio
async def test_candidate_loop_other_redispatch_429_emits_no_terminal_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/openai_passthrough/v1/responses",
            "headers": [(b"user-agent", b"codex-cli/1.0")],
            "query_string": b"",
            "server": ("testserver", 80),
            "client": ("testclient", 123),
            "scheme": "http",
        }
    )
    body = {"model": "basic", "input": "hello", "stream": False}
    detail = {
        "error": {
            "type": "invalid_request_error",
            "code": "aawm_codex_auto_agent_redispatch_required",
            "message": "redispatch required",
        },
        "redispatch_required": True,
    }
    original_exc = HTTPException(
        status_code=429,
        detail=detail,
        headers={"Retry-After": "3"},
    )
    pinned_events: list[dict[str, Any]] = []
    no_candidate_events: list[dict[str, Any]] = []

    async def _select(**_kwargs: Any) -> dict[str, Any]:
        raise original_exc

    async def _perform(**_kwargs: Any) -> object:
        raise AssertionError("redispatch-required selection must stop before provider I/O")

    monkeypatch.setattr(
        candidate_loop,
        "_session_affinity_mod",
        lambda: SimpleNamespace(
            is_replay_safe_session_owner_redispatch_body=lambda _body: False
        ),
    )
    monkeypatch.setattr(
        lpe,
        "_emit_auto_agent_alias_pre_attempt_terminal_event",
        lambda **kwargs: pinned_events.append(kwargs),
    )
    monkeypatch.setattr(
        lpe,
        "_emit_auto_agent_alias_no_candidate_event",
        lambda **kwargs: no_candidate_events.append(kwargs),
    )

    services = SimpleNamespace(
        select_candidate_fn=_select,
        perform_candidate_request_fn=_perform,
        resolve_cooldown_publication_fn=None,
        publish_cooldown_memory_fn=None,
        persist_cooldown_fn=None,
        set_session_affinity_fn=None,
        add_alias_metadata_fn=None,
        raise_redispatch_fn=None,
    )

    with pytest.raises(HTTPException) as caught:
        await candidate_loop.handle_alias_route(
            services,
            alias_family="codex_auto_agent",
            alias_model="basic",
            request=request,
            prepared_request_body=body,
            max_candidate_attempts=1,
            get_active_cooldown_state_fn=None,
            attempts_metadata_key="codex_auto_agent_attempts",
            skipped_candidates_metadata_key="codex_auto_agent_skipped_candidates",
            no_candidate_detail="no candidates",
            log_label="Codex",
        )

    assert caught.value is original_exc
    assert caught.value.detail is detail
    assert caught.value.headers == {"Retry-After": "3"}
    assert pinned_events == []
    assert no_candidate_events == []


@pytest.mark.asyncio
async def test_candidate_loop_records_non_failover_admission_denial_before_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/openai_passthrough/v1/responses",
            "headers": [(b"user-agent", b"codex-cli/1.0")],
            "query_string": b"",
            "server": ("testserver", 80),
            "client": ("testclient", 123),
            "scheme": "http",
        }
    )
    body = {"model": "basic", "input": "hello", "stream": False}
    candidate = {
        "provider": "openai",
        "model": "gpt-5.5-codex",
        "route_family": "codex_responses",
        "codex_oauth_account_hash": "account-hash",
        "codex_oauth_lane_key": "codex-oauth:account",
    }
    selection = {
        "candidate": candidate,
        "cooldown_key": "openai:account",
        "lane_key": "codex-oauth:account",
        "selection_reason": "first_choice",
    }
    admission_decision = SimpleNamespace(
        allowed=False,
        reason="capacity_unavailable",
        detail_code="aawm_provider_lane_capacity_unavailable",
        lane_fingerprint="lane-fingerprint",
        provider="openai",
        account_hash="account-hash",
        limit_scope="concurrency",
        exhaustion_kind=None,
    )
    detail = {
        "error": {
            "type": "rate_limit_error",
            "code": "aawm_provider_lane_capacity_unavailable",
            "message": "lane capacity unavailable",
        }
    }
    original_exc = HTTPException(
        status_code=429,
        detail=detail,
        headers={"Retry-After": "4"},
    )
    emitted: list[dict[str, Any]] = []
    persisted: list[list[dict[str, Any]]] = []
    provider_calls: list[dict[str, Any]] = []

    async def _select(**_kwargs: Any) -> dict[str, Any]:
        return selection

    async def _perform(**kwargs: Any) -> object:
        provider_calls.append(kwargs)
        raise AssertionError("admission denial must stop before provider I/O")

    async def _noop_async(*_args: Any, **_kwargs: Any) -> None:
        return None

    class _Admission:
        async def admit_selected_candidate(self, **_kwargs: Any) -> object:
            return admission_decision

        def admission_deny_error_class(self, _decision: object) -> str:
            return "capacity_exhausted"

        def raise_provider_lane_admission_rejected(self, *_args: Any, **_kwargs: Any) -> None:
            raise original_exc

    monkeypatch.setattr(
        candidate_loop,
        "_session_affinity_mod",
        lambda: SimpleNamespace(
            is_replay_safe_session_owner_redispatch_body=lambda _body: False
        ),
    )
    monkeypatch.setattr(candidate_loop, "_admission_mod", lambda: _Admission())
    monkeypatch.setattr(
        lpe,
        "_plan_codex_oauth_account_failover",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        lpe,
        "_emit_auto_agent_alias_route_event",
        lambda event, **_kwargs: emitted.append(event),
    )
    monkeypatch.setattr(
        lpe,
        "_persist_auto_agent_alias_audit_only_events_best_effort",
        lambda events, *, request_body=None: persisted.append(events),
    )

    services = SimpleNamespace(
        select_candidate_fn=_select,
        perform_candidate_request_fn=_perform,
        resolve_cooldown_publication_fn=None,
        publish_cooldown_memory_fn=None,
        persist_cooldown_fn=None,
        set_session_affinity_fn=_noop_async,
        add_alias_metadata_fn=lambda request_body, **_kwargs: request_body,
        raise_redispatch_fn=None,
    )

    with pytest.raises(HTTPException) as caught:
        await candidate_loop.handle_alias_route(
            services,
            alias_family="codex_auto_agent",
            alias_model="basic",
            request=request,
            prepared_request_body=body,
            max_candidate_attempts=1,
            get_active_cooldown_state_fn=None,
            attempts_metadata_key="codex_auto_agent_attempts",
            skipped_candidates_metadata_key="codex_auto_agent_skipped_candidates",
            no_candidate_detail="no candidates",
            log_label="Codex",
        )

    assert caught.value is original_exc
    assert caught.value.detail is detail
    assert caught.value.headers == {"Retry-After": "4"}
    assert provider_calls == []
    assert len(emitted) == 1
    event = emitted[0]
    assert event["event_type"] == "provider_lane_admission_rejected"
    assert event["candidate_status"] == "admission_denied"
    assert event["failure_phase"] == "provider_lane_admission"
    assert event["error_code"] == "aawm_provider_lane_capacity_unavailable"
    assert event["failure_class"] == "capacity_exhausted"
    assert event["error_status_code"] == 429
    assert event["attempted_provider_call"] is False
    assert event["admission_reason"] == "capacity_unavailable"
    assert event["admission_detail_code"] == "aawm_provider_lane_capacity_unavailable"
    assert event["admission_lane_fingerprint"] == "lane-fingerprint"
    assert len(persisted) == 1
    assert persisted[0][-1] == event


def test_resolve_failure_plan_classifies_alibaba_model_not_found_as_candidate_unavailable() -> None:
    """Alibaba candidate + structured ModelNotFound -> ``candidate_unavailable``.

    Regression: ``_resolve_failure_plan`` must forward ``candidate`` to the
    retryable classifier. The Alibaba Token Plan unsupported-model guard needs
    trusted provider attribution from the failed candidate; without the
    forwarded candidate the structured ``ModelNotFound`` rejection is never
    classified and the plan loses ``candidate_unavailable``.
    """

    class _StructuredModelNotFound(Exception):
        def __init__(self) -> None:
            super().__init__("ModelNotFound")
            self.status_code = 404
            self._aawm_provider_returned = True
            self.detail = {
                "error": {
                    "type": "invalid_request_error",
                    "code": "ModelNotFound",
                    "message": "Model not exist.",
                }
            }

    exc = _StructuredModelNotFound()
    candidate = {
        "provider": CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER,
        "model": "alibaba_token_plan/qwen3.8-max-preview",
        "route_family": "codex_alibaba_token_plan_chat_completions_adapter",
    }
    captured: dict = {}

    def _capture_publication(**kwargs) -> object:
        captured.update(kwargs)
        return lpe._resolve_auto_agent_cooldown_publication_plan(**kwargs)

    def _fail_if_recording(**kwargs) -> None:
        raise AssertionError("codex failure evidence must not be recorded")

    plan = candidate_loop._resolve_failure_plan(
        resolve_cooldown_publication_fn=_capture_publication,
        record_codex_failure_evidence_fn=_fail_if_recording,
        request=SimpleNamespace(),
        candidate=candidate,
        selection={
            "cooldown_key": (
                "alibaba_token_plan:"
                "alibaba_token_plan/qwen3.8-max-preview:"
                "alibaba_token_plan"
            ),
            "lane_key": CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_LANE_KEY,
        },
        attempt_record={},
        exc=exc,
        codex_failure_evidence_alias=None,
        kimi_failure_metadata_fn=lambda exc, candidate=None: None,
        classify_kimi_fn=lambda metadata: None,
        classify_retryable_fn=package.error_signals._classify_codex_auto_agent_retryable_exhaustion,
        grok_quota_fn=lambda exc, candidate=None: False,
        cooldown_seconds_fn=lambda exc, candidate=None, attempted_provider_call=True: 60,
    )

    assert captured["error_class"] == "candidate_unavailable"
    assert plan.applied_scope == "candidate"
    assert plan.memory_keys == (
        "alibaba_token_plan:"
        "alibaba_token_plan/qwen3.8-max-preview:"
        "alibaba_token_plan",
    )
    assert plan.durable_keys == plan.memory_keys


def test_resolve_failure_plan_propagates_one_alibaba_ttl_to_plan_and_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Alibaba exhaustion resolves its TTL once, then reuses it for telemetry."""

    exc = _StructuredWeeklyExhaustion()
    candidate = {"provider": CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER}
    captured: dict = {}
    attempt_record: dict[str, Any] = {}
    monkeypatch.setattr(
        package.error_signals,
        "_resolve_alibaba_token_plan_exhaustion_cooldown_seconds",
        lambda: 8434.5,
    )

    def _capture_publication(**kwargs) -> object:
        captured.update(kwargs)
        return lpe._resolve_auto_agent_cooldown_publication_plan(**kwargs)

    ttl_calls: list[bool] = []

    def _cooldown_seconds(
        exc: Exception,
        *,
        candidate: Optional[dict[str, Any]] = None,
        attempted_provider_call: bool = True,
    ) -> float:
        ttl_calls.append(attempted_provider_call)
        return lpe._get_codex_auto_agent_cooldown_seconds(
            exc,
            candidate=candidate,
            attempted_provider_call=attempted_provider_call,
        )

    plan = candidate_loop._resolve_failure_plan(
        resolve_cooldown_publication_fn=_capture_publication,
        record_codex_failure_evidence_fn=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("confirmed exhaustion bypasses failure evidence")
        ),
        request=SimpleNamespace(),
        candidate=candidate,
        selection={
            "cooldown_key": "alibaba:selected",
            "lane_key": CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_LANE_KEY,
        },
        attempt_record=attempt_record,
        exc=exc,
        codex_failure_evidence_alias=None,
        kimi_failure_metadata_fn=lambda exc, candidate=None: None,
        classify_kimi_fn=lambda metadata: None,
        classify_retryable_fn=(
            lambda exc, candidate=None, attempted_provider_call=True: (
                lpe._classify_codex_auto_agent_retryable_exhaustion(
                    exc,
                    candidate=candidate,
                    attempted_provider_call=attempted_provider_call,
                )
            )
        ),
        grok_quota_fn=lambda exc, candidate=None: False,
        cooldown_seconds_fn=_cooldown_seconds,
    )

    assert (
        lpe._get_codex_auto_agent_candidate_cooldown_scope(
            captured["error_class"],
            candidate=candidate,
        )
        == "candidate"
    )
    assert captured["selected_cooldown_key"] == "alibaba:selected"
    assert plan.memory_keys == (
        CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_ACCOUNT_QUOTA_COOLDOWN_KEY,
    )
    assert plan.durable_keys == (
        CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_ACCOUNT_QUOTA_COOLDOWN_KEY,
    )
    assert plan.duration_seconds == 8434.5
    assert captured["cooldown_seconds"] == plan.duration_seconds
    assert ttl_calls == [True]
    assert attempt_record["cooldown_seconds"] == 8434.5
    assert attempt_record["cooldown_scope"] == "candidate"


def test_resolve_failure_plan_local_matching_alibaba_text_does_not_publish_account_ttl() -> None:
    """Unconfirmed matching text stays generic and cannot cool the account."""

    def _capture_publication(**kwargs) -> object:
        captured.update(kwargs)
        return lpe._resolve_auto_agent_cooldown_publication_plan(**kwargs)

    captured: dict[str, Any] = {}
    ttl_calls: list[bool] = []
    exc = _StructuredWeeklyExhaustion()
    exc.attempted_provider_call = False
    candidate = {"provider": CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER}

    def _cooldown_seconds(
        exc: Exception,
        *,
        candidate: Optional[dict[str, Any]] = None,
        attempted_provider_call: bool = True,
    ) -> float:
        ttl_calls.append(attempted_provider_call)
        return lpe._get_codex_auto_agent_cooldown_seconds(
            exc,
            candidate=candidate,
            attempted_provider_call=attempted_provider_call,
        )

    plan = candidate_loop._resolve_failure_plan(
        resolve_cooldown_publication_fn=_capture_publication,
        record_codex_failure_evidence_fn=lambda **kwargs: None,
        request=SimpleNamespace(),
        candidate=candidate,
        selection={"cooldown_key": "alibaba:selected", "lane_key": None},
        attempt_record={},
        exc=exc,
        codex_failure_evidence_alias=None,
        kimi_failure_metadata_fn=lambda exc, candidate=None: None,
        classify_kimi_fn=lambda metadata: None,
        classify_retryable_fn=lpe._classify_codex_auto_agent_retryable_exhaustion,
        grok_quota_fn=lambda exc, candidate=None: False,
        cooldown_seconds_fn=_cooldown_seconds,
    )

    assert captured["error_class"] == "rate_limited"
    assert captured["selected_cooldown_key"] == "alibaba:selected"
    assert ttl_calls == [False]
    assert plan.memory_keys == ("alibaba:selected",)
    assert plan.durable_keys == ("alibaba:selected",)
    assert plan.applied_scope == "candidate"
    assert plan.duration_seconds == 3 * 60 * 60.0


def test_resolve_failure_plan_local_http_exception_after_attempt_does_not_publish_account_ttl() -> None:
    """A local matching HTTPException cannot publish Alibaba account cooldown."""

    exc = HTTPException(
        status_code=429,
        detail={
            "error": {
                "type": "insufficient_quota",
                "code": "token_plan_quota_exhausted",
                "message": "Your five-hour token quota is exhausted.",
            }
        },
    )
    candidate = {"provider": CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER}
    attempt_record = {"attempted_provider_call": True}
    captured: dict[str, Any] = {}

    def _capture_publication(**kwargs) -> object:
        captured.update(kwargs)
        return lpe._resolve_auto_agent_cooldown_publication_plan(**kwargs)

    plan = candidate_loop._resolve_failure_plan(
        resolve_cooldown_publication_fn=_capture_publication,
        record_codex_failure_evidence_fn=lambda **kwargs: None,
        request=SimpleNamespace(),
        candidate=candidate,
        selection={"cooldown_key": "alibaba:selected", "lane_key": None},
        attempt_record=attempt_record,
        exc=exc,
        codex_failure_evidence_alias=None,
        kimi_failure_metadata_fn=lambda exc, candidate=None: None,
        classify_kimi_fn=lambda metadata: None,
        classify_retryable_fn=lpe._classify_codex_auto_agent_retryable_exhaustion,
        grok_quota_fn=lambda exc, candidate=None: False,
        cooldown_seconds_fn=lpe._get_codex_auto_agent_cooldown_seconds,
    )

    assert getattr(exc, "_aawm_provider_returned", False) is False
    assert captured["error_class"] == "rate_limited"
    assert plan.memory_keys == ("alibaba:selected",)
    assert plan.durable_keys == ("alibaba:selected",)
    assert (
        CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_ACCOUNT_QUOTA_COOLDOWN_KEY
        not in plan.memory_keys
    )
    assert (
        CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_ACCOUNT_QUOTA_COOLDOWN_KEY
        not in plan.durable_keys
    )
    assert attempt_record["attempted_provider_call"] is True


def test_resolve_failure_plan_unmarked_proxy_exception_after_attempt_does_not_publish_account_ttl() -> None:
    """ProxyException type alone is not Alibaba provider-returned evidence."""
    message = "Your five-hour token quota is exhausted."
    exc = ProxyException(
        message=message,
        type="rate_limit_error",
        param=None,
        code=429,
    )
    exc.status_code = 429
    exc.detail = {
        "error": {
            "type": "insufficient_quota",
            "code": "token_plan_quota_exhausted",
            "message": message,
        }
    }
    candidate = {"provider": CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER}
    attempt_record = {"attempted_provider_call": True}
    captured: dict[str, Any] = {}

    def _capture_publication(**kwargs) -> object:
        captured.update(kwargs)
        return lpe._resolve_auto_agent_cooldown_publication_plan(**kwargs)

    plan = candidate_loop._resolve_failure_plan(
        resolve_cooldown_publication_fn=_capture_publication,
        record_codex_failure_evidence_fn=lambda **kwargs: None,
        request=SimpleNamespace(),
        candidate=candidate,
        selection={"cooldown_key": "alibaba:selected", "lane_key": None},
        attempt_record=attempt_record,
        exc=exc,
        codex_failure_evidence_alias=None,
        kimi_failure_metadata_fn=lambda exc, candidate=None: None,
        classify_kimi_fn=lambda metadata: None,
        classify_retryable_fn=lpe._classify_codex_auto_agent_retryable_exhaustion,
        grok_quota_fn=lambda exc, candidate=None: False,
        cooldown_seconds_fn=lpe._get_codex_auto_agent_cooldown_seconds,
    )

    assert getattr(exc, "_aawm_provider_returned", False) is False
    assert captured["error_class"] == "rate_limited"
    assert plan.memory_keys == ("alibaba:selected",)
    assert plan.durable_keys == ("alibaba:selected",)
    assert (
        CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_ACCOUNT_QUOTA_COOLDOWN_KEY
        not in plan.memory_keys
    )
    assert (
        CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_ACCOUNT_QUOTA_COOLDOWN_KEY
        not in plan.durable_keys
    )
    assert attempt_record["attempted_provider_call"] is True


def test_resolve_failure_plan_passes_provider_attempt_evidence_to_classifier() -> None:
    """Classifier evidence is propagated from the failure attempt record."""
    classifier_calls: list[bool] = []

    def _classify(
        exc: Exception,
        *,
        candidate: Optional[dict[str, Any]] = None,
        attempted_provider_call: bool = True,
    ) -> Optional[str]:
        classifier_calls.append(attempted_provider_call)
        return "rate_limited"

    plan = candidate_loop._resolve_failure_plan(
        resolve_cooldown_publication_fn=lambda **kwargs: SimpleNamespace(
            applied_scope="candidate",
            duration_seconds=30.0,
            **kwargs,
        ),
        record_codex_failure_evidence_fn=lambda **kwargs: None,
        request=SimpleNamespace(),
        candidate={"provider": "openai"},
        selection={"cooldown_key": "openai:selected", "lane_key": None},
        attempt_record={"attempted_provider_call": False},
        exc=Exception("rate limited"),
        codex_failure_evidence_alias=None,
        kimi_failure_metadata_fn=lambda exc, candidate=None: None,
        classify_kimi_fn=lambda metadata: None,
        classify_retryable_fn=_classify,
        grok_quota_fn=lambda exc, candidate=None: False,
        cooldown_seconds_fn=lambda exc, candidate=None, attempted_provider_call=True: 30.0,
    )

    assert classifier_calls == [False]
    assert plan.applied_scope == "candidate"


@pytest.mark.parametrize("attempted_provider_call", [True, False])
def test_resolve_failure_plan_dispatcher_short_circuits_generic_classifier(
    attempted_provider_call: bool,
) -> None:
    matcher_calls: list[bool] = []
    generic_calls: list[bool] = []
    captured: dict[str, Any] = {}
    candidate = {"provider": "test-provider", "model": "future-model"}

    def _match(
        exc: Exception,
        *,
        candidate: Optional[dict[str, Any]] = None,
        attempted_provider_call: bool = True,
    ) -> SimpleNamespace:
        del exc
        assert candidate == {"provider": "test-provider", "model": "future-model"}
        matcher_calls.append(attempted_provider_call)
        return SimpleNamespace(error_class="candidate_unavailable")

    def _generic_classifier(*args: Any, **kwargs: Any) -> Optional[str]:
        del args
        generic_calls.append(bool(kwargs.get("attempted_provider_call")))
        raise AssertionError("generic classification must be short-circuited")

    def _capture_publication(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(applied_scope="candidate", duration_seconds=30.0, **kwargs)

    plan = candidate_loop._resolve_failure_plan(
        resolve_cooldown_publication_fn=_capture_publication,
        record_codex_failure_evidence_fn=lambda **kwargs: None,
        request=SimpleNamespace(),
        candidate=candidate,
        selection={"cooldown_key": "test:selected", "lane_key": None},
        attempt_record={"attempted_provider_call": attempted_provider_call},
        exc=Exception("provider model unavailable"),
        codex_failure_evidence_alias=None,
        kimi_failure_metadata_fn=lambda exc, candidate=None: None,
        classify_kimi_fn=lambda metadata: None,
        classify_retryable_fn=_generic_classifier,
        grok_quota_fn=lambda exc, candidate=None: False,
        cooldown_seconds_fn=lambda exc, candidate=None, attempted_provider_call=True: 30.0,
        match_provider_attributed_model_unavailable_fn=_match,
    )

    assert matcher_calls == [attempted_provider_call]
    assert generic_calls == []
    assert captured["error_class"] == "candidate_unavailable"
    assert plan.applied_scope == "candidate"


def test_resolve_failure_plan_classifies_coding_plan_1113_as_terminal_routing() -> None:
    class _InsufficientBalance(Exception):
        def __init__(self) -> None:
            super().__init__("Insufficient balance")
            self.status_code = 429
            self.detail = {
                "error": {
                    "type": "invalid_request_error",
                    "code": 1113,
                    "message": "Insufficient balance",
                }
            }

    exc = _InsufficientBalance()
    candidate = {
        "provider": CODEX_AUTO_AGENT_ZAI_CODING_PLAN_PROVIDER,
        "model": "zai_coding_plan/glm-5.3",
        "route_family": "codex_zai_coding_plan_chat_completions_adapter",
    }
    captured: dict = {}

    def _capture_publication(**kwargs) -> object:
        captured.update(kwargs)
        return lpe._resolve_auto_agent_cooldown_publication_plan(**kwargs)

    plan = candidate_loop._resolve_failure_plan(
        resolve_cooldown_publication_fn=_capture_publication,
        record_codex_failure_evidence_fn=lambda **kwargs: None,
        request=SimpleNamespace(),
        candidate=candidate,
        selection={
            "cooldown_key": f"{CODEX_AUTO_AGENT_ZAI_CODING_PLAN_PROVIDER}:default",
            "lane_key": None,
        },
        attempt_record={},
        exc=exc,
        codex_failure_evidence_alias=None,
        kimi_failure_metadata_fn=lambda exc, candidate=None: None,
        classify_kimi_fn=lambda metadata: None,
        classify_retryable_fn=package.error_signals._classify_codex_auto_agent_retryable_exhaustion,
        grok_quota_fn=lambda exc, candidate=None: False,
        cooldown_seconds_fn=lambda exc, candidate=None, attempted_provider_call=True: 60,
    )

    assert captured["error_class"] == "provider_terminal_error"
    assert plan.applied_scope == "candidate"


@pytest.mark.asyncio
async def test_candidate_loop_non_401_provider_terminal_error_does_not_rotate_account(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/openai_passthrough/v1/responses",
            "headers": [
                (b"user-agent", b"codex-cli/1.0"),
                (b"originator", b"codex_cli_rs"),
            ],
            "query_string": b"",
            "server": ("testserver", 80),
            "client": ("testclient", 123),
            "scheme": "http",
        }
    )
    body = {"model": "basic", "input": "hello", "stream": False}
    candidate = {
        "provider": "openai",
        "model": "gpt-5.3-codex",
        "route_family": "codex_responses",
        "codex_oauth_account_label": "account-test",
        "codex_oauth_account_hash": "hash-account-test",
        "codex_oauth_lane_key": "codex-oauth:test",
        "codex_oauth_credential_affinity": "interchangeable",
    }
    selection = {
        "candidate": candidate,
        "alias_model": "basic",
        "lane_key": "codex-oauth:test",
        "cooldown_key": "codex-oauth:test",
        "selection_reason": "first_choice",
        "skipped": [],
    }
    provider_calls: list[dict[str, Any]] = []
    rotation_calls: list[dict[str, Any]] = []

    class _ProviderTerminalError(HTTPException):
        def __init__(self) -> None:
            super().__init__(
                status_code=403,
                detail={
                    "error": {
                        "message": "provider rejected request",
                        "type": "invalid_request_error",
                        "code": "aawm_auto_agent_failed_responses_payload",
                    }
                },
            )

    async def _select(**_kwargs) -> dict[str, Any]:
        return selection

    async def _perform(
        *,
        candidate: dict[str, Any],
        candidate_body: dict[str, Any],
    ) -> object:
        provider_calls.append(
            {"candidate": dict(candidate), "body": dict(candidate_body)}
        )
        raise _ProviderTerminalError()

    def _plan_account_failover(_request, **kwargs) -> bool:
        rotation_calls.append(kwargs)
        return False

    async def _no_active_cooldown(_key: str) -> tuple[float, str]:
        return 0.0, "memory"

    async def _noop_async(*_args, **_kwargs) -> None:
        return None

    async def _execute_publication(
        *,
        plan,
        publish_cooldown_memory_fn,
        **_kwargs,
    ):
        publish_cooldown_memory_fn(
            keys=plan.memory_keys,
            seconds=plan.duration_seconds,
            allow_ttl_shrink=plan.allow_ttl_shrink,
        )
        return None

    def _publish_cooldown(*, keys, seconds, **_kwargs) -> None:
        for key in keys:
            request.state.__dict__.setdefault("_test_cooldowns", {})[key] = seconds

    class _Admission:
        async def admit_selected_candidate(self, **_kwargs):
            return SimpleNamespace(allowed=True, lease=None)

        async def release_provider_lane_admission(self, _lease):
            return None

    async def _ensure_session_owner_guard(**_kwargs):
        return SimpleNamespace(
            decision=SimpleNamespace(value="no_session"),
            reservation_token=None,
            held_reservation=False,
            provenance=None,
        )

    session_affinity = SimpleNamespace(
        is_replay_safe_session_owner_redispatch_body=lambda _body: False,
        resolve_canonical_session_identity=lambda *_args, **_kwargs: None,
        get_request_codex_auto_review_parent_session_identity=lambda _request: None,
        build_session_owner_attributes=lambda **_kwargs: {},
        ensure_session_owner_guard_for_request=_ensure_session_owner_guard,
        get_request_session_owner_lease=lambda _request: None,
        finalize_session_owner_lease_on_success=_noop_async,
        finalize_session_owner_lease_on_failure=_noop_async,
        get_session_owner_record=lambda **_kwargs: (None, None, None),
        request_has_effective_session_identity=lambda _request: False,
        build_session_owner_provenance=lambda **_kwargs: {},
        reset_released_request_session_owner_guard=lambda _request: False,
        SessionOwnerMutationOutcome=SimpleNamespace(
            CONFLICT="conflict",
            ERROR="error",
            NOT_HELD="not_held",
        ),
    )
    services = SimpleNamespace(
        select_candidate_fn=_select,
        perform_candidate_request_fn=_perform,
        resolve_cooldown_publication_fn=lpe._resolve_auto_agent_cooldown_publication_plan,
        publish_cooldown_memory_fn=_publish_cooldown,
        persist_cooldown_fn=_noop_async,
        set_session_affinity_fn=_noop_async,
        add_alias_metadata_fn=lpe._add_codex_auto_agent_alias_metadata,
        raise_redispatch_fn=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("non-401 terminal failure must not redispatch")
        ),
    )

    monkeypatch.setattr(
        candidate_loop, "_session_affinity_mod", lambda: session_affinity
    )
    monkeypatch.setattr(candidate_loop, "_admission_mod", lambda: _Admission())
    monkeypatch.setattr(
        lpe,
        "_codex_oauth_candidate_slot",
        lambda candidate: candidate["codex_oauth_account_hash"],
    )
    monkeypatch.setattr(
        lpe, "_plan_codex_oauth_account_failover", _plan_account_failover
    )
    monkeypatch.setattr(
        lpe,
        "_classify_codex_auto_agent_retryable_exhaustion",
        lambda exc, *, candidate=None, attempted_provider_call=True: "provider_terminal_error"
        if getattr(exc, "status_code", None) == 403
        else None,
    )
    monkeypatch.setattr(lpe, "_record_codex_failure_evidence", lambda **_kwargs: None)
    monkeypatch.setattr(
        lpe, "execute_cooldown_publication_transaction", _execute_publication
    )
    monkeypatch.setattr(
        lpe,
        "_persist_auto_agent_alias_audit_only_events_best_effort",
        lambda events, *, request_body=None: None,
    )
    monkeypatch.setattr(
        lpe, "_emit_auto_agent_alias_route_event", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "litellm.proxy.aawm_runtime_error_logging.persist_agent_terminal_error",
        lambda **_kwargs: True,
    )

    with pytest.raises(HTTPException) as caught:
        await candidate_loop.handle_alias_route(
            services,
            alias_family="codex_auto_agent",
            alias_model="basic",
            request=request,
            prepared_request_body=body,
            max_candidate_attempts=1,
            get_active_cooldown_state_fn=_no_active_cooldown,
            attempts_metadata_key="codex_auto_agent_attempts",
            skipped_candidates_metadata_key="codex_auto_agent_skipped_candidates",
            no_candidate_detail="no candidates",
            log_label="Codex",
        )

    assert len(provider_calls) == 1
    assert len(rotation_calls) == 1
    assert rotation_calls[0]["provider_status_code"] == 403
    assert caught.value.status_code == 502
    assert caught.value.detail["error"]["type"] == "provider_terminal_error"
    assert not hasattr(
        request.state, "aawm_codex_oauth_request_local_failover_context"
    )


def test_resolve_failure_plan_preserves_genuine_quota_transient_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}
    evidence_decision_calls: list[dict] = []

    def _capture_publication(**kwargs) -> object:
        captured.update(kwargs)
        return lpe._resolve_auto_agent_cooldown_publication_plan(**kwargs)

    def _current_decision(*, canonical_alias: str, cooldown_key: str):
        evidence_decision_calls.append(
            {
                "canonical_alias": canonical_alias,
                "cooldown_key": cooldown_key,
            }
        )
        return SimpleNamespace(should_cool=True, duration_seconds=30.0)

    monkeypatch.setattr(
        lpe._codex_failure_evidence_gate,
        "current_decision",
        _current_decision,
    )

    recorded: list[dict] = []

    def _record_failure_evidence(**kwargs) -> None:
        recorded.append(kwargs)

    class _GenuineQuotaTransientError(HTTPException):
        def __init__(self) -> None:
            super().__init__(
                status_code=429,
                detail={
                    "error": {
                        "code": "rate_limit_exceeded",
                        "message": "provider is temporarily rate limited",
                    }
                },
            )

    plan = candidate_loop._resolve_failure_plan(
        resolve_cooldown_publication_fn=_capture_publication,
        record_codex_failure_evidence_fn=_record_failure_evidence,
        request=SimpleNamespace(),
        candidate={"provider": "openai", "model": "supported", "route_family": "codex_responses"},
        selection={"cooldown_key": "openai:supported", "lane_key": "lane-b"},
        attempt_record={"attempted_provider_call": True},
        exc=_GenuineQuotaTransientError(),
        codex_failure_evidence_alias="work",
        kimi_failure_metadata_fn=lambda exc, candidate=None: None,
        classify_kimi_fn=lambda metadata: None,
        classify_retryable_fn=lambda exc, candidate=None, attempted_provider_call=True: "rate_limited",
        grok_quota_fn=lambda exc, candidate=None: True,
        cooldown_seconds_fn=lambda exc, candidate=None, attempted_provider_call=True: 30.0,
    )

    assert captured["error_class"] == "rate_limited"
    assert captured["grok_account_quota_exhausted"] is True
    assert evidence_decision_calls == [
        {"canonical_alias": "work", "cooldown_key": "openai:supported"}
    ]
    assert plan.applied_scope == "candidate"
    assert plan.memory_keys == ("openai:supported",)
    assert plan.durable_keys == ("openai:supported",)
    assert plan.duration_seconds == 30.0
    assert plan.grok_account_quota_exhausted is True
    assert plan.request_local_action is None
    assert len(recorded) == 1
    assert recorded[0]["cooldown_seconds"] is None


def test_classify_codex_fresh_auth_failure_accepts_fresh_generic_401() -> None:
    result = candidate_loop._classify_codex_fresh_auth_failure(
        HTTPException(status_code=401, detail="generic provider auth failure"),
        candidate={
            "provider": "openai",
            "model": "gpt-5.5",
            "route_family": "codex_responses",
        },
        selection={"cooldown_key": "openai:default"},
        is_codex_alias=True,
        has_continuation_state=False,
        has_previous_response_id=False,
        attempted_provider_call=True,
    )

    assert result == "provider_terminal_error"


@pytest.mark.parametrize(
    (
        "status_code",
        "detail",
        "has_continuation_state",
        "has_previous_response_id",
        "has_account_bound_state",
        "attempted_provider_call",
    ),
    [
        pytest.param(
            401,
            "generic provider auth failure",
            True,
            False,
            False,
            True,
            id="continuation-state",
        ),
        pytest.param(
            401,
            "generic provider auth failure",
            False,
            True,
            False,
            True,
            id="previous-response-id",
        ),
        pytest.param(
            401,
            "generic provider auth failure",
            False,
            False,
            True,
            True,
            id="account-bound-state",
        ),
        pytest.param(
            401,
            "generic provider auth failure",
            False,
            False,
            False,
            False,
            id="provider-call-not-attempted",
        ),
        pytest.param(
            403,
            "generic provider auth failure",
            False,
            False,
            False,
            True,
            id="non-401",
        ),
        pytest.param(
            401,
            {"error": {"code": "token_invalidated"}},
            False,
            False,
            False,
            True,
            id="direct-codex-token-invalidated",
        ),
    ],
)
def test_classify_codex_fresh_auth_failure_rejects_non_fresh_or_direct_token_invalidated(
    status_code: int,
    detail: object,
    has_continuation_state: bool,
    has_previous_response_id: bool,
    has_account_bound_state: bool,
    attempted_provider_call: bool,
) -> None:
    result = candidate_loop._classify_codex_fresh_auth_failure(
        HTTPException(status_code=status_code, detail=detail),
        candidate={
            "provider": "openai",
            "model": "gpt-5.5",
            "route_family": "codex_responses",
        },
        selection={
            "cooldown_key": "openai:default",
            "has_account_bound_state": has_account_bound_state,
        },
        is_codex_alias=True,
        has_continuation_state=has_continuation_state,
        has_previous_response_id=has_previous_response_id,
        attempted_provider_call=attempted_provider_call,
    )

    assert result is None


def test_resolve_failure_plan_uses_fresh_auth_after_normal_classifiers_return_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    normal_calls: list[str] = []

    def _normal_classifier(name: str):
        def _classify(*args, **kwargs):
            normal_calls.append(name)
            return None

        return _classify

    monkeypatch.setattr(
        candidate_loop,
        "_classify_codex_cohere_candidate_failure",
        _normal_classifier("cohere"),
    )
    monkeypatch.setattr(
        candidate_loop,
        "_classify_codex_zai_coding_plan_candidate_failure",
        _normal_classifier("zai"),
    )

    captured: dict = {}

    def _capture_publication(**kwargs) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    plan = candidate_loop._resolve_failure_plan(
        resolve_cooldown_publication_fn=_capture_publication,
        record_codex_failure_evidence_fn=lambda **kwargs: None,
        request=SimpleNamespace(),
        candidate={
            "provider": "openai",
            "model": "gpt-5.5",
            "route_family": "codex_responses",
        },
        selection={"cooldown_key": "openai:default", "lane_key": None},
        attempt_record={},
        exc=HTTPException(status_code=401, detail="generic provider auth failure"),
        codex_failure_evidence_alias="work",
        kimi_failure_metadata_fn=lambda exc, candidate=None: None,
        classify_kimi_fn=_normal_classifier("kimi"),
        classify_retryable_fn=_normal_classifier("retryable"),
        grok_quota_fn=lambda exc, candidate=None: False,
        cooldown_seconds_fn=lambda exc, candidate=None, attempted_provider_call=True: 60,
        fresh_codex_auth_error_class="provider_terminal_error",
    )

    assert normal_calls == ["kimi", "cohere", "zai", "retryable"]
    assert captured["error_class"] == "provider_terminal_error"
    assert plan.error_class == "provider_terminal_error"


class _IneligibleCandidateError(HTTPException):
    def __init__(self) -> None:
        super().__init__(
            status_code=503,
            detail={
                "error": {
                    "code": "aawm_codex_auto_agent_candidate_ineligible",
                }
            },
        )
        self.candidate_status = "ineligible"
        self.ineligibility_reason = "unsupported"


def test_resolve_failure_plan_ineligible_short_circuits_evidence_and_publication() -> None:
    captured: dict = {}

    def _capture_publication(**kwargs) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    plan = candidate_loop._resolve_failure_plan(
        resolve_cooldown_publication_fn=_capture_publication,
        record_codex_failure_evidence_fn=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("codex failure evidence must not be recorded")
        ),
        request=SimpleNamespace(),
        candidate={"provider": "openai", "model": "unsupported", "route_family": "codex_responses"},
        selection={"cooldown_key": "openai:unsupported", "lane_key": None},
        attempt_record={},
        exc=_IneligibleCandidateError(),
        codex_failure_evidence_alias="work",
        kimi_failure_metadata_fn=lambda exc, candidate=None: (_ for _ in ()).throw(
            AssertionError("classification must short-circuit")
        ),
        classify_kimi_fn=lambda metadata: (_ for _ in ()).throw(
            AssertionError("classification must short-circuit")
        ),
        classify_retryable_fn=lambda exc, candidate=None, attempted_provider_call=True: (_ for _ in ()).throw(
            AssertionError("classification must short-circuit")
        ),
        grok_quota_fn=lambda exc, candidate=None: (_ for _ in ()).throw(
            AssertionError("quota classification must not run")
        ),
        cooldown_seconds_fn=lambda exc, candidate=None, attempted_provider_call=True: (_ for _ in ()).throw(
            AssertionError("cooldown seconds must not run")
        ),
    )

    assert captured == {}
    assert plan.memory_keys == ()
    assert plan.durable_keys == ()
    assert plan.duration_seconds == 0.0
    assert plan.applied_scope == "none"
    assert plan.request_local_action is None
    assert plan.grok_account_quota_exhausted is False
    assert plan.kimi_failure_metadata is None
    assert plan.allow_ttl_shrink is False


@pytest.mark.asyncio
async def test_candidate_loop_skips_preflight_cooldown_without_recording_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/openai_passthrough/v1/responses",
            "headers": [(b"user-agent", b"codex-cli/1.0")],
            "query_string": b"",
            "server": ("testserver", 80),
            "client": ("testclient", 123),
            "scheme": "http",
        }
    )
    body = {"model": "basic", "input": "hello", "stream": False}
    selections = [
        {
            "candidate": {
                "provider": "openai",
                "model": "cooling-down",
                "route_family": "codex_responses",
            },
            "lane_key": "lane-cooling-down",
            "cooldown_key": "openai:cooling-down",
            "selection_reason": "first_available",
            "skipped": [],
        },
        {
            "candidate": {
                "provider": "openai",
                "model": "egress",
                "route_family": "codex_responses",
            },
            "lane_key": "lane-egress",
            "cooldown_key": "openai:egress",
            "selection_reason": "first_available",
            "skipped": [],
        },
    ]
    provider_calls: list[str] = []
    metadata_attempts: list[list[dict[str, Any]]] = []

    async def _select(**_kwargs: Any) -> dict[str, Any]:
        return selections.pop(0)

    async def _active_cooldown(key: str) -> tuple[float, str]:
        return (12.0, "memory") if key == "openai:cooling-down" else (0.0, "memory")

    async def _perform(
        *,
        candidate: dict[str, Any],
        candidate_body: dict[str, Any],
    ) -> object:
        provider_calls.append(candidate["model"])
        return {"candidate": candidate["model"], "body": candidate_body}

    async def _noop_async(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(candidate_loop, "alias_routing_state", AliasRoutingStateManager())

    def _add_alias_metadata(body: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        metadata_attempts.append(kwargs["attempts"])
        return body

    services = SimpleNamespace(
        select_candidate_fn=_select,
        perform_candidate_request_fn=_perform,
        resolve_cooldown_publication_fn=None,
        publish_cooldown_memory_fn=None,
        persist_cooldown_fn=None,
        set_session_affinity_fn=_noop_async,
        add_alias_metadata_fn=_add_alias_metadata,
        raise_redispatch_fn=None,
    )

    response = await candidate_loop.handle_alias_route(
        services,
        alias_family="codex_auto_agent",
        alias_model="basic",
        request=request,
        prepared_request_body=body,
        max_candidate_attempts=2,
        get_active_cooldown_state_fn=_active_cooldown,
        attempts_metadata_key="attempts",
        skipped_candidates_metadata_key="skipped",
        no_candidate_detail="no candidates",
        log_label="Codex",
    )

    assert response["candidate"] == "egress"
    assert provider_calls == ["egress"]
    assert metadata_attempts
    attempts = metadata_attempts[-1]
    assert all(captured is attempts for captured in metadata_attempts)
    assert [attempt["model"] for attempt in attempts] == ["egress"]
    assert attempts[0]["attempted_provider_call"] is True


@pytest.mark.asyncio
async def test_candidate_loop_metadata_failure_does_not_record_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/openai_passthrough/v1/responses",
            "headers": [(b"user-agent", b"codex-cli/1.0")],
            "query_string": b"",
            "server": ("testserver", 80),
            "client": ("testclient", 123),
            "scheme": "http",
        }
    )
    captured_attempts: list[list[dict[str, Any]]] = []
    provider_calls: list[str] = []

    async def _select(**_kwargs: Any) -> dict[str, Any]:
        return {
            "candidate": {
                "provider": "openai",
                "model": "metadata-failure",
                "route_family": "codex_responses",
            },
            "lane_key": "lane-metadata-failure",
            "cooldown_key": "openai:metadata-failure",
            "selection_reason": "first_available",
            "skipped": [],
        }

    async def _perform(
        *,
        candidate: dict[str, Any],
        candidate_body: dict[str, Any],
    ) -> object:
        provider_calls.append(candidate["model"])
        return candidate_body

    async def _no_active_cooldown(_key: str) -> tuple[float, str]:
        return 0.0, "memory"

    async def _noop_async(*_args: Any, **_kwargs: Any) -> None:
        return None

    def _fail_attempt_start(**kwargs: Any) -> dict[str, Any]:
        captured_attempts.append(kwargs["attempts"])
        raise asyncio.CancelledError

    monkeypatch.setattr(candidate_loop, "alias_routing_state", AliasRoutingStateManager())
    monkeypatch.setattr(
        lpe,
        "_record_auto_agent_alias_attempt_started",
        _fail_attempt_start,
    )
    services = SimpleNamespace(
        select_candidate_fn=_select,
        perform_candidate_request_fn=_perform,
        resolve_cooldown_publication_fn=None,
        publish_cooldown_memory_fn=None,
        persist_cooldown_fn=None,
        set_session_affinity_fn=_noop_async,
        add_alias_metadata_fn=lambda body, **_kwargs: body,
        raise_redispatch_fn=None,
    )

    with pytest.raises(asyncio.CancelledError):
        await candidate_loop.handle_alias_route(
            services,
            alias_family="codex_auto_agent",
            alias_model="basic",
            request=request,
            prepared_request_body={"model": "basic", "input": "hello"},
            max_candidate_attempts=1,
            get_active_cooldown_state_fn=_no_active_cooldown,
            attempts_metadata_key="attempts",
            skipped_candidates_metadata_key="skipped",
            no_candidate_detail="no candidates",
            log_label="Codex",
        )

    assert provider_calls == []
    assert captured_attempts == [[]]


@pytest.mark.asyncio
async def test_candidate_loop_lease_wrapper_failure_does_not_record_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/openai_passthrough/v1/responses",
            "headers": [(b"user-agent", b"codex-cli/1.0")],
            "query_string": b"",
            "server": ("testserver", 80),
            "client": ("testclient", 123),
            "scheme": "http",
        }
    )
    metadata_attempts: list[list[dict[str, Any]]] = []
    provider_calls: list[str] = []

    async def _select(**_kwargs: Any) -> dict[str, Any]:
        return {
            "candidate": {
                "provider": "openai",
                "model": "lease-failure",
                "route_family": "codex_responses",
            },
            "lane_key": "lane-lease-failure",
            "cooldown_key": "openai:lease-failure",
            "selection_reason": "first_available",
            "skipped": [],
        }

    async def _perform(
        *,
        candidate: dict[str, Any],
        candidate_body: dict[str, Any],
    ) -> object:
        provider_calls.append(candidate["model"])
        return candidate_body

    async def _no_active_cooldown(_key: str) -> tuple[float, str]:
        return 0.0, "memory"

    async def _noop_async(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def _fail_before_callback(
        _lease: object,
        _callback: object,
    ) -> object:
        raise asyncio.CancelledError

    def _add_alias_metadata(body: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        metadata_attempts.append(kwargs["attempts"])
        return body

    monkeypatch.setattr(candidate_loop, "alias_routing_state", AliasRoutingStateManager())
    monkeypatch.setattr(
        session_affinity,
        "run_with_session_owner_lease_renewal",
        _fail_before_callback,
    )
    services = SimpleNamespace(
        select_candidate_fn=_select,
        perform_candidate_request_fn=_perform,
        resolve_cooldown_publication_fn=None,
        publish_cooldown_memory_fn=None,
        persist_cooldown_fn=None,
        set_session_affinity_fn=_noop_async,
        add_alias_metadata_fn=_add_alias_metadata,
        raise_redispatch_fn=None,
    )

    with pytest.raises(asyncio.CancelledError):
        await candidate_loop.handle_alias_route(
            services,
            alias_family="codex_auto_agent",
            alias_model="basic",
            request=request,
            prepared_request_body={"model": "basic", "input": "hello"},
            max_candidate_attempts=1,
            get_active_cooldown_state_fn=_no_active_cooldown,
            attempts_metadata_key="attempts",
            skipped_candidates_metadata_key="skipped",
            no_candidate_detail="no candidates",
            log_label="Codex",
        )

    assert provider_calls == []
    assert metadata_attempts == [[]]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "alias_model",
    ("aawm-test", "codex-auto-review", "auto-review"),
)
async def test_candidate_loop_ineligible_falls_through_without_request_local_state(  # noqa: PLR0915
    monkeypatch: pytest.MonkeyPatch,
    alias_model: str,
) -> None:
    request = SimpleNamespace(state=SimpleNamespace())
    alias_routing_state = AliasRoutingStateManager()
    monkeypatch.setattr(
        candidate_loop,
        "alias_routing_state",
        alias_routing_state,
    )
    selections = [
        {
            "candidate": {
                "provider": "openai",
                "model": "unsupported",
                "route_family": "codex_responses",
            },
            "lane_key": "lane-a",
            "cooldown_key": "openai:unsupported",
            "selection_reason": "first_available",
        },
        {
            "candidate": {
                "provider": "openai",
                "model": "supported",
                "route_family": "codex_responses",
            },
            "lane_key": "lane-b",
            "cooldown_key": "openai:supported",
            "selection_reason": "first_available",
            "skipped": [],
        },
    ]
    provider_calls: list[str] = []
    provider_attempt_snapshots: list[list[tuple[str, bool]]] = []
    selection_kwargs: list[dict] = []
    replay_safety_results: list[
        session_affinity.SessionOwnerReplaySafetyResult
    ] = []
    original_classifier = (
        session_affinity.classify_session_owner_replay_safety_body
    )

    def _classify_replay_safety(
        request_body: dict,
    ) -> session_affinity.SessionOwnerReplaySafetyResult:
        result = original_classifier(request_body)
        replay_safety_results.append(result)
        return result

    monkeypatch.setattr(
        session_affinity,
        "classify_session_owner_replay_safety_body",
        _classify_replay_safety,
    )

    async def _perform_candidate(
        *,
        candidate: dict,
        candidate_body: dict,
    ) -> object:
        provider_calls.append(candidate["model"])
        provider_attempt_snapshots.append(
            [
                (attempt["model"], attempt["attempted_provider_call"])
                for attempt in request.state.aawm_alias_request_outcome["attempts"]
            ]
        )
        if candidate["model"] == "unsupported":
            raise _IneligibleCandidateError()
        return {"candidate": candidate["model"], "body": candidate_body}

    async def _select(**kwargs) -> dict:
        if not selections:
            raise AssertionError("selection called more than twice")
        selection_kwargs.append(kwargs)
        return selections.pop(0)

    async def _no_active_cooldown(_cooldown_key: str) -> tuple[float, str]:
        return 0.0, "memory"

    async def _set_session_affinity(*_args, **_kwargs) -> None:
        return None

    metadata_attempts: list[list[dict]] = []

    def _add_alias_metadata(body: dict, **kwargs) -> dict:
        metadata_attempts.append(kwargs["attempts"])
        return body

    services = SimpleNamespace(
        select_candidate_fn=_select,
        perform_candidate_request_fn=_perform_candidate,
        resolve_cooldown_publication_fn=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("ineligible path must not resolve publication")
        ),
        publish_cooldown_memory_fn=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("ineligible path must not publish cooldown memory")
        ),
        persist_cooldown_fn=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("ineligible path must not persist cooldown")
        ),
        set_session_affinity_fn=_set_session_affinity,
        add_alias_metadata_fn=_add_alias_metadata,
        raise_redispatch_fn=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("ineligible path must not signal redispatch")
        ),
    )

    def _fail_noop(_name: str):
        def _noop(*_args, **_kwargs):
            raise AssertionError(f"{_name} must not run")

        return _noop

    def _no_account_failover(*_args, **_kwargs) -> bool:
        return False

    for name in (
        "_record_codex_failure_evidence",
        "_exclude_codex_auto_agent_request_local_candidate_without_cooldown",
        "_apply_request_local_cooldown_from_plan",
    ):
        monkeypatch.setattr(lpe, name, _fail_noop(name))

    monkeypatch.setattr(
        lpe,
        "_plan_codex_oauth_account_failover",
        _no_account_failover,
    )
    prepared_request_body = (
        {
            "model": alias_model,
            "input": [
                {
                    "type": "reasoning",
                    "id": "rs_adapter_carried_reasoning",
                    "summary": None,
                    "encrypted_content": None,
                    "content": None,
                    "internal_chat_message_metadata_passthrough": {
                        "turn_id": "turn-adapter-item",
                    },
                }
            ],
        }
        if alias_model in {"codex-auto-review", "auto-review"}
        else {}
    )
    response = await candidate_loop.handle_alias_route(
        services,
        alias_family="codex",
        alias_model=alias_model,
        request=request,
        prepared_request_body=prepared_request_body,
        max_candidate_attempts=2,
        get_active_cooldown_state_fn=_no_active_cooldown,
        attempts_metadata_key="attempts",
        skipped_candidates_metadata_key="skipped",
        no_candidate_detail="no candidates",
        log_label="test",
    )

    assert response["candidate"] == "supported"
    assert provider_calls == ["unsupported", "supported"]
    assert provider_attempt_snapshots == [
        [("unsupported", True)],
        [("unsupported", True), ("supported", True)],
    ]
    assert selection_kwargs[0]["excluded_candidate_keys"] == frozenset()
    assert selection_kwargs[1]["excluded_candidate_keys"] == frozenset(
        {"openai:unsupported"}
    )
    assert len(replay_safety_results) == 1
    if alias_model in {"codex-auto-review", "auto-review"}:
        assert replay_safety_results[0].safe is True
        assert selection_kwargs[0]["_replay_safety"] is replay_safety_results[0]
        assert selection_kwargs[1]["_replay_safety"] is replay_safety_results[0]
    else:
        assert "_replay_safety" not in selection_kwargs[0]
        assert "_replay_safety" not in selection_kwargs[1]
    assert metadata_attempts
    assert all(captured is metadata_attempts[0] for captured in metadata_attempts)
    attempts = metadata_attempts[-1]
    assert len(attempts) == 2
    assert attempts[0]["status"] == "candidate_ineligible_no_cooldown"
    assert attempts[0]["error_class"] == "candidate_deterministically_ineligible"
    assert attempts[0]["cooldown_scope"] == "none"
    assert attempts[0]["candidate_status"] == "ineligible"
    assert attempts[0]["ineligibility_reason"] == "unsupported"
    assert "cooldown_seconds" not in attempts[0]
    assert getattr(
        request.state,
        "aawm_alias_request_local_excluded_keys",
        None,
    ) is None
    assert getattr(
        request.state,
        "aawm_alias_request_local_cooldown_until",
        None,
    ) is None


@pytest.mark.asyncio
async def test_candidate_loop_preserves_cursor_ineligible_terminal_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.llms.cursor_agent.connect import CursorConnectProtocolError
    from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime import (
        codex_candidate_calls,
    )

    request = SimpleNamespace(state=SimpleNamespace())
    candidate = {
        "provider": "cursor_agent",
        "model": "cursor_agent/cursor-grok-4.6-high",
        "route_family": "codex_cursor_agent_aiserver_adapter",
    }
    selection = {
        "candidate": candidate,
        "lane_key": "cursor_agent_cli",
        "cooldown_key": "cursor_agent:cursor-grok-4.6-high",
        "selection_reason": "first_available",
        "skipped": [],
    }
    with pytest.raises(ProxyException) as mapped_exc_info:
        codex_candidate_calls._raise_cursor_agent_alias_error(
            exc=CursorConnectProtocolError(
                "Cursor Agent requested unsupported local exec operation field 4."
            ),
            candidate=candidate,
        )
    mapped_exc = mapped_exc_info.value

    async def _select(**_kwargs: Any) -> dict[str, Any]:
        return selection

    async def _perform(**_kwargs: Any) -> object:
        raise mapped_exc

    async def _no_active_cooldown(_key: str) -> tuple[float, str]:
        return 0.0, "memory"

    async def _noop_async(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def _owner_guard(**_kwargs: Any) -> object:
        return SimpleNamespace(
            decision=SimpleNamespace(value="no_session"),
            reservation_token=None,
            held_reservation=False,
            provenance=None,
        )

    session_affinity_seam = SimpleNamespace(
        is_replay_safe_session_owner_redispatch_body=lambda _body: True,
        resolve_canonical_session_identity=lambda *_args, **_kwargs: None,
        get_request_codex_auto_review_parent_session_identity=lambda _request: None,
        build_session_owner_attributes=lambda **_kwargs: {},
        ensure_session_owner_guard_for_request=_owner_guard,
        get_request_session_owner_lease=lambda _request: None,
        finalize_session_owner_lease_on_success=_noop_async,
        finalize_session_owner_lease_on_failure=_noop_async,
    )

    class _Admission:
        async def admit_selected_candidate(self, **_kwargs: Any) -> object:
            return SimpleNamespace(allowed=True, lease=None)

        async def release_provider_lane_admission(self, _lease: object) -> None:
            return None

    failure_records: list[dict[str, Any]] = []
    terminal_events: list[dict[str, Any]] = []
    monkeypatch.setattr(candidate_loop, "alias_routing_state", AliasRoutingStateManager())
    monkeypatch.setattr(
        candidate_loop,
        "_session_affinity_mod",
        lambda: session_affinity_seam,
    )
    monkeypatch.setattr(candidate_loop, "_admission_mod", lambda: _Admission())
    monkeypatch.setattr(
        lpe,
        "_record_auto_agent_alias_attempt_started",
        lambda **kwargs: kwargs["prepared_request_body"],
    )
    monkeypatch.setattr(
        lpe,
        "_record_auto_agent_alias_attempt_failure",
        lambda **kwargs: failure_records.append(kwargs),
    )
    monkeypatch.setattr(
        lpe,
        "_emit_auto_agent_alias_no_candidate_event",
        lambda **kwargs: terminal_events.append(kwargs),
    )
    monkeypatch.setattr(
        lpe,
        "_plan_codex_oauth_account_failover",
        lambda *_args, **_kwargs: False,
    )

    services = SimpleNamespace(
        select_candidate_fn=_select,
        perform_candidate_request_fn=_perform,
        resolve_cooldown_publication_fn=lpe._resolve_auto_agent_cooldown_publication_plan,
        publish_cooldown_memory_fn=None,
        persist_cooldown_fn=None,
        set_session_affinity_fn=_noop_async,
        add_alias_metadata_fn=lambda body, **_kwargs: body,
        raise_redispatch_fn=None,
    )

    with pytest.raises(HTTPException) as caught:
        await candidate_loop.handle_alias_route(
            services,
            alias_family="codex_auto_agent",
            alias_model="work",
            request=request,
            prepared_request_body={"model": "work", "input": "dispatch basic"},
            max_candidate_attempts=1,
            get_active_cooldown_state_fn=_no_active_cooldown,
            attempts_metadata_key="codex_auto_agent_attempts",
            skipped_candidates_metadata_key="codex_auto_agent_skipped_candidates",
            no_candidate_detail="no candidates",
            log_label="Codex",
        )

    terminal_exc = caught.value
    assert terminal_exc.status_code == 400
    assert terminal_exc.detail is mapped_exc.detail
    assert mapped_exc.type == "invalid_request_error"
    assert (
        terminal_exc.detail["error"]["code"]
        == "aawm_codex_auto_agent_candidate_ineligible"
    )
    assert terminal_exc.candidate_status == "ineligible"
    assert terminal_exc.ineligibility_reason == "unsupported"
    assert terminal_exc.failure_phase == "candidate_preflight"
    assert terminal_exc.attempted_provider_call is True
    assert len(failure_records) == 1
    assert failure_records[0]["attempt_record"]["error_class"] == (
        "candidate_deterministically_ineligible"
    )
    assert len(terminal_events) == 1
    assert terminal_events[0]["exc"] is terminal_exc
    assert terminal_events[0]["exc"].status_code == 400
    assert terminal_events[0]["exc"].detail["error"]["code"] == (
        "aawm_codex_auto_agent_candidate_ineligible"
    )
    assert terminal_events[0]["exc"].candidate_status == "ineligible"
    assert terminal_events[0]["exc"].ineligibility_reason == "unsupported"
    assert terminal_events[0]["exc"].failure_phase == "candidate_preflight"
    assert terminal_events[0]["exc"].attempted_provider_call is True


@pytest.mark.asyncio
async def test_candidate_loop_fresh_cursor_ineligible_does_not_consume_attempt_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.llms.cursor_agent.connect import CursorConnectProtocolError
    from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime import (
        codex_candidate_calls,
    )

    cursor_candidate = {
        "provider": "cursor_agent",
        "model": "cursor_agent/cursor-grok-4.6-high",
        "route_family": "codex_cursor_agent_aiserver_adapter",
    }
    xai_candidate = {
        "provider": "xai",
        "model": "xai/grok-4.6",
        "route_family": "codex_responses",
    }
    selections = [
        {
            "candidate": cursor_candidate,
            "lane_key": "cursor_agent_cli",
            "cooldown_key": "cursor_agent:cursor-grok-4.6-high",
            "selection_reason": "first_available",
            "skipped": [],
        },
        {
            "candidate": xai_candidate,
            "lane_key": "xai",
            "cooldown_key": "xai:xai/grok-4.6",
            "selection_reason": "next_available",
            "skipped": [],
        },
    ]
    request = SimpleNamespace(state=SimpleNamespace())
    selection_calls: list[dict[str, Any]] = []
    provider_calls: list[str] = []
    failure_records: list[dict[str, Any]] = []
    with pytest.raises(ProxyException) as mapped_exc_info:
        codex_candidate_calls._raise_cursor_agent_alias_error(
            exc=CursorConnectProtocolError(
                "Cursor Agent requested unsupported local exec operation field 4."
            ),
            candidate=cursor_candidate,
        )
    mapped_exc = mapped_exc_info.value

    async def _select(**kwargs: Any) -> dict[str, Any]:
        selection_calls.append(kwargs)
        if not selections:
            raise AssertionError("candidate loop selected more than two candidates")
        return selections.pop(0)

    async def _perform(
        *,
        candidate: dict[str, Any],
        candidate_body: dict[str, Any],
    ) -> object:
        provider_calls.append(candidate["provider"])
        if candidate["provider"] == "cursor_agent":
            raise mapped_exc
        return {"candidate": candidate["model"], "body": candidate_body}

    async def _no_active_cooldown(_key: str) -> tuple[float, str]:
        return 0.0, "memory"

    async def _noop_async(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def _owner_guard(**_kwargs: Any) -> object:
        return SimpleNamespace(
            decision=SimpleNamespace(value="no_session"),
            reservation_token=None,
            held_reservation=False,
            provenance=None,
        )

    session_affinity_seam = SimpleNamespace(
        is_replay_safe_session_owner_redispatch_body=lambda _body: True,
        resolve_canonical_session_identity=lambda *_args, **_kwargs: None,
        get_request_codex_auto_review_parent_session_identity=lambda _request: None,
        build_session_owner_attributes=lambda **_kwargs: {},
        ensure_session_owner_guard_for_request=_owner_guard,
        get_request_session_owner_lease=lambda _request: None,
        finalize_session_owner_lease_on_success=_noop_async,
        finalize_session_owner_lease_on_failure=_noop_async,
    )

    class _Admission:
        async def admit_selected_candidate(self, **_kwargs: Any) -> object:
            return SimpleNamespace(allowed=True, lease=None)

        async def release_provider_lane_admission(self, _lease: object) -> None:
            return None

    monkeypatch.setattr(candidate_loop, "alias_routing_state", AliasRoutingStateManager())
    monkeypatch.setattr(
        candidate_loop,
        "_session_affinity_mod",
        lambda: session_affinity_seam,
    )
    monkeypatch.setattr(candidate_loop, "_admission_mod", lambda: _Admission())
    monkeypatch.setattr(
        lpe,
        "_record_auto_agent_alias_attempt_started",
        lambda **kwargs: kwargs["prepared_request_body"],
    )
    monkeypatch.setattr(
        lpe,
        "_record_auto_agent_alias_attempt_failure",
        lambda **kwargs: failure_records.append(kwargs),
    )
    def _fail_noop(_name: str):
        def _noop(*_args: Any, **_kwargs: Any):
            raise AssertionError(f"{_name} must not run")

        return _noop

    for name in (
        "_record_codex_failure_evidence",
        "_exclude_codex_auto_agent_request_local_candidate_without_cooldown",
        "_apply_request_local_cooldown_from_plan",
    ):
        monkeypatch.setattr(lpe, name, _fail_noop(name))
    monkeypatch.setattr(
        lpe,
        "_plan_codex_oauth_account_failover",
        lambda *_args, **_kwargs: False,
    )

    services = SimpleNamespace(
        select_candidate_fn=_select,
        perform_candidate_request_fn=_perform,
        resolve_cooldown_publication_fn=lpe._resolve_auto_agent_cooldown_publication_plan,
        publish_cooldown_memory_fn=None,
        persist_cooldown_fn=None,
        set_session_affinity_fn=_noop_async,
        add_alias_metadata_fn=lambda body, **_kwargs: body,
        raise_redispatch_fn=None,
    )

    response = await candidate_loop.handle_alias_route(
        services,
        alias_family="codex_auto_agent",
        alias_model="work",
        request=request,
        prepared_request_body={"model": "work", "input": "dispatch basic"},
        max_candidate_attempts=1,
        get_active_cooldown_state_fn=_no_active_cooldown,
        attempts_metadata_key="codex_auto_agent_attempts",
        skipped_candidates_metadata_key="codex_auto_agent_skipped_candidates",
        no_candidate_detail="no candidates",
        log_label="Codex",
    )

    assert response == {
        "candidate": xai_candidate["model"],
        "body": {"model": "work", "input": "dispatch basic"},
    }
    assert provider_calls == ["cursor_agent", "xai"]
    assert selection_calls[0]["excluded_candidate_keys"] == frozenset()
    assert selection_calls[1]["excluded_candidate_keys"] == frozenset(
        {"cursor_agent:cursor-grok-4.6-high"}
    )
    assert mapped_exc.type == "invalid_request_error"
    assert mapped_exc.attempted_provider_call is True
    assert len(failure_records) == 1
    assert failure_records[0]["attempt_record"]["error_class"] == (
        "candidate_deterministically_ineligible"
    )
    assert failure_records[0]["attempt_record"]["attempted_provider_call"] is True


@pytest.mark.asyncio
async def test_candidate_loop_cursor_session_continuation_is_session_scoped(  # noqa: PLR0915
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.llms.cursor_agent.connect import CursorConnectError
    from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime import (
        codex_candidate_calls,
    )

    candidate = {
        "provider": "cursor_agent",
        "model": "cursor_agent/cursor-grok-4.6-high",
        "route_family": "codex_cursor_agent_aiserver_adapter",
    }
    selection = {
        "candidate": candidate,
        "lane_key": "cursor_agent_cli",
        "cooldown_key": "cursor_agent:cursor-grok-4.6-high",
        "selection_reason": "first_available",
    }
    continuation_body = {
        "model": "work",
        "previous_response_id": "resp-unretained",
        "input": [
            {
                "type": "reasoning",
                "id": "rs_provider_owned",
            },
            {
                "type": "function_call_output",
                "call_id": "call-1",
                "output": "pwd output",
            }
        ],
    }
    fresh_body = {"model": "work", "input": "run pwd"}
    source_exc = CursorConnectError(
        "Cursor retained session unavailable",
        status_code=409,
    )
    setattr(
        source_exc,
        codex_candidate_calls._CURSOR_SESSION_CONTINUATION_FAILURE_MARKER,
        True,
    )
    with pytest.raises(ProxyException) as mapped_exc_info:
        codex_candidate_calls._raise_cursor_agent_alias_error(
            exc=source_exc,
            candidate=candidate,
        )
    mapped_exc = mapped_exc_info.value
    replay_safe_classifier = (
        session_affinity.is_replay_safe_session_owner_redispatch_body
    )
    assert replay_safe_classifier(continuation_body) is False
    stripped_continuation_body = dict(continuation_body)
    stripped_continuation_body.pop("previous_response_id")
    assert replay_safe_classifier(stripped_continuation_body) is False

    routing_state = AliasRoutingStateManager()
    monkeypatch.setattr(candidate_loop, "alias_routing_state", routing_state)
    selection_calls: list[dict[str, Any]] = []
    provider_calls: list[dict[str, Any]] = []
    failure_records: list[dict[str, Any]] = []
    terminal_events: list[dict[str, Any]] = []
    publication_calls: list[dict[str, Any]] = []
    evidence_calls: list[dict[str, Any]] = []

    async def _select(**kwargs: Any) -> dict[str, Any]:
        selection_calls.append(kwargs)
        return dict(selection)

    async def _perform(
        *,
        candidate: dict[str, Any],
        candidate_body: dict[str, Any],
    ) -> object:
        provider_calls.append(
            {"candidate": candidate, "body": candidate_body}
        )
        if "previous_response_id" in candidate_body:
            raise mapped_exc
        return {"candidate": candidate["model"]}

    async def _no_active_cooldown(_key: str) -> tuple[float, str]:
        return 0.0, "memory"

    async def _noop_async(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def _owner_guard(**_kwargs: Any) -> object:
        return SimpleNamespace(
            decision=SimpleNamespace(value="no_session"),
            reservation_token=None,
            held_reservation=False,
            provenance=None,
        )

    class _Admission:
        async def admit_selected_candidate(self, **_kwargs: Any) -> object:
            return SimpleNamespace(allowed=True, lease=None)

        async def release_provider_lane_admission(self, _lease: object) -> None:
            return None

    session_affinity_seam = SimpleNamespace(
        is_replay_safe_session_owner_redispatch_body=replay_safe_classifier,
        classify_session_owner_replay_safety_body=(
            session_affinity.classify_session_owner_replay_safety_body
        ),
        resolve_canonical_session_identity=lambda *_args, **_kwargs: None,
        get_request_codex_auto_review_parent_session_identity=lambda _request: None,
        build_session_owner_attributes=lambda **_kwargs: {},
        ensure_session_owner_guard_for_request=_owner_guard,
        get_request_session_owner_lease=lambda _request: None,
        finalize_session_owner_lease_on_success=_noop_async,
        finalize_session_owner_lease_on_failure=_noop_async,
        reset_released_request_session_owner_guard=lambda _request: False,
        SessionOwnerMutationOutcome=SimpleNamespace(
            CONFLICT="conflict",
            ERROR="error",
            NOT_HELD="not_held",
        ),
    )

    def _unsafe_rebuild(
        _body: dict[str, Any],
        *,
        continuation_exc: Exception,
        rejection_diagnostic_out: dict[str, Any],
    ) -> dict[str, Any]:
        _ = continuation_exc
        rejection_diagnostic_out.clear()
        return {
            "model": "work",
            "input": [
                {
                    "type": "reasoning",
                    "id": "rs_id_only",
                }
            ],
        }

    def _record_failure(**kwargs: Any) -> None:
        failure_records.append(kwargs)

    async def _capture_publication(**kwargs: Any) -> None:
        publication_calls.append(kwargs)

    monkeypatch.setattr(
        candidate_loop,
        "_session_affinity_mod",
        lambda: session_affinity_seam,
    )
    monkeypatch.setattr(candidate_loop, "_admission_mod", lambda: _Admission())
    monkeypatch.setattr(
        codex_candidate_calls,
        "_build_cursor_replay_safe_fresh_dispatch_body",
        _unsafe_rebuild,
    )
    monkeypatch.setattr(
        lpe,
        "_record_auto_agent_alias_attempt_failure",
        _record_failure,
    )
    monkeypatch.setattr(
        lpe,
        "_emit_auto_agent_alias_no_candidate_event",
        lambda **kwargs: terminal_events.append(kwargs),
    )
    monkeypatch.setattr(
        lpe,
        "execute_cooldown_publication_transaction",
        _capture_publication,
    )
    monkeypatch.setattr(
        lpe,
        "_record_codex_failure_evidence",
        lambda **kwargs: evidence_calls.append(kwargs),
    )
    monkeypatch.setattr(
        lpe,
        "_exclude_codex_auto_agent_request_local_candidate_without_cooldown",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("session-scoped failure must not exclude the candidate")
        ),
    )
    monkeypatch.setattr(
        lpe,
        "_apply_request_local_cooldown_from_plan",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("session-scoped failure must not apply request-local state")
        ),
    )

    with pytest.raises(HTTPException) as caught:
        await candidate_loop.handle_alias_route(
            SimpleNamespace(
                select_candidate_fn=_select,
                perform_candidate_request_fn=_perform,
                resolve_cooldown_publication_fn=lpe._resolve_auto_agent_cooldown_publication_plan,
                publish_cooldown_memory_fn=_capture_publication,
                persist_cooldown_fn=_capture_publication,
                set_session_affinity_fn=_noop_async,
                add_alias_metadata_fn=lambda body, **_kwargs: body,
                raise_redispatch_fn=lambda **_kwargs: (_ for _ in ()).throw(
                    AssertionError("session-scoped failure must not redispatch")
                ),
            ),
            alias_family="codex_auto_agent",
            alias_model="work",
            request=SimpleNamespace(headers={}, state=SimpleNamespace()),
            prepared_request_body=continuation_body,
            max_candidate_attempts=1,
            get_active_cooldown_state_fn=_no_active_cooldown,
            attempts_metadata_key="attempts",
            skipped_candidates_metadata_key="skipped",
            no_candidate_detail="no candidates",
            log_label="Codex",
        )

    assert caught.value.status_code == 409
    assert caught.value.detail["error"]["code"] == (
        "aawm_codex_auto_agent_candidate_ineligible"
    )
    assert caught.value.candidate_status == "ineligible"
    assert caught.value.ineligibility_reason == "preflight_skipped"
    assert caught.value.failure_phase == "cursor_session_continuation"
    assert caught.value.attempted_provider_call is False
    assert len(provider_calls) == 1
    assert len(failure_records) == 1
    rejection_field = codex_candidate_calls._CURSOR_REPLAY_FRESH_DISPATCH_REJECT_FIELD
    expected_rejection = {
        "stage": "rebuilt_body_replay_unsafe",
        "reason": "id_only_reasoning_reference",
    }
    assert terminal_events[0]["extra_fields"][rejection_field] == expected_rejection
    assert "aawm_passthrough_request_shape_summary" in (
        terminal_events[0]["extra_fields"]
    )
    assert failure_records[0]["attempt_record"][rejection_field] == (
        expected_rejection
    )
    assert failure_records[0]["error_class"] == "continuation_state_unavailable"
    attempt_record = failure_records[0]["attempt_record"]
    assert attempt_record["status"] == "cooldown_set"
    assert attempt_record["cooldown_scope"] == "candidate"
    assert attempt_record["cooldown_seconds"] == 300.0
    assert attempt_record["attempted_provider_call"] is False
    assert attempt_record["failure_phase"] == "cursor_session_continuation"
    assert attempt_record["source_error"]
    assert evidence_calls[0]["cooldown_key"] == selection["cooldown_key"]
    assert len(publication_calls) == 1
    publication_plan = publication_calls[0]["plan"]
    assert publication_plan.applied_scope == "candidate"
    assert publication_plan.duration_seconds == 300.0
    assert publication_plan.memory_keys == (selection["cooldown_key"],)
    assert publication_plan.durable_keys == (selection["cooldown_key"],)
    assert terminal_events and terminal_events[0]["exc"].status_code == 409
    assert routing_state.codex.cooldown_until_monotonic_by_key == {}
    assert routing_state.codex.candidate_semantic_ineligibility_by_key == {}
    assert selection_calls[0]["excluded_candidate_keys"] == frozenset()

    fresh_response = await candidate_loop.handle_alias_route(
        SimpleNamespace(
            select_candidate_fn=_select,
            perform_candidate_request_fn=_perform,
            resolve_cooldown_publication_fn=lpe._resolve_auto_agent_cooldown_publication_plan,
            publish_cooldown_memory_fn=_capture_publication,
            persist_cooldown_fn=_capture_publication,
            set_session_affinity_fn=_noop_async,
            add_alias_metadata_fn=lambda body, **_kwargs: body,
            raise_redispatch_fn=None,
        ),
        alias_family="codex_auto_agent",
        alias_model="work",
        request=SimpleNamespace(headers={}, state=SimpleNamespace()),
        prepared_request_body=fresh_body,
        max_candidate_attempts=1,
        get_active_cooldown_state_fn=_no_active_cooldown,
        attempts_metadata_key="attempts",
        skipped_candidates_metadata_key="skipped",
        no_candidate_detail="no candidates",
        log_label="Codex",
    )

    assert fresh_response == {"candidate": candidate["model"]}
    assert len(provider_calls) == 2
    assert selection_calls[1]["excluded_candidate_keys"] == frozenset()


@pytest.mark.asyncio
async def test_candidate_loop_cursor_full_history_continuation_uses_fresh_next_candidate(  # noqa: PLR0915
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.llms.cursor_agent.connect import CursorConnectError
    from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime import (
        codex_candidate_calls,
    )

    cursor_candidate = {
        "provider": "cursor_agent",
        "model": "cursor_agent/cursor-grok-4.6-high",
        "route_family": "codex_cursor_agent_aiserver_adapter",
    }
    fallback_candidate = {
        "provider": "openrouter",
        "model": "openrouter/fallback",
        "route_family": "codex_openrouter_responses_adapter",
    }
    replay_messages = [
        {
            "role": "user",
            "content": "Complete the original assignment in /workspace.",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "pwd-call",
                    "type": "function",
                    "function": {
                        "name": "exec_command",
                        "arguments": '{"cmd":"pwd","workdir":"/workspace"}',
                    },
                }
            ],
        },
    ]
    replay_tools = [
        {
            "type": "function",
            "function": {"name": "exec_command"},
        }
    ]
    selections = [
        {
            "candidate": cursor_candidate,
            "lane_key": "cursor_agent_cli",
            "cooldown_key": "cursor_agent:cursor-grok-4.6-high",
            "selection_reason": "first_available",
            "has_account_bound_state": True,
            "in_flight_session": True,
        },
        {
            "candidate": fallback_candidate,
            "lane_key": "openrouter",
            "cooldown_key": "openrouter:fallback",
            "selection_reason": "next_available",
            "has_account_bound_state": True,
            "in_flight_session": True,
        },
    ]
    prepared_body = {
        "model": "work",
        "previous_response_id": "cursor-unretained",
        "tools": replay_tools,
        "input": [
            {
                "type": "function_call_output",
                "call_id": "pwd-call",
                "output": "/workspace",
            },
        ],
    }
    codex_candidate_calls._store_cursor_replay_state(
        "cursor-unretained",
        messages=replay_messages,
        tools=replay_tools,
    )
    replay_state = codex_candidate_calls._peek_cursor_replay_state(
        "cursor-unretained"
    )
    with pytest.raises(CursorConnectError) as source_exc_info:
        codex_candidate_calls._raise_cursor_session_continuation_unavailable(
            previous_response_id="cursor-unretained",
            replay_state=replay_state,
        )
    source_exc = source_exc_info.value
    with pytest.raises(ProxyException) as mapped_exc_info:
        codex_candidate_calls._raise_cursor_agent_alias_error(
            exc=source_exc,
            candidate=cursor_candidate,
        )
    mapped_exc = mapped_exc_info.value
    candidate_bodies: list[dict[str, Any]] = []
    metadata_input_bodies: list[tuple[str, dict[str, Any]]] = []
    owner_guard_bodies: list[dict[str, Any]] = []
    rebuilt_request_body = (
        codex_candidate_calls._build_cursor_replay_safe_fresh_dispatch_body(
            prepared_body,
            continuation_exc=mapped_exc,
        )
    )
    assert rebuilt_request_body is not None
    replay_safe_classifier = (
        session_affinity.is_replay_safe_session_owner_redispatch_body
    )
    assert replay_safe_classifier(rebuilt_request_body) is True

    routing_state = AliasRoutingStateManager()
    monkeypatch.setattr(candidate_loop, "alias_routing_state", routing_state)
    selection_calls: list[dict[str, Any]] = []
    selected_candidates: list[dict[str, Any]] = []
    provider_calls: list[str] = []
    classifier_calls: list[dict[str, Any]] = []
    publication_calls: list[dict[str, Any]] = []

    async def _select(**kwargs: Any) -> dict[str, Any]:
        selection_calls.append(kwargs)
        if not selections:
            raise AssertionError("candidate loop selected more than two candidates")
        selected = dict(selections.pop(0))
        selected_candidates.append(selected)
        return selected

    async def _perform(
        *,
        candidate: dict[str, Any],
        candidate_body: dict[str, Any],
    ) -> object:
        assert candidate_body["model"] == candidate["model"]
        candidate_bodies.append(candidate_body)
        provider_calls.append(str(candidate["provider"]))
        if candidate["provider"] == "cursor_agent":
            assert candidate_body["previous_response_id"] == "cursor-unretained"
            raise mapped_exc
        assert "previous_response_id" not in candidate_body
        assert candidate_body["input"] == rebuilt_request_body["input"]
        assert candidate_body["tools"] == replay_tools
        assert replay_safe_classifier(candidate_body) is True
        return {"candidate": candidate["model"]}

    async def _no_active_cooldown(_key: str) -> tuple[float, str]:
        return 0.0, "memory"

    async def _noop_async(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def _owner_guard(**kwargs: Any) -> object:
        owner_guard_bodies.append(kwargs["request_body"])
        return SimpleNamespace(
            decision=SimpleNamespace(value="no_session"),
            reservation_token=None,
            held_reservation=False,
            provenance=None,
        )

    class _Admission:
        async def admit_selected_candidate(self, **_kwargs: Any) -> object:
            return SimpleNamespace(allowed=True, lease=None)

        async def release_provider_lane_admission(self, _lease: object) -> None:
            return None

    def _classify_rebuilt_request(body: dict[str, Any]) -> bool:
        classifier_calls.append(body)
        return replay_safe_classifier(body)

    def _add_candidate_metadata(
        body: dict[str, Any],
        *,
        selection: dict[str, Any],
        **_kwargs: Any,
    ) -> dict[str, Any]:
        metadata_input_bodies.append(
            (str(selection["candidate"]["provider"]), body)
        )
        rebuilt = dict(body)
        rebuilt["model"] = selection["candidate"]["model"]
        return rebuilt

    session_affinity_seam = SimpleNamespace(
        is_replay_safe_session_owner_redispatch_body=_classify_rebuilt_request,
        resolve_canonical_session_identity=lambda *_args, **_kwargs: None,
        get_request_codex_auto_review_parent_session_identity=lambda _request: None,
        build_session_owner_attributes=lambda **_kwargs: {},
        ensure_session_owner_guard_for_request=_owner_guard,
        get_request_session_owner_lease=lambda _request: None,
        finalize_session_owner_lease_on_success=_noop_async,
        finalize_session_owner_lease_on_failure=_noop_async,
        reset_released_request_session_owner_guard=lambda _request: False,
        SessionOwnerMutationOutcome=SimpleNamespace(
            CONFLICT="conflict",
            ERROR="error",
            NOT_HELD="not_held",
        ),
    )

    async def _capture_publication(**kwargs: Any) -> None:
        publication_calls.append(kwargs)

    monkeypatch.setattr(
        candidate_loop,
        "_session_affinity_mod",
        lambda: session_affinity_seam,
    )
    monkeypatch.setattr(candidate_loop, "_admission_mod", lambda: _Admission())
    monkeypatch.setattr(
        lpe,
        "execute_cooldown_publication_transaction",
        _capture_publication,
    )

    response = await candidate_loop.handle_alias_route(
        SimpleNamespace(
            select_candidate_fn=_select,
            perform_candidate_request_fn=_perform,
            resolve_cooldown_publication_fn=lpe._resolve_auto_agent_cooldown_publication_plan,
            publish_cooldown_memory_fn=_capture_publication,
            persist_cooldown_fn=_capture_publication,
            set_session_affinity_fn=_noop_async,
            add_alias_metadata_fn=_add_candidate_metadata,
            raise_redispatch_fn=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("self-contained continuation must try the next candidate")
            ),
        ),
        alias_family="codex_auto_agent",
        alias_model="work",
        request=SimpleNamespace(state=SimpleNamespace()),
        prepared_request_body=prepared_body,
        max_candidate_attempts=2,
        get_active_cooldown_state_fn=_no_active_cooldown,
        attempts_metadata_key="attempts",
        skipped_candidates_metadata_key="skipped",
        no_candidate_detail="no candidates",
        log_label="Codex",
    )

    assert response == {"candidate": fallback_candidate["model"]}
    assert provider_calls == ["cursor_agent", "openrouter"]
    assert selection_calls[0]["excluded_candidate_keys"] == frozenset()
    assert selection_calls[1]["excluded_candidate_keys"] == frozenset(
        {"cursor_agent:cursor-grok-4.6-high"}
    )
    assert selection_calls[0]["request_body"] is prepared_body
    assert selection_calls[1]["request_body"] is classifier_calls[1]
    assert all(
        selected["has_account_bound_state"] is True
        and selected["in_flight_session"] is True
        for selected in selected_candidates
    )
    assert classifier_calls[0] is prepared_body
    assert classifier_calls[1] == rebuilt_request_body
    assert owner_guard_bodies == classifier_calls
    assert all(
        body is prepared_body
        for provider, body in metadata_input_bodies
        if provider == "cursor_agent"
    )
    assert all(
        body is classifier_calls[1]
        for provider, body in metadata_input_bodies
        if provider == "openrouter"
    )
    assert candidate_bodies[0]["previous_response_id"] == "cursor-unretained"
    assert "previous_response_id" not in candidate_bodies[1]
    assert candidate_bodies[1]["input"] == [
        {
            "role": "user",
            "content": "Complete the original assignment in /workspace.",
        },
        {
            "type": "function_call",
            "call_id": "pwd-call",
            "name": "exec_command",
            "arguments": '{"cmd":"pwd","workdir":"/workspace"}',
        },
        {
            "type": "function_call_output",
            "call_id": "pwd-call",
            "output": "/workspace",
        },
    ]
    assert replay_safe_classifier(candidate_bodies[1]) is True
    assert len(publication_calls) == 1
    publication_plan = publication_calls[0]["plan"]
    assert publication_plan.applied_scope == "candidate"
    assert publication_plan.duration_seconds == 300.0
    assert publication_plan.memory_keys == (
        "cursor_agent:cursor-grok-4.6-high",
    )
    assert publication_plan.durable_keys == (
        "cursor_agent:cursor-grok-4.6-high",
    )
    assert publication_calls[0]["candidate"] is cursor_candidate
    assert routing_state.codex.cooldown_until_monotonic_by_key == {}
    assert routing_state.codex.candidate_semantic_ineligibility_by_key == {}


@pytest.mark.asyncio
async def test_candidate_loop_cursor_continuation_refunds_slot_before_xai_failover(  # noqa: PLR0915
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.llms.cursor_agent.connect import CursorConnectError
    from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime import (
        codex_candidate_calls,
    )

    cursor_candidate = {
        "provider": "cursor_agent",
        "model": "cursor_agent/cursor-grok-4.6-high",
        "route_family": "codex_cursor_agent_aiserver_adapter",
    }
    native_xai_candidate = {
        "provider": "xai",
        "model": "xai/grok-4.6",
        "route_family": "codex_grok_native_responses_adapter",
    }
    managed_xai_candidate = {
        "provider": "xai",
        "model": "oa_xai/grok-4.6",
        "route_family": "codex_xai_oauth_responses_adapter",
    }
    turn_id = "01a06269-1662-7c02-a81a-031c450f8606"
    function_names = (
        "exec_command",
        "write_stdin",
        "view_image",
        "get_goal",
        "create_goal",
        "update_goal",
        "list_mcp_resources",
        "list_mcp_resource_templates",
        "read_mcp_resource",
        "request_user_input",
        "request_plugin_install",
    )
    replay_tools = [
        {
            "type": "function",
            "name": name,
            "description": f"Run the {name} tool.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            "strict": False,
        }
        for name in function_names
    ]
    replay_tools.extend(
        [
            {
                "type": "custom",
                "name": "apply_patch",
                "description": "Apply a patch.",
            },
            {
                "description": (
                    "# Tool discovery\n\n"
                    "Searches over deferred tool metadata with BM25 and "
                    "exposes matching tools for the next model call.\n\n"
                    "You have access to tools from the following sources:\n"
                    "- Codex: Built-in tools.\n"
                    "Some of the tools may not have been provided to you "
                    "upfront, and you should use this tool (`tool_search`) "
                    "to search for the required tools."
                ),
                "execution": "client",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "Search query for deferred tools."
                            ),
                        },
                        "limit": {
                            "type": "number",
                            "description": (
                                "Maximum number of tools to return. "
                                "Defaults to 8."
                            ),
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                "type": "tool_search",
            },
            {
                "type": "web_search",
                "filters": {"allowed_domains": ["example.com"]},
            },
        ]
    )
    assert len(replay_tools) == 14
    assert set(replay_tools[12]) == {
        "description",
        "execution",
        "parameters",
        "type",
    }
    selections = [
        {
            "candidate": cursor_candidate,
            "lane_key": "cursor_agent_cli",
            "cooldown_key": "cursor_agent:cursor-grok-4.6-high",
            "selection_reason": "first_available",
            "has_account_bound_state": True,
            "in_flight_session": True,
        },
        {
            "candidate": native_xai_candidate,
            "lane_key": "xai_native",
            "cooldown_key": "xai:grok-4.6",
            "selection_reason": "next_available",
            "has_account_bound_state": True,
            "in_flight_session": True,
        },
        {
            "candidate": managed_xai_candidate,
            "lane_key": "xai_managed",
            "cooldown_key": "oa_xai:grok-4.6",
            "selection_reason": "next_available",
            "has_account_bound_state": True,
            "in_flight_session": True,
        },
    ]
    prepared_body = {
        "model": "work",
        "tools": replay_tools,
        "message_id": "cursor-message-snake",
        "messageId": "cursor-message-camel",
        "conversation_id": "cursor-conversation-snake",
        "conversationId": "cursor-conversation-camel",
        "conversation_group_id": "cursor-group-snake",
        "conversationGroupId": "cursor-group-camel",
        "run_id": "cursor-run-snake",
        "runId": "cursor-run-camel",
        "agent_session_id": "cursor-session-snake",
        "agentSessionId": "cursor-session-camel",
        "input": [
            {
                "type": "message",
                "id": "msg_01a06269-1827-79d2-b3a5-41d50a2fad1a",
                "role": "developer",
                "content": [
                    {"type": "input_text", "text": "model instructions"},
                    {"type": "input_text", "text": "developer instructions"},
                    {"type": "input_text", "text": "memory instructions"},
                    {"type": "input_text", "text": "skill instructions"},
                    {"type": "input_text", "text": "permission instructions"},
                    {"type": "input_text", "text": "app instructions"},
                ],
                "internal_chat_message_metadata_passthrough": {
                    "turn_id": turn_id,
                    "create_time": 1788357449.7673376,
                    "content_item_kinds": [
                        "model_switch.instructions",
                        "generic.developer_instructions",
                        "memories.instructions",
                        "host_skills.instructions",
                        "permissions.instructions",
                        "apps.instructions",
                    ],
                },
            },
            {
                "type": "message",
                "id": "msg_01a06269-1827-79d2-b3a5-41ed4566fa70",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "plugin recommendations"},
                    {"type": "input_text", "text": "AGENTS instructions"},
                    {"type": "input_text", "text": "environment context"},
                ],
                "internal_chat_message_metadata_passthrough": {
                    "turn_id": turn_id,
                    "create_time": 1788357449.7673385,
                    "content_item_kinds": [
                        "plugins.recommendations",
                        "agents_md.instructions",
                        "environments.environment_context",
                    ],
                },
            },
            {
                "type": "message",
                "id": "msg_01a06269-1847-7802-a387-e9c9ef8d2032",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Complete the original assignment in /workspace.",
                    }
                ],
                "internal_chat_message_metadata_passthrough": {
                    "turn_id": turn_id,
                    "create_time": 1788357449.79907,
                    "content_item_kinds": ["user.text"],
                },
            },
            {
                "type": "message",
                "id": "msg_resp_277feeae9f69433ab4e4ef2597a25db8",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "I will run pwd first.",
                    }
                ],
                "internal_chat_message_metadata_passthrough": {
                    "turn_id": turn_id,
                    "content_item_kinds": ["unknown"],
                },
            },
            {
                "type": "function_call",
                "id": "fc_a4335d9d-8539-9945-bdc7-f14243b0e9b8_0",
                "name": "exec_command",
                "arguments": '{"cmd":"pwd","workdir":"/workspace"}',
                "call_id": "call-8ef73738-1b5f-4aab-8789-fa1f309bb320-0",
                "internal_chat_message_metadata_passthrough": {
                    "turn_id": turn_id,
                },
            },
            {
                "type": "function_call_output",
                "id": "fco_01a06269-2d51-7de0-a7e3-bddda588ad80",
                "call_id": "call-8ef73738-1b5f-4aab-8789-fa1f309bb320-0",
                "output": "/workspace",
                "internal_chat_message_metadata_passthrough": {
                    "turn_id": turn_id,
                    "create_time": 1788357455.1854475,
                },
            },
        ],
    }
    for item in prepared_body["input"]:
        item.pop("internal_chat_message_metadata_passthrough", None)
    assert [set(item) for item in prepared_body["input"]] == [
        {"type", "id", "role", "content"},
        {"type", "id", "role", "content"},
        {"type", "id", "role", "content"},
        {"type", "id", "role", "content"},
        {"type", "id", "name", "arguments", "call_id"},
        {"type", "id", "call_id", "output"},
    ]
    with pytest.raises(CursorConnectError) as source_exc_info:
        codex_candidate_calls._raise_cursor_session_continuation_unavailable()
    source_exc = source_exc_info.value
    with pytest.raises(ProxyException) as mapped_exc_info:
        codex_candidate_calls._raise_cursor_agent_alias_error(
            exc=source_exc,
            candidate=cursor_candidate,
        )
    mapped_exc = mapped_exc_info.value
    rebuilt_request_body = (
        codex_candidate_calls._build_cursor_replay_safe_fresh_dispatch_body(
            prepared_body,
            continuation_exc=mapped_exc,
        )
    )
    assert rebuilt_request_body is not None
    assert all(
        field not in rebuilt_request_body
        for field in (
            "previous_response_id",
            "message_id",
            "messageId",
            "conversation_id",
            "conversationId",
            "conversation_group_id",
            "conversationGroupId",
            "run_id",
            "runId",
            "agent_session_id",
            "agentSessionId",
        )
    )
    assert rebuilt_request_body["input"][2] == {
        "role": "user",
        "content": "Complete the original assignment in /workspace.",
    }
    assert rebuilt_request_body["input"][-1] == {
        "type": "function_call_output",
        "call_id": "call-8ef73738-1b5f-4aab-8789-fa1f309bb320-0",
        "output": "/workspace",
    }
    replay_safe_classifier = (
        session_affinity.is_replay_safe_session_owner_redispatch_body
    )
    assert replay_safe_classifier(rebuilt_request_body) is True

    routing_state = AliasRoutingStateManager()
    monkeypatch.setattr(candidate_loop, "alias_routing_state", routing_state)
    selection_calls: list[dict[str, Any]] = []
    provider_calls: list[str] = []
    candidate_bodies: list[dict[str, Any]] = []
    metadata_input_bodies: list[tuple[str, dict[str, Any]]] = []
    classifier_calls: list[dict[str, Any]] = []
    metadata_attempts: list[list[dict[str, Any]]] = []
    publication_calls: list[dict[str, Any]] = []

    async def _select(**kwargs: Any) -> dict[str, Any]:
        selection_calls.append(kwargs)
        if not selections:
            raise AssertionError("candidate loop selected more than three candidates")
        return dict(selections.pop(0))

    async def _perform(
        *,
        candidate: dict[str, Any],
        candidate_body: dict[str, Any],
    ) -> object:
        provider_calls.append(candidate["model"])
        candidate_bodies.append(candidate_body)
        if candidate["provider"] == "cursor_agent":
            assert "previous_response_id" not in candidate_body
            assert candidate_body["input"] is prepared_body["input"]
            raise mapped_exc
        assert "previous_response_id" not in candidate_body
        assert candidate_body["input"] == rebuilt_request_body["input"]
        assert candidate_body["tools"] == replay_tools
        assert all(
            field not in candidate_body
            for field in (
                "message_id",
                "messageId",
                "conversation_id",
                "conversationId",
                "conversation_group_id",
                "conversationGroupId",
                "run_id",
                "runId",
                "agent_session_id",
                "agentSessionId",
            )
        )
        assert replay_safe_classifier(candidate_body) is True
        if candidate["model"] == "xai/grok-4.6":
            raise HTTPException(
                status_code=429,
                detail={"error": {"code": "rate_limit_exceeded"}},
            )
        return {"candidate": candidate["model"]}

    async def _no_active_cooldown(_key: str) -> tuple[float, str]:
        return 0.0, "memory"

    async def _noop_async(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def _owner_guard(**_kwargs: Any) -> object:
        return SimpleNamespace(
            decision=SimpleNamespace(value="no_session"),
            reservation_token=None,
            held_reservation=False,
            provenance=None,
        )

    class _Admission:
        async def admit_selected_candidate(self, **_kwargs: Any) -> object:
            return SimpleNamespace(allowed=True, lease=None)

        async def release_provider_lane_admission(self, _lease: object) -> None:
            return None

    def _classify_rebuilt_request(body: dict[str, Any]) -> bool:
        classifier_calls.append(body)
        return replay_safe_classifier(body)

    def _add_candidate_metadata(
        body: dict[str, Any],
        *,
        selection: dict[str, Any],
        attempts: list[dict[str, Any]],
        **_kwargs: Any,
    ) -> dict[str, Any]:
        metadata_input_bodies.append(
            (str(selection["candidate"]["provider"]), body)
        )
        metadata_attempts.append(attempts)
        rebuilt = dict(body)
        rebuilt["model"] = selection["candidate"]["model"]
        return rebuilt

    session_affinity_seam = SimpleNamespace(
        is_replay_safe_session_owner_redispatch_body=_classify_rebuilt_request,
        resolve_canonical_session_identity=lambda *_args, **_kwargs: None,
        get_request_codex_auto_review_parent_session_identity=lambda _request: None,
        build_session_owner_attributes=lambda **_kwargs: {},
        ensure_session_owner_guard_for_request=_owner_guard,
        get_request_session_owner_lease=lambda _request: None,
        finalize_session_owner_lease_on_success=_noop_async,
        finalize_session_owner_lease_on_failure=_noop_async,
        reset_released_request_session_owner_guard=lambda _request: False,
        SessionOwnerMutationOutcome=SimpleNamespace(
            CONFLICT="conflict",
            ERROR="error",
            NOT_HELD="not_held",
        ),
    )

    async def _capture_publication(**kwargs: Any) -> None:
        publication_calls.append(kwargs)

    monkeypatch.setattr(
        candidate_loop,
        "_session_affinity_mod",
        lambda: session_affinity_seam,
    )
    monkeypatch.setattr(candidate_loop, "_admission_mod", lambda: _Admission())
    monkeypatch.setattr(lpe, "_record_codex_failure_evidence", lambda **_kwargs: None)
    monkeypatch.setattr(
        lpe,
        "_plan_codex_oauth_account_failover",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        lpe,
        "_exclude_codex_auto_agent_request_local_candidate_without_cooldown",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        lpe,
        "execute_cooldown_publication_transaction",
        _capture_publication,
    )

    response = await candidate_loop.handle_alias_route(
        SimpleNamespace(
            select_candidate_fn=_select,
            perform_candidate_request_fn=_perform,
            resolve_cooldown_publication_fn=lpe._resolve_auto_agent_cooldown_publication_plan,
            publish_cooldown_memory_fn=_capture_publication,
            persist_cooldown_fn=_capture_publication,
            set_session_affinity_fn=_noop_async,
            add_alias_metadata_fn=_add_candidate_metadata,
                raise_redispatch_fn=lambda **_kwargs: (_ for _ in ()).throw(
                    AssertionError("this regression must not redispatch")
                ),
        ),
        alias_family="codex_auto_agent",
        alias_model="work",
        request=SimpleNamespace(state=SimpleNamespace()),
        prepared_request_body=prepared_body,
        max_candidate_attempts=2,
        get_active_cooldown_state_fn=_no_active_cooldown,
        attempts_metadata_key="attempts",
        skipped_candidates_metadata_key="skipped",
        no_candidate_detail="no candidates",
        log_label="Codex",
    )

    assert response == {"candidate": managed_xai_candidate["model"]}
    assert provider_calls == [
        cursor_candidate["model"],
        native_xai_candidate["model"],
        managed_xai_candidate["model"],
    ]
    assert len(selection_calls) == 3
    assert selection_calls[0]["excluded_candidate_keys"] == frozenset()
    assert selection_calls[1]["excluded_candidate_keys"] == frozenset(
        {"cursor_agent:cursor-grok-4.6-high"}
    )
    assert selection_calls[2]["excluded_candidate_keys"] == frozenset(
        {"cursor_agent:cursor-grok-4.6-high"}
    )
    assert all(
        "previous_response_id" not in body for body in candidate_bodies
    )
    assert candidate_bodies[1]["input"][-2:] == [
        {
            "type": "function_call",
            "call_id": "call-8ef73738-1b5f-4aab-8789-fa1f309bb320-0",
            "name": "exec_command",
            "arguments": '{"cmd":"pwd","workdir":"/workspace"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call-8ef73738-1b5f-4aab-8789-fa1f309bb320-0",
            "output": "/workspace",
        },
    ]
    assert all(
        "id" not in item
        and "internal_chat_message_metadata_passthrough" not in item
        for item in candidate_bodies[1]["input"]
    )
    assert candidate_bodies[2]["input"] == candidate_bodies[1]["input"]
    assert classifier_calls[0] is prepared_body
    assert classifier_calls[1] == rebuilt_request_body
    assert metadata_input_bodies[0][1] is prepared_body
    assert metadata_attempts
    assert all(captured is metadata_attempts[0] for captured in metadata_attempts)
    attempts = metadata_attempts[-1]
    assert attempts[0]["status"] == "cooldown_set"
    assert attempts[0]["error_class"] == "continuation_state_unavailable"
    assert attempts[0]["attempted_provider_call"] is False
    assert attempts[0]["cooldown_scope"] == "candidate"
    assert attempts[0]["cooldown_seconds"] == 300.0
    assert attempts[0]["failure_phase"] == "cursor_session_continuation"
    assert attempts[0]["source_error"]
    assert attempts[1]["error_class"] == "rate_limited"
    assert len(publication_calls) == 1
    publication_plan = publication_calls[0]["plan"]
    assert publication_plan.applied_scope == "candidate"
    assert publication_plan.duration_seconds == 300.0
    assert publication_plan.memory_keys == (
        "cursor_agent:cursor-grok-4.6-high",
    )
    assert publication_plan.durable_keys == (
        "cursor_agent:cursor-grok-4.6-high",
    )
    assert routing_state.codex.cooldown_until_monotonic_by_key == {}
    assert routing_state.codex.candidate_semantic_ineligibility_by_key == {}


@pytest.mark.asyncio
async def test_candidate_loop_cursor_sanitized_proto_structure_reaches_attempt_and_terminal_audit(  # noqa: PLR0915
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.llms.cursor_agent.connect import (
        CursorConnectProtocolError,
    )
    from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime import (
        codex_candidate_calls,
    )

    candidate = {
        "provider": "cursor_agent",
        "model": "cursor_agent/cursor-grok-4.6-high",
        "route_family": "codex_cursor_agent_aiserver_adapter",
    }
    selection = {
        "candidate": candidate,
        "lane_key": "cursor_agent_cli",
        "cooldown_key": "cursor_agent:cursor-grok-4.6-high",
        "selection_reason": "first_available",
        "skipped": [],
    }
    secret_prompt = "PROMPT_VALUE_MUST_NOT_APPEAR"
    secret_output = "OUTPUT_VALUE_MUST_NOT_APPEAR"
    secret_tool_description = "TOOL_SCHEMA_MUST_NOT_APPEAR"
    body = {
        "model": "work",
        "previous_response_id": "cursor-unretained",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": secret_prompt,
            },
            {
                "type": "function_call_output",
                "call_id": "call-opaque-secret",
                "output": secret_output,
            },
        ],
        "stream": False,
        "tools": [
            {
                "type": "function",
                "name": "secret-tool",
                "description": secret_tool_description,
                "parameters": {"type": "object"},
            },
            {"type": "custom", "description": secret_tool_description},
        ],
    }
    expected_structure = {
        "fields": [
            {
                "field_number": 1,
                "wire_type": 0,
                "payload_length": 1,
            },
            {
                "field_number": 4,
                "wire_type": 2,
                "payload_length": 12,
                "nested_fields": [
                    {
                        "field_number": 2,
                        "wire_type": 2,
                        "payload_length": 16,
                    }
                ],
            },
        ]
    }
    source_exc = CursorConnectProtocolError(
        "Cursor Agent requested unsupported local exec operation field 4.",
        body={
            "fields": [
                {
                    "field_number": 1,
                    "wire_type": 0,
                    "payload_length": 1,
                    "value": "secret-command",
                },
                {
                    "field_number": 4,
                    "wire_type": 2,
                    "payload_length": 12,
                    "nested_fields": [
                        {
                            "field_number": 2,
                            "wire_type": 2,
                            "payload_length": 16,
                            "value": "/secret/workspace",
                        }
                    ],
                },
            ],
            "raw": b"opaque-provider-bytes",
        },
    )
    setattr(
        source_exc,
        codex_candidate_calls._CURSOR_SESSION_CONTINUATION_FAILURE_MARKER,
        True,
    )
    with pytest.raises(ProxyException) as mapped_exc_info:
        codex_candidate_calls._raise_cursor_agent_alias_error(
            exc=source_exc,
            candidate=candidate,
        )
    mapped_exc = mapped_exc_info.value
    field_name = codex_candidate_calls._CURSOR_SANITIZED_PROTO_STRUCTURE_FIELD
    assert mapped_exc.detail[field_name] == expected_structure
    assert getattr(
        mapped_exc,
        codex_candidate_calls._CURSOR_SESSION_CONTINUATION_FAILURE_MARKER,
    )

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/openai_passthrough/v1/responses",
            "headers": [
                (b"user-agent", b"codex-cli/1.0"),
                (b"originator", b"codex_cli_rs"),
            ],
            "query_string": b"",
            "server": ("testserver", 80),
            "client": ("testclient", 123),
            "scheme": "http",
        }
    )
    failure_records: list[dict[str, Any]] = []
    persisted: list[list[dict[str, Any]]] = []
    terminal_records: list[dict[str, Any]] = []
    cooldown_memory_publications: list[dict[str, Any]] = []
    cooldown_persistences: list[dict[str, Any]] = []
    builder_results: list[object] = []
    original_builder = (
        codex_candidate_calls._build_cursor_replay_safe_fresh_dispatch_body
    )

    def _build_and_record(*args: Any, **kwargs: Any) -> object:
        result = original_builder(*args, **kwargs)
        builder_results.append(result)
        return result

    monkeypatch.setattr(
        codex_candidate_calls,
        "_build_cursor_replay_safe_fresh_dispatch_body",
        _build_and_record,
    )

    async def _select(**_kwargs: Any) -> dict[str, Any]:
        return selection

    async def _perform(**_kwargs: Any) -> object:
        raise mapped_exc

    async def _no_active_cooldown(_key: str) -> tuple[float, str]:
        return 0.0, "memory"

    async def _noop_async(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def _owner_guard(**_kwargs: Any) -> object:
        return SimpleNamespace(
            decision=SimpleNamespace(value="no_session"),
            reservation_token=None,
            held_reservation=False,
            provenance=None,
        )

    class _Admission:
        async def admit_selected_candidate(self, **_kwargs: Any) -> object:
            return SimpleNamespace(allowed=True, lease=None)

        async def release_provider_lane_admission(self, _lease: object) -> None:
            return None

    def _record_failure(**kwargs: Any) -> None:
        failure_records.append(kwargs)

    def _capture_cooldown_memory_publication(**kwargs: Any) -> None:
        cooldown_memory_publications.append(kwargs)

    async def _capture_cooldown_persistence(**kwargs: Any) -> None:
        cooldown_persistences.append(kwargs)

    def _add_metadata(
        candidate_body: dict[str, Any],
        *,
        attempts: list[dict[str, Any]],
        **_kwargs: Any,
    ) -> dict[str, Any]:
        return {
            **candidate_body,
            "litellm_metadata": {"codex_auto_agent_attempts": attempts},
        }

    session_affinity = SimpleNamespace(
        is_replay_safe_session_owner_redispatch_body=lambda _body: False,
        resolve_canonical_session_identity=lambda *_args, **_kwargs: None,
        get_request_codex_auto_review_parent_session_identity=lambda _request: None,
        build_session_owner_attributes=lambda **_kwargs: {},
        ensure_session_owner_guard_for_request=_owner_guard,
        get_request_session_owner_lease=lambda _request: None,
        finalize_session_owner_lease_on_success=_noop_async,
        finalize_session_owner_lease_on_failure=_noop_async,
        reset_released_request_session_owner_guard=lambda _request: False,
        SessionOwnerMutationOutcome=SimpleNamespace(
            CONFLICT="conflict",
            ERROR="error",
            NOT_HELD="not_held",
        ),
    )

    class _Services(SimpleNamespace):
        pass

    services = _Services(
        select_candidate_fn=_select,
        perform_candidate_request_fn=_perform,
        resolve_cooldown_publication_fn=lpe._resolve_auto_agent_cooldown_publication_plan,
        publish_cooldown_memory_fn=_capture_cooldown_memory_publication,
        persist_cooldown_fn=_capture_cooldown_persistence,
        set_session_affinity_fn=_noop_async,
        add_alias_metadata_fn=_add_metadata,
        raise_redispatch_fn=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("Cursor protocol ineligibility must not redispatch")
        ),
    )

    monkeypatch.setattr(candidate_loop, "alias_routing_state", AliasRoutingStateManager())
    monkeypatch.setattr(candidate_loop, "_session_affinity_mod", lambda: session_affinity)
    monkeypatch.setattr(candidate_loop, "_admission_mod", lambda: _Admission())
    monkeypatch.setattr(
        lpe,
        "_record_auto_agent_alias_attempt_failure",
        _record_failure,
    )
    monkeypatch.setattr(
        lpe,
        "_persist_auto_agent_alias_audit_only_events_best_effort",
        lambda events, *, request_body=None: persisted.append(events),
    )
    monkeypatch.setattr(
        lpe,
        "_emit_auto_agent_alias_route_event",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "litellm.proxy.aawm_runtime_error_logging.persist_agent_terminal_error",
        lambda **kwargs: terminal_records.append(kwargs) or True,
    )

    with pytest.raises(HTTPException) as caught:
        await candidate_loop.handle_alias_route(
            services,
            alias_family="codex_auto_agent",
            alias_model="work",
            request=request,
            prepared_request_body=body,
            max_candidate_attempts=1,
            get_active_cooldown_state_fn=_no_active_cooldown,
            attempts_metadata_key="codex_auto_agent_attempts",
            skipped_candidates_metadata_key="codex_auto_agent_skipped_candidates",
            no_candidate_detail="no candidates",
            log_label="Codex",
        )

    assert caught.value.status_code == 409
    assert caught.value.detail["error"]["code"] == (
        "aawm_codex_auto_agent_candidate_ineligible"
    )
    assert caught.value.detail[field_name] == expected_structure
    assert builder_results == [None]
    assert len(failure_records) == 1
    attempt = failure_records[0]["attempt_record"]
    assert attempt[field_name] == expected_structure
    assert attempt["error_class"] == "continuation_state_unavailable"
    assert attempt["cooldown_scope"] == "candidate"
    assert attempt["cooldown_seconds"] == 300.0
    assert attempt["attempted_provider_call"] is False
    assert attempt["failure_phase"] == "cursor_session_continuation"
    rejection_field = codex_candidate_calls._CURSOR_REPLAY_FRESH_DISPATCH_REJECT_FIELD
    expected_rejection = {
        "stage": "fresh_body_copy",
        "reason": "replay_state_lookup",
    }
    assert attempt[rejection_field] == expected_rejection
    assert cooldown_memory_publications == [
        {"keys": (selection["cooldown_key"],), "seconds": 300.0}
    ]
    assert cooldown_persistences == [
        {"keys": (selection["cooldown_key"],), "seconds": 300.0}
    ]
    assert len(persisted) == 1
    terminal_event = persisted[0][-1]
    assert terminal_event["event_type"] == "no_candidate_available"
    assert terminal_event[rejection_field] == expected_rejection
    assert terminal_event["attempts"][0][field_name] == expected_structure
    assert terminal_event["attempts"][0][rejection_field] == expected_rejection
    expected_request_shape_summary = {
        "body_container_type": "object",
        "body_top_level_keys": [
            "input",
            "model",
            "previous_response_id",
            "stream",
            "tools",
        ],
        "previous_response_id_state": "nonempty",
        "input_container_type": "array",
        "input_item_count": 2,
        "input_item_type_counts": {
            "function_call_output": 1,
            "message": 1,
        },
        "input_item_shape_samples": [
            {
                "index": 0,
                "container_type": "object",
                "type": "message",
                "keys": ["content", "role", "type"],
            },
            {
                "index": 1,
                "container_type": "object",
                "type": "function_call_output",
                "keys": ["call_id", "output", "type"],
            },
        ],
        "tool_count": 2,
        "tool_type_counts": {"custom": 1, "function": 1},
    }
    assert (
        terminal_event["aawm_passthrough_request_shape_summary"]
        == expected_request_shape_summary
    )
    assert len(terminal_records) == 1
    terminal_context = terminal_records[0]["error_context"]
    assert terminal_context["attempts"][0][field_name] == expected_structure
    assert terminal_context[rejection_field] == expected_rejection
    assert terminal_context["attempts"][0][rejection_field] == expected_rejection
    assert (
        terminal_context["aawm_passthrough_request_shape_summary"]
        == expected_request_shape_summary
    )
    from litellm.proxy.aawm_runtime_error_logging import (
        build_agent_terminal_error_record,
    )

    terminal_record = build_agent_terminal_error_record(
        error_context=terminal_context,
        terminal_outcome="agent_session_terminated",
        fallback_result="no_candidate_available",
        redispatch_required=False,
        agent_session_killed=True,
    )
    assert (
        terminal_record["context"]["aawm_passthrough_request_shape_summary"]
        == expected_request_shape_summary
    )
    assert terminal_record["context"][rejection_field] == expected_rejection
    assert terminal_record["context"]["attempts"][0][rejection_field] == (
        expected_rejection
    )

    serialized = json.dumps(
        {
            "attempt": attempt,
            "terminal_event": terminal_event,
            "terminal_context": terminal_context,
            "terminal_detail": caught.value.detail,
        }
    )
    assert "secret-command" not in serialized
    assert "/secret/workspace" not in serialized
    assert "opaque-provider-bytes" not in serialized
    assert secret_prompt not in serialized
    assert secret_output not in serialized
    assert secret_tool_description not in serialized
    assert "cursor-unretained" not in serialized
    assert '"value"' not in serialized
    assert '"raw"' not in serialized


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", (400, 422))
@pytest.mark.parametrize("include_failure_metadata", (True, False))
async def test_candidate_loop_kimi_invalid_request_persists_terminal_inventory_before_raise(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
    include_failure_metadata: bool,
) -> None:
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/openai_passthrough/v1/responses",
            "headers": [
                (b"user-agent", b"codex-cli/1.0"),
                (b"originator", b"codex_cli_rs"),
            ],
            "query_string": b"",
            "server": ("testserver", 80),
            "client": ("testclient", 123),
            "scheme": "http",
        }
    )
    body = {"model": "kimi-terminal-test", "input": "hello", "stream": False}
    candidate = {
        "provider": "kimi_code",
        "model": "kimi_code/k3-high",
        "route_family": "codex_kimi_chat_completions_adapter",
        "last_resort": True,
    }
    selection = {
        "candidate": candidate,
        "lane_key": "kimi_code_managed_account",
        "cooldown_key": "kimi_code:kimi_code/k3-high:kimi_code_managed_account",
        "selection_reason": "last_resort",
        "skipped": [],
    }

    class _KimiInvalidRequest(RuntimeError):
        def __init__(self) -> None:
            super().__init__("Managed Kimi Code rejected the request shape.")
            self.status_code = status_code
            self.detail = {
                "error": {
                    "message": "Managed Kimi Code rejected the request shape.",
                    "type": "invalid_request_error",
                    "code": "kimi_code_invalid_request",
                }
            }
            if include_failure_metadata:
                self.kimi_code_probe_failure_metadata = {
                    "kind": "malformed",
                    "scope": "none",
                    "upstream_id": "k3",
                    "metadata_gate": "none",
                    "status_code": status_code,
                    "trace_id": "trace-candidate-loop",
                    "reset_reason": "malformed_provider_response",
                }

    async def _select(**_kwargs) -> dict[str, Any]:
        return selection

    async def _perform(**_kwargs) -> object:
        raise _KimiInvalidRequest()

    async def _no_active_cooldown(_key: str) -> tuple[float, str]:
        return 0.0, "memory"

    async def _noop_async(*_args, **_kwargs) -> None:
        return None

    def _add_metadata(
        candidate_body: dict[str, Any],
        *,
        attempts: list[dict[str, Any]],
        **_kwargs,
    ) -> dict[str, Any]:
        return {
            **candidate_body,
            "litellm_metadata": {"codex_auto_agent_attempts": attempts},
        }

    services = SimpleNamespace(
        select_candidate_fn=_select,
        perform_candidate_request_fn=_perform,
        resolve_cooldown_publication_fn=lpe._resolve_auto_agent_cooldown_publication_plan,
        publish_cooldown_memory_fn=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("Kimi invalid requests must not publish cooldown memory")
        ),
        persist_cooldown_fn=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("Kimi invalid requests must not persist cooldown state")
        ),
        set_session_affinity_fn=_noop_async,
        add_alias_metadata_fn=_add_metadata,
        raise_redispatch_fn=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("Kimi invalid requests must not signal redispatch")
        ),
    )

    async def _owner_lookup(**_kwargs):
        return None, None, None

    async def _owner_guard(**_kwargs):
        return SimpleNamespace(
            decision=SimpleNamespace(value="no_session"),
            reservation_token=None,
            held_reservation=False,
            provenance=None,
        )

    session_affinity = SimpleNamespace(
        is_replay_safe_session_owner_redispatch_body=lambda _body: False,
        resolve_canonical_session_identity=lambda *_args, **_kwargs: None,
        get_request_codex_auto_review_parent_session_identity=lambda _request: None,
        build_session_owner_attributes=lambda **_kwargs: {},
        ensure_session_owner_guard_for_request=_owner_guard,
        get_request_session_owner_lease=lambda _request: None,
        finalize_session_owner_lease_on_success=_noop_async,
        finalize_session_owner_lease_on_failure=_noop_async,
        get_session_owner_record=_owner_lookup,
        request_has_effective_session_identity=lambda _request: False,
        build_session_owner_provenance=lambda **_kwargs: {},
        reset_released_request_session_owner_guard=lambda _request: False,
        SessionOwnerMutationOutcome=SimpleNamespace(
            CONFLICT="conflict",
            ERROR="error",
            NOT_HELD="not_held",
        ),
    )

    class _Admission:
        async def admit_selected_candidate(self, **_kwargs):
            return SimpleNamespace(allowed=True, lease=None)

        async def release_provider_lane_admission(self, _lease):
            return None

    persisted: list[list[dict[str, Any]]] = []
    terminal_records: list[dict[str, Any]] = []
    monkeypatch.setattr(candidate_loop, "alias_routing_state", AliasRoutingStateManager())
    monkeypatch.setattr(candidate_loop, "_session_affinity_mod", lambda: session_affinity)
    monkeypatch.setattr(candidate_loop, "_admission_mod", lambda: _Admission())
    monkeypatch.setattr(lpe, "_record_codex_failure_evidence", lambda **_kwargs: None)
    monkeypatch.setattr(lpe, "_plan_codex_oauth_account_failover", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        lpe,
        "_persist_auto_agent_alias_audit_only_events_best_effort",
        lambda events, *, request_body=None: persisted.append(events),
    )
    monkeypatch.setattr(lpe, "_emit_auto_agent_alias_route_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "litellm.proxy.aawm_runtime_error_logging.persist_agent_terminal_error",
        lambda **kwargs: terminal_records.append(kwargs) or True,
    )

    with pytest.raises(HTTPException) as caught:
        await candidate_loop.handle_alias_route(
            services,
            alias_family="codex_auto_agent",
            alias_model=body["model"],
            request=request,
            prepared_request_body=body,
            max_candidate_attempts=1,
            get_active_cooldown_state_fn=_no_active_cooldown,
            attempts_metadata_key="codex_auto_agent_attempts",
            skipped_candidates_metadata_key="codex_auto_agent_skipped_candidates",
            no_candidate_detail="no candidates",
            log_label="Codex",
        )

    assert caught.value.status_code == status_code
    assert caught.value.detail["error"]["code"] == "kimi_code_invalid_request"
    assert len(persisted) == 1
    terminal_event = persisted[0][-1]
    assert terminal_event["event_type"] == "no_candidate_available"
    assert terminal_event["error_status_code"] == status_code
    assert terminal_event["candidate_count"] == 1
    assert terminal_event["candidates"][0]["model"] == "kimi_code/k3-high"
    assert terminal_event["candidates"][0]["terminal_disposition"] == "attempted"
    assert terminal_event["candidates"][0]["reason"] == "kimi_code_no_cooldown"
    assert terminal_event["attempts"][0]["cooldown_scope"] == "none"
    assert terminal_event["attempts"][0]["request_outcome"] == "failed"
    assert len(terminal_records) == 1
    assert terminal_records[0]["error_context"]["candidate_count"] == 1


@pytest.mark.asyncio
async def test_candidate_loop_compiled_basic_failure_matrix_reaches_luna_and_accounts_terminal_inventory(  # noqa: PLR0915
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_snapshot = snapshot_select.set_active_routing_snapshot(
        compile_directory(DEFAULT_CONFIG_DIR)
    )
    # 15:00 UTC is 23:00 at +08:00, inside Alibaba's 22:00-08:00 window.
    fixed_schedule_now = datetime(2026, 9, 1, 15, 0, tzinfo=timezone.utc)
    schedule_predicate = snapshot_select._is_snapshot_candidate_in_schedule_window

    def _fixed_schedule_predicate(candidate: Any, *, now_utc: datetime) -> bool:
        _ = now_utc
        return schedule_predicate(candidate, now_utc=fixed_schedule_now)

    monkeypatch.setattr(
        snapshot_select,
        "_is_snapshot_candidate_in_schedule_window",
        _fixed_schedule_predicate,
    )
    cooldowns: dict[str, float] = {}
    persisted: list[list[dict[str, Any]]] = []

    async def _noop_async(*_args, **_kwargs) -> None:
        return None

    async def _owner_lookup(**_kwargs):
        return None, None, None

    async def _owner_guard(**_kwargs):
        return SimpleNamespace(
            decision=SimpleNamespace(value="no_session"),
            reservation_token=None,
            held_reservation=False,
            provenance=None,
        )

    session_affinity = SimpleNamespace(
        is_replay_safe_session_owner_redispatch_body=lambda _body: False,
        resolve_canonical_session_identity=lambda *_args, **_kwargs: None,
        get_request_codex_auto_review_parent_session_identity=lambda _request: None,
        build_session_owner_attributes=lambda **_kwargs: {},
        ensure_session_owner_guard_for_request=_owner_guard,
        get_request_session_owner_lease=lambda _request: None,
        finalize_session_owner_lease_on_success=_noop_async,
        finalize_session_owner_lease_on_failure=_noop_async,
        get_session_owner_record=_owner_lookup,
        request_has_effective_session_identity=lambda _request: False,
        build_session_owner_provenance=lambda **_kwargs: {},
        reset_released_request_session_owner_guard=lambda _request: False,
        SessionOwnerMutationOutcome=SimpleNamespace(
            CONFLICT="conflict",
            ERROR="error",
            NOT_HELD="not_held",
        ),
    )

    class _Admission:
        async def admit_selected_candidate(self, **_kwargs):
            return SimpleNamespace(allowed=True, lease=None)

        async def release_provider_lane_admission(self, _lease):
            return None

    async def _active_cooldown(key: str) -> tuple[float, str]:
        return cooldowns.get(key, 0.0), "memory"

    async def _codex_oauth_contexts(
        _request,
        *,
        candidate_template: dict[str, Any],
        affinity=None,
    ) -> list[dict[str, Any]]:
        _ = affinity
        return [
            {
                "candidate": {
                    **candidate_template,
                    "codex_oauth_account_label": "account-test",
                    "codex_oauth_account_hash": "account-hash-test",
                    "codex_oauth_lane_key": "codex-oauth:test",
                    "codex_oauth_credential_affinity": "interchangeable",
                },
                "lane_key": "codex-oauth:test",
                "auth_status": "ready",
            }
        ]

    async def _execute_publication(
        *,
        plan,
        publish_cooldown_memory_fn,
        **_kwargs,
    ):
        publish_cooldown_memory_fn(
            keys=plan.memory_keys,
            seconds=plan.duration_seconds,
            allow_ttl_shrink=plan.allow_ttl_shrink,
        )
        return None

    def _publish_cooldown(
        *,
        keys,
        seconds: float,
        allow_ttl_shrink: bool = False,
    ) -> None:
        _ = allow_ttl_shrink
        for key in keys:
            cooldowns[key] = seconds

    async def _no_openrouter_quota(
        *,
        candidate,
        cooldown_seconds,
        cooldown_state_source,
        skip_reason,
    ):
        _ = candidate
        return cooldown_seconds, cooldown_state_source, skip_reason

    async def _no_openrouter_adapter_cooldown(_model: str) -> float:
        return 0.0

    monkeypatch.setattr(candidate_loop, "_session_affinity_mod", lambda: session_affinity)
    monkeypatch.setattr(candidate_loop, "_admission_mod", lambda: _Admission())
    monkeypatch.setattr(lpe, "_session_affinity_mod", lambda: session_affinity)
    monkeypatch.setattr(lpe, "_get_codex_auto_agent_active_cooldown_state", _active_cooldown)
    monkeypatch.setattr(
        lpe,
        "_get_openrouter_adapter_active_cooldown_seconds",
        _no_openrouter_adapter_cooldown,
    )
    monkeypatch.setattr(
        lpe,
        "_apply_openrouter_durable_quota_candidate_cooldown",
        _no_openrouter_quota,
    )
    monkeypatch.setattr(lpe, "_apply_cohere_local_quota_state", lambda state: state)
    monkeypatch.setattr(lpe, "_resolve_codex_oauth_account_candidate_contexts", _codex_oauth_contexts)
    monkeypatch.setattr(lpe._aawm_selection, "_hydrate_codex_oauth_quota_observations", _noop_async)
    monkeypatch.setattr(lpe, "_get_codex_auto_agent_session_affinity", lambda _key: _noop_async())
    monkeypatch.setattr(lpe, "_record_codex_failure_evidence", lambda **_kwargs: None)
    monkeypatch.setattr(lpe, "_plan_codex_oauth_account_failover", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(lpe, "execute_cooldown_publication_transaction", _execute_publication)
    monkeypatch.setattr(
        lpe._codex_failure_evidence_gate,
        "current_decision",
        lambda **_kwargs: SimpleNamespace(should_cool=True, duration_seconds=30.0),
    )
    monkeypatch.setattr(
        lpe,
        "_persist_auto_agent_alias_audit_only_events_best_effort",
        lambda events, *, request_body=None: persisted.append(events),
    )
    monkeypatch.setattr(lpe, "_emit_auto_agent_alias_route_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "litellm.proxy.aawm_runtime_error_logging.persist_agent_terminal_error",
        lambda **_kwargs: True,
    )

    def _request(*, claude_origin: bool = False) -> Request:
        return Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/openai_passthrough/v1/responses",
                "headers": [
                    (
                        b"user-agent",
                        b"claude-code/1.0" if claude_origin else b"codex-cli/1.0",
                    ),
                    (
                        b"originator",
                        b"claude-code" if claude_origin else b"codex_cli_rs",
                    ),
                ],
                "query_string": b"",
                "server": ("testserver", 80),
                "client": ("testclient", 123),
                "scheme": "http",
            }
        )

    def _failure_for(model: str) -> Exception:
        if model in {
            "openrouter/cohere/north-mini-code:free",
            "alibaba_token_plan/deepseek-v4-flash-0731",
            "zai_coding_plan/glm-5.3-flash",
        }:
            return _IneligibleCandidateError()
        if model == "big-pickle":
            return HTTPException(status_code=401, detail="generic provider auth failure")
        if model == "deepseek-v4-flash-free":
            return HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "code": "OPENROUTER_INVALID_CHAT_MESSAGE",
                        "message": "adapter rejected message shape",
                    }
                },
            )
        if model == "cursor_agent/composer-2.5":
            return HTTPException(status_code=504, detail="adapter timed out")
        return HTTPException(
            status_code=429,
            detail={"error": {"code": "rate_limit_exceeded"}},
        )

    async def _run(*, success_model: str | None):
        cooldowns.clear()
        request = _request()
        body = {"model": "basic", "input": "hello", "stream": False}
        calls: list[tuple[str, dict[str, Any]]] = []

        async def _perform(
            *,
            candidate: dict[str, Any],
            candidate_body: dict[str, Any],
        ) -> object:
            calls.append((candidate["model"], candidate_body))
            if candidate["model"] == success_model:
                return {"model": candidate["model"], "body": candidate_body}
            raise _failure_for(candidate["model"])

        services = SimpleNamespace(
            select_candidate_fn=lpe._select_codex_auto_agent_candidate,
            perform_candidate_request_fn=_perform,
            resolve_cooldown_publication_fn=lpe._resolve_auto_agent_cooldown_publication_plan,
            publish_cooldown_memory_fn=_publish_cooldown,
            persist_cooldown_fn=_noop_async,
            set_session_affinity_fn=_noop_async,
            add_alias_metadata_fn=lpe._add_codex_auto_agent_alias_metadata,
            raise_redispatch_fn=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("fresh basic traversal must not signal redispatch")
            ),
        )
        client_product = lpe._extract_auto_agent_alias_client_product_label(
            request,
            body,
        )
        enumeration = lpe._resolve_aawm_alias_selection_enumeration(
            request,
            "basic",
            ingress="codex",
            client_product_label=client_product,
        )
        return await candidate_loop.handle_alias_route(
            services,
            alias_family="codex_auto_agent",
            alias_model="basic",
            request=request,
            prepared_request_body=body,
            max_candidate_attempts=len(enumeration.candidates),
            get_active_cooldown_state_fn=_active_cooldown,
            attempts_metadata_key="codex_auto_agent_attempts",
            skipped_candidates_metadata_key="codex_auto_agent_skipped_candidates",
            no_candidate_detail="no candidates",
            log_label="Codex",
        ), calls

    try:
        monkeypatch.setattr(candidate_loop, "alias_routing_state", AliasRoutingStateManager())
        luna_response, luna_calls = await _run(success_model="gpt-5.6-luna")
        assert luna_response["model"] == "gpt-5.6-luna"
        assert [model for model, _body in luna_calls] == [
            "cohere/north-mini-code-1-0",
            "openrouter/cohere/north-mini-code:free",
            "big-pickle",
            "alibaba_token_plan/deepseek-v4-flash-0731",
            "zai_coding_plan/glm-5.3-flash",
            "cursor_agent/composer-2.5",
            "gpt-5.6-luna",
        ]
        luna_body = luna_calls[-1][1]
        assert luna_body["model"] == "gpt-5.6-luna"
        assert luna_body["reasoning"] == {"effort": "low"}

        monkeypatch.setattr(candidate_loop, "alias_routing_state", AliasRoutingStateManager())
        early_response, early_calls = await _run(success_model="big-pickle")
        assert early_response["model"] == "big-pickle"
        assert [model for model, _body in early_calls] == [
            "cohere/north-mini-code-1-0",
            "openrouter/cohere/north-mini-code:free",
            "big-pickle",
        ]
        assert all(model != "gpt-5.6-luna" for model, _body in early_calls)

        claude_request = _request(claude_origin=True)
        claude_body = {"model": "basic", "input": "hello", "stream": False}
        claude_product = lpe._extract_auto_agent_alias_client_product_label(
            claude_request,
            claude_body,
        )
        assert claude_product == "Claude/1.0"
        claude_candidates = lpe._resolve_aawm_alias_selection_enumeration(
            claude_request,
            "basic",
            ingress="codex",
            client_product_label=claude_product,
        ).candidates
        assert len(claude_candidates) == 6
        assert all(
            candidate["model"] != "gpt-5.6-luna"
            for candidate in claude_candidates
        )

        monkeypatch.setattr(candidate_loop, "alias_routing_state", AliasRoutingStateManager())
        with pytest.raises(HTTPException) as caught:
            await _run(success_model=None)
        assert caught.value.status_code == 429
        terminal_event = persisted[-1][-1]
        inventory = terminal_event["candidates"]
        assert [candidate["model"] for candidate in inventory] == [
            "cohere/north-mini-code-1-0",
            "openrouter/cohere/north-mini-code:free",
            "big-pickle",
            "alibaba_token_plan/deepseek-v4-flash-0731",
            "zai_coding_plan/glm-5.3-flash",
            "cursor_agent/composer-2.5",
            "gpt-5.6-luna",
        ]
        assert [
            (candidate["terminal_disposition"], candidate["reason"])
            for candidate in inventory
        ] == [
            ("attempted", "rate_limited"),
            ("attempted", "candidate_deterministically_ineligible"),
            ("attempted", "provider_terminal_error"),
            ("attempted", "candidate_deterministically_ineligible"),
            ("attempted", "candidate_deterministically_ineligible"),
            ("attempted", "upstream_timeout"),
            ("attempted", "rate_limited"),
        ]
        assert inventory[-1]["reasoning_effort"] == "low"
        assert terminal_event["candidate_count"] == len(inventory) == 7
    finally:
        snapshot_select.set_active_routing_snapshot(previous_snapshot)
