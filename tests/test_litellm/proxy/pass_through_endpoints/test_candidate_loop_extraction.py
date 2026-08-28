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

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi import HTTPException
import pytest
from starlette.requests import Request

from litellm.proxy.pass_through_endpoints import aawm_alias_routing as package
from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import candidate_loop
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import snapshot_select
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_startup import (
    DEFAULT_CONFIG_DIR,
    compile_directory,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
    AliasRoutingStateManager,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.policy import (
    CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER,
    CODEX_AUTO_AGENT_ZAI_CODING_PLAN_PROVIDER,
)

PACKAGE_DIR = Path(package.__file__).resolve().parent
GOD_PATH = Path(lpe.__file__).resolve()
CANDIDATE_LOOP_PATH = PACKAGE_DIR / "candidate_loop.py"
INTERFACES_PATH = PACKAGE_DIR / "interfaces.py"


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
            self.detail = {
                "error": {
                    "type": "invalid_request_error",
                    "code": "ModelNotFound",
                    "message": "Model not exist",
                }
            }

    exc = _StructuredModelNotFound()
    candidate = {"provider": CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER}
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
        selection={"cooldown_key": f"{CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER}:default", "lane_key": None},
        attempt_record={},
        exc=exc,
        codex_failure_evidence_alias=None,
        kimi_failure_metadata_fn=lambda exc, candidate=None: None,
        classify_kimi_fn=lambda metadata: None,
        classify_retryable_fn=package.error_signals._classify_codex_auto_agent_retryable_exhaustion,
        grok_quota_fn=lambda exc, candidate=None: False,
        cooldown_seconds_fn=lambda exc, candidate=None: 60,
    )

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
        cooldown_seconds_fn=lambda exc, candidate=None: 60,
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
        lambda exc, *, candidate=None: "provider_terminal_error"
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
        classify_retryable_fn=lambda exc, candidate=None: "rate_limited",
        grok_quota_fn=lambda exc, candidate=None: True,
        cooldown_seconds_fn=lambda exc, candidate=None: 30.0,
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
        cooldown_seconds_fn=lambda exc, candidate=None: 60,
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
        classify_retryable_fn=lambda exc, candidate=None: (_ for _ in ()).throw(
            AssertionError("classification must short-circuit")
        ),
        grok_quota_fn=lambda exc, candidate=None: (_ for _ in ()).throw(
            AssertionError("quota classification must not run")
        ),
        cooldown_seconds_fn=lambda exc, candidate=None: (_ for _ in ()).throw(
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
async def test_candidate_loop_ineligible_falls_through_without_request_local_state(
    monkeypatch: pytest.MonkeyPatch,
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
    selection_kwargs: list[dict] = []

    async def _perform_candidate(
        *,
        candidate: dict,
        candidate_body: dict,
    ) -> object:
        provider_calls.append(candidate["model"])
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
    response = await candidate_loop.handle_alias_route(
        services,
        alias_family="codex",
        alias_model="aawm-test",
        request=request,
        prepared_request_body={},
        max_candidate_attempts=2,
        get_active_cooldown_state_fn=_no_active_cooldown,
        attempts_metadata_key="attempts",
        skipped_candidates_metadata_key="skipped",
        no_candidate_detail="no candidates",
        log_label="test",
    )

    assert response["candidate"] == "supported"
    assert provider_calls == ["unsupported", "supported"]
    assert selection_kwargs[0]["excluded_candidate_keys"] == frozenset()
    assert selection_kwargs[1]["excluded_candidate_keys"] == frozenset(
        {"openai:unsupported"}
    )
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
        if "zai_coding_plan/glm-5.3-flash" in key:
            return 30.0, "memory"
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
        }:
            return _IneligibleCandidateError()
        if model in {"cohere/north-mini-code-1-0", "big-pickle"}:
            return HTTPException(status_code=401, detail="generic provider auth failure")
        if model == "openrouter/owl-alpha":
            return HTTPException(
                status_code=429,
                detail={"error": {"code": "rate_limit_exceeded"}},
            )
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
        if model == "alibaba_token_plan/qwen3.6-flash":
            return HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "type": "invalid_request_error",
                        "code": "ModelNotFound",
                        "message": "Model not exist",
                    }
                },
            )
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
            "openrouter/owl-alpha",
            "deepseek-v4-flash-free",
            "big-pickle",
            "alibaba_token_plan/deepseek-v4-flash-0731",
            "cursor_agent/composer-2.5",
            "alibaba_token_plan/qwen3.6-flash",
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
            "openrouter/owl-alpha",
            "deepseek-v4-flash-free",
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
        assert len(claude_candidates) == 9
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
            "zai_coding_plan/glm-5.3-flash",
            "cohere/north-mini-code-1-0",
            "openrouter/cohere/north-mini-code:free",
            "openrouter/owl-alpha",
            "deepseek-v4-flash-free",
            "big-pickle",
            "alibaba_token_plan/deepseek-v4-flash-0731",
            "cursor_agent/composer-2.5",
            "alibaba_token_plan/qwen3.6-flash",
            "gpt-5.6-luna",
        ]
        assert [
            (candidate["terminal_disposition"], candidate["reason"])
            for candidate in inventory
        ] == [
            ("skipped", "cooldown"),
            ("attempted", "provider_terminal_error"),
            ("attempted", "candidate_deterministically_ineligible"),
            ("attempted", "rate_limited"),
            ("attempted", "provider_format_rejected"),
            ("attempted", "provider_terminal_error"),
            ("attempted", "candidate_deterministically_ineligible"),
            ("attempted", "upstream_timeout"),
            ("attempted", "candidate_unavailable"),
            ("attempted", "rate_limited"),
        ]
        assert inventory[-1]["reasoning_effort"] == "low"
        assert terminal_event["candidate_count"] == len(inventory) == 10
    finally:
        snapshot_select.set_active_routing_snapshot(previous_snapshot)
