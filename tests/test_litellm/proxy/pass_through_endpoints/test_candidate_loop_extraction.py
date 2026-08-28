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

from fastapi import HTTPException
import pytest
from starlette.requests import Request

from litellm.proxy.pass_through_endpoints import aawm_alias_routing as package
from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import candidate_loop
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
