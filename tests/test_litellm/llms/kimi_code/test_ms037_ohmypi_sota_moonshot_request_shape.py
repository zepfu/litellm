"""MS-037: Ohmypi ``sota-moonshot`` request-shape must not kill the alias.

Ohmypi TUI traffic to alias ``sota-moonshot`` (``kimi_code/k3``, YAML
``reasoning_effort: max``) currently dies as ``agent_alias_no_candidate`` /
``kimi_code_no_cooldown`` / HTTP 502 when Managed Kimi returns a request-shape
400. These tests lock the sanitization + YAML-max + bounded-400 contracts.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from starlette.requests import Request
from starlette.responses import Response

from litellm.llms.kimi_code.adapters.adapter import (
    _apply_kimi_reasoning_effort,
    _extract_explicit_responses_reasoning_effort,
    prepare_codex_kimi_chat_completions_adapter_route,
)
from litellm.proxy.pass_through_endpoints import (
    llm_passthrough_endpoints as lpe,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    admission,
    cooldown_apply,
    cooldown_state,
    error_signals,
    snapshot_select,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.candidate_loop import (
    handle_alias_route,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.interfaces import (
    AliasRouteServices,
    CooldownPublicationPlan,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_startup import (
    DEFAULT_CONFIG_DIR,
    compile_directory,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
    alias_routing_state,
)


def _request() -> MagicMock:
    request = MagicMock()
    request.headers = {}
    request.scope = {}
    return request


def _starlette_request(path: str = "/v1/responses") -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 43123),
            "server": ("testserver", 80),
        }
    )


def _ohmypi_sota_moonshot_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": "sota-moonshot",
        "max_output_tokens": 64000,
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "ping"}],
            }
        ],
        "tools": [
            {
                "type": "function",
                "name": "read_file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            }
        ],
        "stream": False,
        "litellm_metadata": {"session_id": "ohmypi-sota-moonshot-session"},
    }
    body.update(overrides)
    return body


def _sota_moonshot_candidate() -> dict[str, Any]:
    return {
        "provider": "kimi_code",
        "model": "kimi_code/k3",
        "route_family": "codex_kimi_chat_completions_adapter",
        "priority": 100,
        "last_resort": False,
        "reasoning_effort": "max",
    }


def _sota_moonshot_selection() -> dict[str, Any]:
    candidate = _sota_moonshot_candidate()
    return {
        "candidate": candidate,
        "lane_key": "kimi_code:kimi_code/k3",
        "cooldown_key": "kimi_code:kimi_code/k3:kimi_code_managed_account",
        "alias_model": "sota-moonshot",
        "selection_reason": "priority",
        "skipped": [],
    }


def _malformed_kimi_probe_metadata(*, status_code: int = 400) -> dict[str, Any]:
    return {
        "kind": "malformed",
        "scope": "none",
        "upstream_id": "k3",
        "metadata_gate": "none",
        "status_code": status_code,
        "trace_id": "trace-ohmypi-ms037",
        "reset_reason": "malformed_provider_response",
    }


class _KimiProbeFailure(RuntimeError):
    def __init__(
        self,
        *,
        status_code: int,
        message: str,
        metadata: dict[str, Any],
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.headers = {"x-trace-id": "trace-ohmypi-ms037"}
        self.kimi_code_probe_failure_metadata = metadata
        self.detail = detail or {
            "error": {
                "message": "Managed Kimi Code rejected the request shape.",
                "type": "invalid_request_error",
                "code": "kimi_code_invalid_request",
            }
        }


def _assert_ohmypi_max_output_tokens_dropped(completion_kwargs: dict[str, Any]) -> None:
    assert completion_kwargs.get("max_tokens") != 64000
    assert "max_output_tokens" not in completion_kwargs
    max_tokens = completion_kwargs.get("max_tokens")
    if max_tokens is not None:
        assert max_tokens != 64000


@pytest.mark.asyncio
async def test_should_drop_ohmypi_max_output_tokens_on_kimi_prepare() -> None:
    body = _ohmypi_sota_moonshot_body(model="kimi_code/k3")

    public_plan = await prepare_codex_kimi_chat_completions_adapter_route(
        request=MagicMock(),
        adapter_model="kimi_code/k3",
        prepared_request_body=dict(body),
    )
    public_kwargs = public_plan.perform_kwargs["completion_kwargs"]
    _assert_ohmypi_max_output_tokens_dropped(public_kwargs)
    assert "max_output_tokens" not in public_plan.prepared_request_body

    host_plan = await lpe._prepare_codex_kimi_chat_completions_adapter_route(
        request=_request(),
        prepared_request_body=dict(body),
        adapter_model="kimi_code/k3",
    )
    host_kwargs = host_plan.perform_kwargs["completion_kwargs"]
    _assert_ohmypi_max_output_tokens_dropped(host_kwargs)
    assert "max_output_tokens" not in host_plan.prepared_request_body


@pytest.mark.asyncio
async def test_should_apply_yaml_max_when_ohmypi_sends_none_effort() -> None:
    assert (
        _extract_explicit_responses_reasoning_effort({"reasoning_effort": "none"})
        is None
    )
    assert (
        _extract_explicit_responses_reasoning_effort(
            {"reasoning": {"effort": "none"}}
        )
        is None
    )
    assert (
        _extract_explicit_responses_reasoning_effort({"reasoning_effort": "off"})
        is None
    )
    assert (
        _extract_explicit_responses_reasoning_effort(
            {"reasoning": {"effort": "off"}}
        )
        is None
    )

    leftover_kwargs: dict[str, Any] = {"reasoning_effort": "none"}
    _apply_kimi_reasoning_effort(
        request_body={"reasoning": {"effort": "none"}, "reasoning_effort": "none"},
        upstream_model="k3",
        forced_effort=None,
        completion_kwargs=leftover_kwargs,
    )
    assert leftover_kwargs.get("reasoning_effort") != "none"

    body = _ohmypi_sota_moonshot_body(
        reasoning={"effort": "none"},
        reasoning_effort="none",
    )
    rewritten = lpe._add_codex_auto_agent_alias_metadata(
        body,
        request=_starlette_request(),
        selection=_sota_moonshot_selection(),
        attempts=[],
    )
    assert rewritten.get("reasoning_effort") != "none"
    assert (rewritten.get("reasoning") or {}).get("effort") == "max"

    plan = await prepare_codex_kimi_chat_completions_adapter_route(
        request=MagicMock(),
        adapter_model="kimi_code/k3",
        prepared_request_body=rewritten,
    )
    completion_kwargs = plan.perform_kwargs["completion_kwargs"]
    assert completion_kwargs["reasoning_effort"] == "max"


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", (400, 422))
async def test_should_not_cooldown_malformed_kimi_request_shape(status_code) -> None:
    raw_provider_detail = "text content is empty: raw-provider-detail"
    upstream_failure = _KimiProbeFailure(
        status_code=status_code,
        message=raw_provider_detail,
        metadata=_malformed_kimi_probe_metadata(status_code=status_code),
        detail=None,
    )
    completion_mock = AsyncMock(side_effect=upstream_failure)

    with (
        patch("litellm.acompletion", new=completion_mock),
        pytest.raises(_KimiProbeFailure) as caught,
    ):
        await lpe._handle_codex_kimi_chat_completions_adapter_route(
            endpoint="/v1/responses",
            request=_request(),
            fastapi_response=MagicMock(spec=Response),
            user_api_key_dict=MagicMock(),
            prepared_request_body={
                "model": "kimi_code/k3",
                "input": "hello",
                "stream": False,
            },
            adapter_model="kimi_code/k3",
            use_alias_candidate_probe=True,
        )

    metadata = caught.value.kimi_code_probe_failure_metadata
    assert metadata["kind"] == "malformed"
    assert metadata["scope"] == "none"
    assert (
        error_signals._classify_kimi_code_auto_agent_probe_failure(metadata)
        == "kimi_code_no_cooldown"
    )
    assert (
        error_signals._get_codex_auto_agent_candidate_cooldown_scope(
            "kimi_code_no_cooldown",
            candidate=_sota_moonshot_candidate(),
            kimi_failure_metadata=metadata,
        )
        == "none"
    )

    candidate_key = "kimi_code:kimi_code/k3:kimi_code_managed_account"
    persist_mock = AsyncMock()
    with patch.object(
        cooldown_state,
        "_set_codex_auto_agent_cooldown",
        new=persist_mock,
    ):
        applied_scope = await cooldown_apply._apply_auto_agent_alias_cooldown(
            request=_starlette_request(),
            candidate=_sota_moonshot_candidate(),
            lane_key="kimi_code:kimi_code/k3",
            selected_cooldown_key=candidate_key,
            cooldown_seconds=60.0,
            error_class="kimi_code_no_cooldown",
            set_candidate_cooldown=cooldown_state._set_codex_auto_agent_cooldown,
            kimi_failure_metadata=metadata,
        )

    assert applied_scope == "none"
    persist_mock.assert_not_called()
    assert (
        await cooldown_state._get_codex_auto_agent_active_cooldown_seconds(
            candidate_key
        )
        == 0
    )
    assert (
        await cooldown_state._get_codex_auto_agent_active_cooldown_seconds(
            error_signals._get_kimi_code_managed_account_cooldown_key()
        )
        == 0
    )
    assert raw_provider_detail not in str(caught.value.detail)


@pytest.fixture
def _reset_sota_moonshot_alias_state():
    previous_snapshot = snapshot_select.get_active_routing_snapshot()
    snapshot_select.set_active_routing_snapshot(compile_directory(DEFAULT_CONFIG_DIR))
    alias_routing_state.codex.cooldown_until_monotonic_by_key.clear()
    alias_routing_state.anthropic.cooldown_until_monotonic_by_key.clear()
    alias_routing_state.codex.cooldown_negative_until_monotonic_by_key.clear()
    alias_routing_state.anthropic.cooldown_negative_until_monotonic_by_key.clear()
    alias_routing_state.codex.session_affinity_by_key.clear()
    alias_routing_state.anthropic.session_affinity_by_key.clear()
    yield
    alias_routing_state.codex.cooldown_until_monotonic_by_key.clear()
    alias_routing_state.anthropic.cooldown_until_monotonic_by_key.clear()
    alias_routing_state.codex.cooldown_negative_until_monotonic_by_key.clear()
    alias_routing_state.anthropic.cooldown_negative_until_monotonic_by_key.clear()
    alias_routing_state.codex.session_affinity_by_key.clear()
    alias_routing_state.anthropic.session_affinity_by_key.clear()
    snapshot_select.set_active_routing_snapshot(previous_snapshot)


@pytest.mark.asyncio
async def test_should_return_bounded_400_not_502_when_sota_moonshot_malformed_is_only_candidate(
    _reset_sota_moonshot_alias_state,
) -> None:
    admission.reset_admission_state_for_tests()
    alias_routing_state.reset_for_tests()
    request = _starlette_request()
    body = _ohmypi_sota_moonshot_body(
        reasoning={"effort": "none"},
        reasoning_effort="none",
    )
    selection = _sota_moonshot_selection()
    selection.update(
        {
            "session_key": "ohmypi-sota-moonshot-session",
            "in_flight_session": False,
            "cooldown_seconds": 0.0,
            "cooldown_state_source": "none",
        }
    )
    probe_failure = _KimiProbeFailure(
        status_code=400,
        message="text content is empty: raw-provider-detail",
        metadata=_malformed_kimi_probe_metadata(),
    )
    perform_mock = AsyncMock(side_effect=probe_failure)

    async def _select(*, request, request_body):
        _ = request, request_body
        return dict(selection)

    async def _noop_persist(*, keys, seconds):
        _ = keys, seconds
        return None

    def _publish(*, keys, seconds):
        _ = keys, seconds
        return None

    async def _set_affinity(session_key, candidate):
        _ = session_key, candidate
        return None

    def _add_meta(body, *, request, selection, attempts):
        _ = request, selection
        out = dict(body)
        out["litellm_metadata"] = {"attempts": attempts}
        return out

    def _resolve(**kwargs):
        return CooldownPublicationPlan(
            applied_scope="none",
            duration_seconds=0.0,
            kimi_failure_metadata=kwargs.get("kimi_failure_metadata"),
        )

    def _raise_redispatch(**kwargs):
        raise AssertionError("malformed scope=none must not redispatch")

    services = AliasRouteServices(
        select_candidate_fn=_select,
        perform_candidate_request_fn=perform_mock,
        resolve_cooldown_publication_fn=_resolve,
        publish_cooldown_memory_fn=_publish,
        persist_cooldown_fn=_noop_persist,
        set_session_affinity_fn=_set_affinity,
        add_alias_metadata_fn=_add_meta,
        raise_redispatch_fn=_raise_redispatch,
    )
    sa = MagicMock()
    sa.resolve_canonical_session_identity.return_value = "sess"
    sa.get_request_codex_auto_review_parent_session_identity.return_value = None
    sa.build_session_owner_attributes.return_value = {"provider": "kimi_code"}
    sa.ensure_session_owner_guard_for_request = AsyncMock(
        return_value=SimpleNamespace(
            decision=SimpleNamespace(value="no_session"),
            reservation_token=None,
            held_reservation=False,
            provenance=None,
        )
    )
    sa.get_request_session_owner_lease.return_value = None
    sa.finalize_session_owner_lease_on_success = AsyncMock(return_value=None)
    sa.finalize_session_owner_lease_on_failure = AsyncMock(return_value=None)

    with patch(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing.candidate_loop._session_affinity_mod",
        return_value=sa,
    ):
        with pytest.raises(HTTPException) as caught:
            await handle_alias_route(
                services,
                alias_family="codex",
                alias_model="sota-moonshot",
                request=request,
                prepared_request_body=body,
                max_candidate_attempts=1,
                get_active_cooldown_state_fn=AsyncMock(return_value=(0.0, "none")),
                attempts_metadata_key="codex_auto_agent_attempts",
                skipped_candidates_metadata_key="codex_auto_agent_skipped_candidates",
                no_candidate_detail=(
                    "No Codex auto-agent alias candidates were available."
                ),
                log_label="Codex",
            )

    assert perform_mock.await_count == 1
    assert caught.value.status_code == 400
    error = caught.value.detail["error"]
    assert error["code"] == "kimi_code_invalid_request"
    assert error["type"] == "invalid_request_error"
    assert caught.value.status_code != 502
    assert error["code"] != "all_candidates_unavailable"
    assert error["type"] != "kimi_code_no_cooldown"
    assert error["type"] != "agent_alias_no_candidate"

    auth_failure = HTTPException(
        status_code=401,
        detail={
            "error": {
                "message": (
                    "Managed Kimi Code authentication requires the shared "
                    "credential to be refreshed."
                ),
                "type": "authentication_error",
                "code": "kimi_code_auth_refresh_required",
            }
        },
    )
    assert auth_failure.status_code == 401
    assert auth_failure.detail["error"]["code"] == "kimi_code_auth_refresh_required"
