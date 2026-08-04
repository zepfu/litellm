"""Wave 7 rollup.py module-local tests.

Covers: outgoing target mapping, model label, status classification,
status message, group-header resolution, and the full
_record_auto_agent_alias_route_status_rollup orchestration including
emit_aawm_route_status_event / record_aawm_route_rollup delegation,
turns=0, omission behavior, and candidate model expansion.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from litellm.proxy.pass_through_endpoints.aawm_alias_routing import rollup as rollup_mod
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.rollup import (
    _auto_agent_alias_model_rollup_label,
    _auto_agent_alias_route_rollup_status,
    _auto_agent_alias_route_status_message,
    _build_auto_agent_alias_rollup_group_header_label,
    _record_auto_agent_alias_route_status_rollup,
    _resolve_auto_agent_alias_route_rollup_group_header_label,
    _resolve_auto_agent_alias_route_rollup_outgoing_target,
    configure_rollup_runtime,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _inject_runtime():
    """Provide the cross-module seam and an isolated owner runtime.

    Importing ``llm_passthrough_endpoints`` calls ``rollup.install(lpe_globals)``,
    which rebinds each rollup function's ``__globals__`` to the LPE namespace.
    When that import happens before this module is collected, the names imported
    above reference LPE-globals functions, so ``@patch("...rollup.X")`` and
    ``configure_rollup_runtime`` (which mutate ``rollup``'s namespace) no longer
    affect the functions under test.

    Re-install the functions into ``rollup``'s own namespace so the owner tests
    exercise rollup-owned globals regardless of import order, refresh this
    module's imported references to the rebound objects, then restore the prior
    bindings on teardown so unrelated tests are unaffected.
    """
    host_fn_names = rollup_mod._HOST_FUNCTION_NAMES
    test_ns = globals()

    saved_rollup_fns = {name: rollup_mod.__dict__[name] for name in host_fn_names}
    saved_test_fns = {name: test_ns[name] for name in host_fn_names}
    saved_seam = rollup_mod._get_anthropic_adapter_access_log_target_label
    saved_host_globals = rollup_mod._host_globals

    rollup_mod.install(rollup_mod.__dict__)
    for name in host_fn_names:
        test_ns[name] = rollup_mod.__dict__[name]

    fake_label = MagicMock(return_value="fake.host/v1/path")
    configure_rollup_runtime(get_access_log_target_label=fake_label)
    yield fake_label

    rollup_mod.__dict__.update(saved_rollup_fns)
    test_ns.update(saved_test_fns)
    rollup_mod._get_anthropic_adapter_access_log_target_label = saved_seam
    rollup_mod._host_globals = saved_host_globals


# ---------------------------------------------------------------------------
# _resolve_auto_agent_alias_route_rollup_outgoing_target
# ---------------------------------------------------------------------------


class TestResolveOutgoingTarget:
    def test_known_route_family_codex_opencode_zen(self):
        result = _resolve_auto_agent_alias_route_rollup_outgoing_target(
            route_family="codex_opencode_zen_adapter",
        )
        assert result == "opencode.ai/zen/v1/chat/completions"

    def test_known_route_family_codex_openrouter(self):
        result = _resolve_auto_agent_alias_route_rollup_outgoing_target(
            route_family="codex_openrouter_completion_adapter",
        )
        assert result == "openrouter.ai/api/v1/chat/completions"

    def test_known_route_family_anthropic_opencode_zen_responses(self):
        result = _resolve_auto_agent_alias_route_rollup_outgoing_target(
            route_family="anthropic_opencode_zen_responses_adapter",
        )
        assert result == "opencode.ai/zen/v1/responses"

    def test_known_route_family_anthropic_opencode_zen_completion(self):
        result = _resolve_auto_agent_alias_route_rollup_outgoing_target(
            route_family="anthropic_opencode_zen_completion_adapter",
        )
        assert result == "opencode.ai/zen/v1/chat/completions"

    def test_unknown_route_family_returns_cleaned_value(self):
        result = _resolve_auto_agent_alias_route_rollup_outgoing_target(
            route_family="some_custom_adapter",
        )
        assert result == "some_custom_adapter"

    def test_none_route_family_returns_none(self):
        result = _resolve_auto_agent_alias_route_rollup_outgoing_target(
            route_family=None,
        )
        assert result is None

    def test_target_url_delegates_to_injected_seam(self, _inject_runtime):
        result = _resolve_auto_agent_alias_route_rollup_outgoing_target(
            route_family="codex_opencode_zen_adapter",
            target_url="https://example.com/v1/chat",
        )
        assert result == "fake.host/v1/path"
        _inject_runtime.assert_called_once_with("https://example.com/v1/chat")

    def test_target_url_httpx_url(self, _inject_runtime):
        url = httpx.URL("https://example.com/v1/responses")
        result = _resolve_auto_agent_alias_route_rollup_outgoing_target(
            route_family=None,
            target_url=url,
        )
        assert result == "fake.host/v1/path"
        _inject_runtime.assert_called_once_with(url)


# ---------------------------------------------------------------------------
# _auto_agent_alias_model_rollup_label
# ---------------------------------------------------------------------------


class TestModelRollupLabel:
    def test_model_and_alias_differ(self):
        event = {"model": "gpt-4o", "alias_model": "basic"}
        assert _auto_agent_alias_model_rollup_label(event) == "gpt-4o(basic)"

    def test_model_and_alias_same(self):
        event = {"model": "gpt-4o", "alias_model": "gpt-4o"}
        assert _auto_agent_alias_model_rollup_label(event) == "gpt-4o"

    def test_model_only(self):
        event = {"model": "gpt-4o"}
        assert _auto_agent_alias_model_rollup_label(event) == "gpt-4o"

    def test_alias_only(self):
        event = {"alias_model": "basic"}
        assert _auto_agent_alias_model_rollup_label(event) == "basic"

    def test_neither(self):
        assert _auto_agent_alias_model_rollup_label({}) is None


# ---------------------------------------------------------------------------
# _auto_agent_alias_route_rollup_status
# ---------------------------------------------------------------------------


class TestRollupStatus:
    def test_exhausted(self):
        event = {"event_type": "no_candidate_available"}
        assert _auto_agent_alias_route_rollup_status(event) == "Exhausted"

    def test_degraded_candidate_status(self):
        event = {"candidate_status": "auth_degraded_fallback"}
        assert _auto_agent_alias_route_rollup_status(event) == "Degraded"

    def test_degraded_selection_reason(self):
        event = {"selection_reason": "auth_degraded_no_valid_token"}
        assert _auto_agent_alias_route_rollup_status(event) == "Degraded"

    def test_retryable_no_cooldown_with_error(self):
        event = {"candidate_status": "retryable_no_cooldown", "error_status_code": 500}
        assert _auto_agent_alias_route_rollup_status(event) == "Failed"

    def test_retryable_no_cooldown_without_error(self):
        event = {"candidate_status": "retryable_no_cooldown"}
        assert _auto_agent_alias_route_rollup_status(event) is None

    def test_cooldown_scope_none_with_failure_class(self):
        event = {"cooldown_scope": "none", "failure_class": "timeout"}
        assert _auto_agent_alias_route_rollup_status(event) == "Failed"

    def test_cooldown_scope_none_without_error(self):
        event = {"cooldown_scope": "none"}
        assert _auto_agent_alias_route_rollup_status(event) is None

    def test_request_local_with_error(self):
        event = {"cooldown_scope": "request_local", "error_status_code": 429}
        assert _auto_agent_alias_route_rollup_status(event) == "Failed"

    def test_request_local_with_redispatch(self):
        event = {"cooldown_scope": "request_local", "redispatch_required": True}
        assert _auto_agent_alias_route_rollup_status(event) == "Failed"

    def test_request_local_clean(self):
        event = {"cooldown_scope": "request_local"}
        assert _auto_agent_alias_route_rollup_status(event) is None

    def test_cooldown_set(self):
        event = {"candidate_status": "cooldown_set"}
        assert _auto_agent_alias_route_rollup_status(event) == "Cooling Down"

    def test_terminal_in_flight_cooldown_set(self):
        event = {"candidate_status": "terminal_in_flight_cooldown_set"}
        assert _auto_agent_alias_route_rollup_status(event) == "Cooling Down"

    def test_skipped_cooldown(self):
        event = {"candidate_status": "skipped_cooldown"}
        assert _auto_agent_alias_route_rollup_status(event) == "Cooling Down"

    def test_skipped_prefix_with_cooldown_substring(self):
        event = {"candidate_status": "skipped_durable_cooldown"}
        assert _auto_agent_alias_route_rollup_status(event) == "Cooling Down"

    def test_skipped_prefix_auth_degraded_not_cooling(self):
        # auth_degraded takes priority
        event = {"candidate_status": "skipped_auth_degraded_cooldown"}
        assert _auto_agent_alias_route_rollup_status(event) == "Degraded"

    def test_cooldown_scope_candidate(self):
        event = {"cooldown_scope": "candidate"}
        assert _auto_agent_alias_route_rollup_status(event) == "Cooling Down"

    def test_redispatch_required_non_request_local(self):
        event = {"redispatch_required": True, "cooldown_scope": "global"}
        assert _auto_agent_alias_route_rollup_status(event) == "Cooling Down"

    def test_failure_class_rate_limited(self):
        event = {"failure_class": "rate_limited"}
        assert _auto_agent_alias_route_rollup_status(event) == "Cooling Down"

    def test_failure_class_capacity_exhausted(self):
        event = {"failure_class": "capacity_exhausted"}
        assert _auto_agent_alias_route_rollup_status(event) == "Cooling Down"

    def test_failure_class_transient_error(self):
        event = {"failure_class": "transient_error"}
        assert _auto_agent_alias_route_rollup_status(event) == "Cooling Down"

    def test_generic_error_status_code(self):
        event = {"error_status_code": 503}
        assert _auto_agent_alias_route_rollup_status(event) == "Failed"

    def test_generic_failure_class(self):
        event = {"failure_class": "auth_error"}
        assert _auto_agent_alias_route_rollup_status(event) == "Failed"

    def test_empty_event_returns_none(self):
        assert _auto_agent_alias_route_rollup_status({}) is None


# ---------------------------------------------------------------------------
# _auto_agent_alias_route_status_message
# ---------------------------------------------------------------------------


class TestStatusMessage:
    def test_source_error_included(self):
        event = {"source_error": "connection reset"}
        msg = _auto_agent_alias_route_status_message(event)
        assert "source_error=connection reset" in msg

    def test_structured_fields(self):
        event = {
            "failure_class": "timeout",
            "error_type": "ReadTimeout",
            "error_code": "ETIMEDOUT",
            "error_status_code": 504,
            "candidate_status": "cooldown_set",
            "selection_reason": "primary",
        }
        msg = _auto_agent_alias_route_status_message(event)
        assert "failure_class=timeout" in msg
        assert "error_type=ReadTimeout" in msg
        assert "error_code=ETIMEDOUT" in msg
        assert "error_status_code=504" in msg
        assert "candidate_status=cooldown_set" in msg
        assert "selection_reason=primary" in msg

    def test_error_tokens_truncated_to_5(self):
        event = {"error_tokens": [1, 2, 3, 4, 5, 6, 7]}
        msg = _auto_agent_alias_route_status_message(event)
        assert "error_tokens=1,2,3,4,5" in msg
        assert "6" not in msg.split("error_tokens=")[1]

    def test_empty_event_fallback(self):
        msg = _auto_agent_alias_route_status_message({})
        assert msg == "route status changed"


# ---------------------------------------------------------------------------
# _build_auto_agent_alias_rollup_group_header_label
# ---------------------------------------------------------------------------


class TestBuildGroupHeaderLabel:
    @patch(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing.rollup.build_aawm_route_rollup_group_header_label",
    )
    def test_delegates_to_aawm_route_logging(self, mock_build):
        mock_build.return_value = "repo@host"
        result = _build_auto_agent_alias_rollup_group_header_label(
            repository="my/repo",
            client_product_label="Codex/1.0",
            host_name="myhost",
        )
        assert result == "repo@host"
        mock_build.assert_called_once_with(
            repository="my/repo",
            client_product_label="Codex/1.0",
            host_name="myhost",
        )


# ---------------------------------------------------------------------------
# _resolve_auto_agent_alias_route_rollup_group_header_label
# ---------------------------------------------------------------------------


class TestResolveGroupHeaderLabel:
    def test_no_label_returns_none(self):
        assert _resolve_auto_agent_alias_route_rollup_group_header_label({}) is None

    def test_label_with_at_sign_unchanged(self):
        event = {"rollup_group_header_label": "repo@host", "host_name": "other"}
        assert _resolve_auto_agent_alias_route_rollup_group_header_label(event) == "repo@host"

    def test_label_without_host_unchanged(self):
        event = {"rollup_group_header_label": "repo"}
        assert _resolve_auto_agent_alias_route_rollup_group_header_label(event) == "repo"

    def test_label_appends_host(self):
        event = {"rollup_group_header_label": "repo", "host_name": "myhost"}
        assert _resolve_auto_agent_alias_route_rollup_group_header_label(event) == "repo@myhost"


# ---------------------------------------------------------------------------
# _record_auto_agent_alias_route_status_rollup (orchestration)
# ---------------------------------------------------------------------------


class TestRecordRouteStatusRollup:
    def _make_event(self, **overrides: Any) -> dict[str, Any]:
        base: dict[str, Any] = {
            "event_type": "no_candidate_available",
            "alias_model": "basic",
            "model": "gpt-4o",
            "rollup_group_header_label": "myrepo",
            "host_name": "myhost",
            "incoming_endpoint": "/v1/chat/completions",
            "route_family": "codex_opencode_zen_adapter",
        }
        base.update(overrides)
        return base

    @patch(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing.rollup.record_aawm_route_rollup",
    )
    @patch(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing.rollup.emit_aawm_route_status_event",
    )
    def test_full_happy_path(self, mock_emit, mock_record):
        event = self._make_event()
        _record_auto_agent_alias_route_status_rollup(event)

        mock_emit.assert_called_once_with(
            alias_model="basic",
            model_label="gpt-4o",
            status="Exhausted",
            message="route status changed",
        )
        mock_record.assert_called_once_with(
            group_header_label="myrepo@myhost",
            incoming_endpoint="/v1/chat/completions",
            outgoing_target="opencode.ai/zen/v1/chat/completions",
            model_label="gpt-4o(basic)",
            turns=0,
            status="Exhausted",
            message=None,
        )

    @patch(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing.rollup.record_aawm_route_rollup",
    )
    @patch(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing.rollup.emit_aawm_route_status_event",
    )
    def test_status_none_omits_all(self, mock_emit, mock_record):
        event = self._make_event(event_type="selection", candidate_status="ok")
        _record_auto_agent_alias_route_status_rollup(event)
        mock_emit.assert_not_called()
        mock_record.assert_not_called()

    @patch(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing.rollup.record_aawm_route_rollup",
    )
    @patch(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing.rollup.emit_aawm_route_status_event",
    )
    def test_no_model_labels_omits_all(self, mock_emit, mock_record):
        event = self._make_event(model=None, alias_model=None)
        _record_auto_agent_alias_route_status_rollup(event)
        mock_emit.assert_not_called()
        mock_record.assert_not_called()

    @patch(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing.rollup.record_aawm_route_rollup",
    )
    @patch(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing.rollup.emit_aawm_route_status_event",
    )
    def test_no_group_header_skips_record_but_emits(self, mock_emit, mock_record):
        event = self._make_event(rollup_group_header_label=None)
        _record_auto_agent_alias_route_status_rollup(event)
        mock_emit.assert_called_once()
        mock_record.assert_not_called()

    @patch(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing.rollup.record_aawm_route_rollup",
    )
    @patch(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing.rollup.emit_aawm_route_status_event",
    )
    def test_no_incoming_endpoint_skips_record_but_emits(self, mock_emit, mock_record):
        event = self._make_event(incoming_endpoint=None)
        _record_auto_agent_alias_route_status_rollup(event)
        mock_emit.assert_called_once()
        mock_record.assert_not_called()

    @patch(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing.rollup.record_aawm_route_rollup",
    )
    @patch(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing.rollup.emit_aawm_route_status_event",
    )
    def test_candidates_expand_model_labels(self, mock_emit, mock_record):
        event = self._make_event(
            candidates=[
                {"model": "claude-sonnet-4-20250514"},
                {"model": "gpt-4o"},  # duplicate of primary
                {"model": "grok-4"},
            ],
        )
        _record_auto_agent_alias_route_status_rollup(event)

        # 3 unique labels: gpt-4o(alias), claude-sonnet(alias), grok-4(alias)
        assert mock_emit.call_count == 3
        assert mock_record.call_count == 3

        emitted_labels = [c.kwargs["model_label"] for c in mock_emit.call_args_list]
        assert emitted_labels == ["gpt-4o", "claude-sonnet-4-20250514", "grok-4"]

    @patch(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing.rollup.record_aawm_route_rollup",
    )
    @patch(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing.rollup.emit_aawm_route_status_event",
    )
    def test_outgoing_target_explicit_overrides_route_family(self, mock_emit, mock_record):
        event = self._make_event(outgoing_target="custom.endpoint/v1")
        _record_auto_agent_alias_route_status_rollup(event)
        assert mock_record.call_args.kwargs["outgoing_target"] == "custom.endpoint/v1"

    @patch(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing.rollup.record_aawm_route_rollup",
    )
    @patch(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing.rollup.emit_aawm_route_status_event",
    )
    def test_outgoing_target_fallback_candidate_selection(self, mock_emit, mock_record):
        event = self._make_event(route_family=None, outgoing_target=None)
        _record_auto_agent_alias_route_status_rollup(event)
        assert mock_record.call_args.kwargs["outgoing_target"] == "candidate_selection"

    @patch(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing.rollup.record_aawm_route_rollup",
    )
    @patch(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing.rollup.emit_aawm_route_status_event",
    )
    def test_source_error_in_record_message(self, mock_emit, mock_record):
        event = self._make_event(source_error="upstream 502")
        _record_auto_agent_alias_route_status_rollup(event)
        assert mock_record.call_args.kwargs["message"] == "upstream 502"

    @patch(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing.rollup.record_aawm_route_rollup",
    )
    @patch(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing.rollup.emit_aawm_route_status_event",
    )
    def test_turns_always_zero(self, mock_emit, mock_record):
        event = self._make_event()
        _record_auto_agent_alias_route_status_rollup(event)
        assert mock_record.call_args.kwargs["turns"] == 0

    @patch(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing.rollup.record_aawm_route_rollup",
    )
    @patch(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing.rollup.emit_aawm_route_status_event",
    )
    def test_candidate_model_same_as_alias_not_suffixed(self, mock_emit, mock_record):
        event = self._make_event(
            model=None,
            alias_model="gpt-4o",
            candidates=[{"model": "gpt-4o"}],
        )
        _record_auto_agent_alias_route_status_rollup(event)
        # model == alias_model so no suffix
        recorded_labels = [c.kwargs["model_label"] for c in mock_record.call_args_list]
        assert recorded_labels == ["gpt-4o"]

    @patch(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing.rollup.record_aawm_route_rollup",
    )
    @patch(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing.rollup.emit_aawm_route_status_event",
    )
    def test_non_dict_candidates_skipped(self, mock_emit, mock_record):
        event = self._make_event(candidates=["not-a-dict", 42, None])
        _record_auto_agent_alias_route_status_rollup(event)
        # Only primary model label
        assert mock_emit.call_count == 1
        assert mock_record.call_count == 1


# ---------------------------------------------------------------------------
# Signature pinning
# ---------------------------------------------------------------------------


class TestSignaturePinning:
    def test_configure_rollup_runtime_accepts_keyword_only(self):
        import inspect

        sig = inspect.signature(configure_rollup_runtime)
        params = list(sig.parameters.values())
        assert all(p.kind == inspect.Parameter.KEYWORD_ONLY for p in params)
        assert "get_access_log_target_label" in sig.parameters

    def test_resolve_outgoing_target_keyword_only(self):
        import inspect

        sig = inspect.signature(_resolve_auto_agent_alias_route_rollup_outgoing_target)
        params = list(sig.parameters.values())
        assert all(p.kind == inspect.Parameter.KEYWORD_ONLY for p in params)
        assert "route_family" in sig.parameters
        assert "target_url" in sig.parameters

    def test_build_group_header_keyword_only(self):
        import inspect

        sig = inspect.signature(_build_auto_agent_alias_rollup_group_header_label)
        params = list(sig.parameters.values())
        assert all(p.kind == inspect.Parameter.KEYWORD_ONLY for p in params)
        assert set(sig.parameters.keys()) == {"repository", "client_product_label", "host_name"}

    def test_record_rollup_is_sync(self):
        import inspect

        assert not inspect.iscoroutinefunction(_record_auto_agent_alias_route_status_rollup)

    def test_status_functions_are_sync(self):
        import inspect

        for fn in (
            _auto_agent_alias_route_rollup_status,
            _auto_agent_alias_model_rollup_label,
            _auto_agent_alias_route_status_message,
            _resolve_auto_agent_alias_route_rollup_group_header_label,
        ):
            assert not inspect.iscoroutinefunction(fn)
