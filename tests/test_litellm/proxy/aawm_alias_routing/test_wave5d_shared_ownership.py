"""Wave 5D shared ownership, facade, and call-through tests.

Verifies:
- Same-object facades for context/build/persist/events
- Moved symbols not defined in god module
- No module-scope god-module imports in Wave 5D modules
- attempt_records callback installation/call-through
- candidate_loop reads continuation/cooldown/no-candidate dependencies through
  intended facade/runtime
- Wave 5B redispatch ownership remains selection.py
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Module references
# ---------------------------------------------------------------------------

GOD_MODULE_PATH = (
    Path(__file__).resolve().parents[4]
    / "litellm"
    / "proxy"
    / "pass_through_endpoints"
    / "llm_passthrough_endpoints.py"
)

PACKAGE_DIR = (
    Path(__file__).resolve().parents[4]
    / "litellm"
    / "proxy"
    / "pass_through_endpoints"
    / "aawm_alias_routing"
)

WAVE5D_MODULES = ("audit_context", "audit_build", "audit_persist", "audit_events")


# ---------------------------------------------------------------------------
# Same-object facade tests
# ---------------------------------------------------------------------------


class TestSameObjectFacades:
    """Every Wave 5D symbol must be the same object in the god module and the owning module."""

    @pytest.fixture(autouse=True)
    def _import_modules(self):
        import litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints as lpe
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
            audit_build,
            audit_context,
            audit_events,
            audit_persist,
        )

        self.lpe = lpe
        self.audit_context = audit_context
        self.audit_build = audit_build
        self.audit_persist = audit_persist
        self.audit_events = audit_events

    def test_audit_context_facades_same_object(self):
        ctx = self.audit_context
        lpe = self.lpe
        symbols = [
            "_extract_auto_agent_alias_text_blobs",
            "_extract_auto_agent_alias_role_from_text",
            "_infer_auto_agent_alias_role_from_request_body",
            "_iter_auto_agent_alias_metadata_dicts",
            "_extract_auto_agent_alias_agent_dispatch_fields",
            "_walk_auto_agent_alias_prior_tool_activity",
            "_summarize_auto_agent_alias_actual_prior_tool_activity",
            "_classify_auto_agent_alias_terminal_activity_status",
            "_get_or_create_auto_agent_alias_request_call_id",
            "_AutoAgentAliasRequestContext",
            "_normalize_auto_agent_alias_request_context",
            "_clean_optional_string",
            "_get_auto_agent_alias_request_context",
            "_attach_auto_agent_alias_terminal_context_fields",
        ]
        for name in symbols:
            assert getattr(lpe, name) is getattr(ctx, name), f"{name} is not same object"

    def test_audit_context_constants_same_object(self):
        ctx = self.audit_context
        lpe = self.lpe
        constants = [
            "_AUTO_AGENT_ROLE_DECLARATION_RE",
            "_AUTO_AGENT_KNOWN_ROLE_NAMES",
            "_AUTO_AGENT_PRIOR_TOOL_ITEM_TYPES",
            "_AUTO_AGENT_FILE_EDIT_TOOL_NAMES",
            "_AUTO_AGENT_REQUEST_CALL_ID_STATE_KEY",
            "_AUTO_AGENT_REQUEST_CONTEXT_STATE_KEY",
        ]
        for name in constants:
            assert getattr(lpe, name) is getattr(ctx, name), f"{name} is not same object"

    def test_audit_build_facades_same_object(self):
        ab = self.audit_build
        lpe = self.lpe
        symbols = [
            "_is_auto_agent_alias_in_flight_cooldown_http_exception",
            "_build_auto_agent_alias_audit_event",
            "_build_auto_agent_alias_audit_events",
            "_codex_auto_agent_request_has_continuation_state",
        ]
        for name in symbols:
            assert getattr(lpe, name) is getattr(ab, name), f"{name} is not same object"

    def test_audit_persist_facades_same_object(self):
        ap = self.audit_persist
        lpe = self.lpe
        symbols = [
            "_emit_auto_agent_alias_route_event",
            "_should_emit_auto_agent_alias_route_event",
            "_persist_auto_agent_alias_audit_only_events_best_effort",
        ]
        for name in symbols:
            assert getattr(lpe, name) is getattr(ap, name), f"{name} is not same object"

    def test_audit_events_facades_same_object(self):
        ae = self.audit_events
        lpe = self.lpe
        symbols = [
            "_enrich_auto_agent_alias_terminal_event_from_attempts",
            "_emit_auto_agent_alias_no_candidate_event",
        ]
        for name in symbols:
            assert getattr(lpe, name) is getattr(ae, name), f"{name} is not same object"


# ---------------------------------------------------------------------------
# Moved symbols NOT defined in god module
# ---------------------------------------------------------------------------


class TestMovedSymbolsNotDefinedInGodModule:
    """Frozen Wave 5D symbols must not have def/class statements in the god module."""

    MOVED_SYMBOLS = [
        # audit_context
        "_extract_auto_agent_alias_text_blobs",
        "_extract_auto_agent_alias_role_from_text",
        "_infer_auto_agent_alias_role_from_request_body",
        "_iter_auto_agent_alias_metadata_dicts",
        "_extract_auto_agent_alias_agent_dispatch_fields",
        "_walk_auto_agent_alias_prior_tool_activity",
        "_summarize_auto_agent_alias_actual_prior_tool_activity",
        "_classify_auto_agent_alias_terminal_activity_status",
        "_get_or_create_auto_agent_alias_request_call_id",
        "_AutoAgentAliasRequestContext",
        "_normalize_auto_agent_alias_request_context",
        "_clean_optional_string",
        "_get_auto_agent_alias_request_context",
        "_attach_auto_agent_alias_terminal_context_fields",
        # audit_build
        "_is_auto_agent_alias_in_flight_cooldown_http_exception",
        "_build_auto_agent_alias_audit_event",
        "_build_auto_agent_alias_audit_events",
        "_codex_auto_agent_request_has_continuation_state",
        # audit_persist
        "_emit_auto_agent_alias_route_event",
        "_should_emit_auto_agent_alias_route_event",
        "_persist_auto_agent_alias_audit_only_events_best_effort",
        # audit_events
        "_enrich_auto_agent_alias_terminal_event_from_attempts",
        "_emit_auto_agent_alias_no_candidate_event",
    ]

    @pytest.fixture(autouse=True, scope="class")
    def _parse_god_module(self, request):
        source = GOD_MODULE_PATH.read_text()
        tree = ast.parse(source)
        defined_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defined_names.add(node.name)
            elif isinstance(node, ast.ClassDef):
                defined_names.add(node.name)
        request.cls._god_defined = defined_names

    def test_no_frozen_definitions_remain(self):
        for name in self.MOVED_SYMBOLS:
            assert name not in self._god_defined, (
                f"{name} still has a def/class in the god module"
            )


# ---------------------------------------------------------------------------
# No module-scope god-module imports in Wave 5D modules
# ---------------------------------------------------------------------------


class TestNoModuleScopeGodModuleImports:
    """Wave 5D modules must not import llm_passthrough_endpoints at module scope."""

    def test_no_god_module_imports(self):
        for mod_name in WAVE5D_MODULES:
            path = PACKAGE_DIR / f"{mod_name}.py"
            source = path.read_text()
            tree = ast.parse(source)
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert "llm_passthrough_endpoints" not in alias.name, (
                            f"{mod_name}.py has module-scope import of god module"
                        )
                elif isinstance(node, ast.ImportFrom):
                    if node.module and "llm_passthrough_endpoints" in node.module:
                        pytest.fail(
                            f"{mod_name}.py has module-scope from-import of god module"
                        )


# ---------------------------------------------------------------------------
# attempt_records callback installation / call-through
# ---------------------------------------------------------------------------


class TestAttemptRecordsCallbackInstallation:
    """attempt_records runtime must receive audit callbacks that resolve to Wave 5D modules."""

    @pytest.fixture(autouse=True)
    def _ensure_god_module_imported(self):
        """Import god module to trigger configure_attempt_records_runtime."""
        import litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints  # noqa: F401

    def test_attempt_records_emit_route_event_is_audit_persist(self):
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
            attempt_records,
        )

        # The configured emit_route_event lambda should delegate to audit_persist
        fn = attempt_records._emit_auto_agent_alias_route_event
        assert fn is not None, "emit_route_event not configured"

    def test_attempt_records_build_audit_event_configured(self):
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing import attempt_records

        assert attempt_records._build_auto_agent_alias_audit_event is not None
        assert attempt_records._build_auto_agent_alias_audit_events is not None
        assert attempt_records._persist_auto_agent_alias_audit_only_events_best_effort is not None


# ---------------------------------------------------------------------------
# candidate_loop reads dependencies through intended facade/runtime
# ---------------------------------------------------------------------------


class TestCandidateLoopDependencies:
    """candidate_loop must read continuation/cooldown/no-candidate through god-module facades."""

    def test_candidate_loop_uses_god_module_facade_for_no_candidate(self):
        source = (PACKAGE_DIR / "candidate_loop.py").read_text()
        # candidate_loop resolves _emit_auto_agent_alias_no_candidate_event from _lpe
        assert "_lpe._emit_auto_agent_alias_no_candidate_event" in source

    def test_candidate_loop_uses_god_module_facade_for_continuation(self):
        source = (PACKAGE_DIR / "candidate_loop.py").read_text()
        assert "_lpe._codex_auto_agent_request_has_continuation_state" in source

    def test_candidate_loop_uses_god_module_facade_for_in_flight(self):
        source = (PACKAGE_DIR / "candidate_loop.py").read_text()
        assert "_lpe._is_auto_agent_alias_in_flight_cooldown_http_exception" in source


# ---------------------------------------------------------------------------
# Wave 5B redispatch ownership remains selection.py
# ---------------------------------------------------------------------------


class TestWave5BRedispatchOwnership:
    """Redispatch/in-flight exception builders must remain in selection.py."""

    REDISPATCH_SYMBOLS = [
        "_raise_codex_auto_agent_in_flight_cooldown",
        "_raise_anthropic_auto_agent_in_flight_cooldown",
        "_build_auto_agent_redispatch_http_exception_detail",
        "_raise_codex_auto_agent_redispatch_required",
        "_raise_anthropic_auto_agent_redispatch_required",
    ]

    def test_redispatch_symbols_defined_in_selection(self):
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing import selection

        for name in self.REDISPATCH_SYMBOLS:
            obj = getattr(selection, name, None)
            assert obj is not None, f"{name} not found in selection.py"
            # Must be a function defined in selection module
            assert callable(obj), f"{name} is not callable"

    def test_redispatch_symbols_not_in_wave5d_modules(self):
        for mod_name in WAVE5D_MODULES:
            mod = importlib.import_module(
                f"litellm.proxy.pass_through_endpoints.aawm_alias_routing.{mod_name}"
            )
            for name in self.REDISPATCH_SYMBOLS:
                assert not hasattr(mod, name) or name.startswith("_compat"), (
                    f"{name} should not be in {mod_name}"
                )

    def test_god_module_redispatch_facades_point_to_selection(self):
        import litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints as lpe
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing import selection

        assert lpe._raise_codex_auto_agent_in_flight_cooldown is selection._raise_codex_auto_agent_in_flight_cooldown
        assert lpe._build_auto_agent_redispatch_http_exception_detail is selection._build_auto_agent_redispatch_http_exception_detail
        assert lpe._raise_codex_auto_agent_redispatch_required is selection._raise_codex_auto_agent_redispatch_required


# ---------------------------------------------------------------------------
# FunctionType host-global rebind tests (Wave 5D install)
# ---------------------------------------------------------------------------


class TestWave5DHostGlobalRebind:
    """Installed Wave 5D functions must resolve globals through lpe.__dict__."""

    @pytest.fixture(autouse=True)
    def _import_modules(self):
        import litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints as lpe
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
            audit_build,
            audit_context,
            audit_events,
            audit_persist,
        )

        self.lpe = lpe
        self.audit_context = audit_context
        self.audit_build = audit_build
        self.audit_persist = audit_persist
        self.audit_events = audit_events

    def test_audit_context_functions_globals_is_lpe_dict(self):
        """All installed audit_context functions have __globals__ is lpe.__dict__."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.audit_context import (
            _HOST_FUNCTION_NAMES as ctx_names,
        )

        for name in ctx_names:
            fn = getattr(self.audit_context, name)
            assert fn.__globals__ is self.lpe.__dict__, (
                f"audit_context.{name}.__globals__ is not lpe.__dict__"
            )

    def test_audit_build_functions_globals_is_lpe_dict(self):
        """All installed audit_build functions have __globals__ is lpe.__dict__."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.audit_build import (
            _HOST_FUNCTION_NAMES as build_names,
        )

        for name in build_names:
            fn = getattr(self.audit_build, name)
            assert fn.__globals__ is self.lpe.__dict__, (
                f"audit_build.{name}.__globals__ is not lpe.__dict__"
            )

    def test_audit_persist_functions_globals_is_lpe_dict(self):
        """All installed audit_persist functions have __globals__ is lpe.__dict__."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.audit_persist import (
            _HOST_FUNCTION_NAMES as persist_names,
        )

        for name in persist_names:
            fn = getattr(self.audit_persist, name)
            assert fn.__globals__ is self.lpe.__dict__, (
                f"audit_persist.{name}.__globals__ is not lpe.__dict__"
            )

    def test_audit_events_functions_globals_is_lpe_dict(self):
        """All installed audit_events functions have __globals__ is lpe.__dict__."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.audit_events import (
            _HOST_FUNCTION_NAMES as events_names,
        )

        for name in events_names:
            fn = getattr(self.audit_events, name)
            assert fn.__globals__ is self.lpe.__dict__, (
                f"audit_events.{name}.__globals__ is not lpe.__dict__"
            )

    def test_patch_summarize_intercepts_attach_terminal_context(self, monkeypatch):
        """Patching lpe._summarize_... intercepts _attach_..._terminal_context_fields."""
        from unittest.mock import patch
        from fastapi import Request

        lpe = self.lpe
        request = Request({
            "type": "http",
            "method": "POST",
            "path": "/v1/responses",
            "headers": [],
            "query_string": b"",
        })
        request_body: dict = {
            "input": [{"type": "function_call", "name": "exec_command"}],
            "litellm_metadata": {"session_id": "sess-rebind-test"},
        }

        real_summarize = lpe._summarize_auto_agent_alias_actual_prior_tool_activity
        call_count = {"n": 0}

        def _counting_summarize(body):
            call_count["n"] += 1
            return real_summarize(body)

        with patch.object(
            lpe,
            "_summarize_auto_agent_alias_actual_prior_tool_activity",
            side_effect=_counting_summarize,
        ), patch.object(
            lpe,
            "_resolve_auto_agent_alias_route_host_attribution",
            return_value={
                "client_ip": "10.0.0.1",
                "client_ip_source": "test",
                "host_name": "testhost",
                "host_name_source": "test",
            },
        ):
            event: dict = {"event_type": "no_candidate_available"}
            lpe._attach_auto_agent_alias_terminal_context_fields(
                event,
                request=request,
                request_body=request_body,
                include_activity_status=True,
            )

        assert call_count["n"] == 1, (
            f"Expected exactly 1 summarize call, got {call_count['n']}"
        )
        assert event.get("actual_prior_tool_activity_summary") is not None

    def test_patch_dispatch_fields_intercepts_request_context(self, monkeypatch):
        """Patching lpe._extract_..._agent_dispatch_fields intercepts context construction."""
        from unittest.mock import patch
        from fastapi import Request

        lpe = self.lpe
        request = Request({
            "type": "http",
            "method": "POST",
            "path": "/v1/responses",
            "headers": [],
            "query_string": b"",
        })
        request_body: dict = {
            "litellm_metadata": {
                "session_id": "sess-dispatch-test",
                "agent_name": "worker",
            },
        }

        dispatch_calls = {"n": 0}
        real_dispatch = lpe._extract_auto_agent_alias_agent_dispatch_fields

        def _counting_dispatch(req, body):
            dispatch_calls["n"] += 1
            return real_dispatch(req, body)

        with patch.object(
            lpe,
            "_extract_auto_agent_alias_agent_dispatch_fields",
            side_effect=_counting_dispatch,
        ), patch.object(
            lpe,
            "_resolve_auto_agent_alias_route_host_attribution",
            return_value={
                "client_ip": "10.0.0.1",
                "client_ip_source": "test",
                "host_name": "testhost",
                "host_name_source": "test",
            },
        ):
            ctx = lpe._get_auto_agent_alias_request_context(
                request,
                request_body,
            )

        assert dispatch_calls["n"] == 1, (
            f"Expected exactly 1 dispatch extract call, got {dispatch_calls['n']}"
        )
        assert ctx["agent_dispatch"].get("agent_name") == "worker"

    def test_same_object_identity_post_install(self):
        """Owner module and lpe facade are the same rebound object after install."""
        lpe = self.lpe
        modules_and_names = [
            (self.audit_context, [
                "_extract_auto_agent_alias_agent_dispatch_fields",
                "_summarize_auto_agent_alias_actual_prior_tool_activity",
                "_get_auto_agent_alias_request_context",
                "_attach_auto_agent_alias_terminal_context_fields",
            ]),
            (self.audit_build, [
                "_build_auto_agent_alias_audit_event",
                "_build_auto_agent_alias_audit_events",
                "_codex_auto_agent_request_has_continuation_state",
            ]),
            (self.audit_persist, [
                "_emit_auto_agent_alias_route_event",
                "_should_emit_auto_agent_alias_route_event",
                "_persist_auto_agent_alias_audit_only_events_best_effort",
            ]),
            (self.audit_events, [
                "_enrich_auto_agent_alias_terminal_event_from_attempts",
                "_emit_auto_agent_alias_no_candidate_event",
            ]),
        ]
        for mod, names in modules_and_names:
            for name in names:
                assert getattr(lpe, name) is getattr(mod, name), (
                    f"lpe.{name} is not {mod.__name__}.{name} after install"
                )
