"""D1-591: Parity and boundary tests for owner-concrete helper APIs.

Verifies that the concrete implementations in audit_build.py and
audit_persist.py match the god-module semantics exactly, and that
configure/install APIs accept both explicit overrides and None (owner default).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from unittest.mock import patch

import pytest

_MISSING = object()

_ATTEMPT_RECORD_RUNTIME_NAMES = (
    "_extract_codex_auto_agent_error_tokens",
    "_extract_codex_auto_agent_error_type_and_code",
    "_parse_codex_auto_agent_header_wait_seconds",
    "_get_codex_auto_agent_source_error_summary",
    "_build_safe_kimi_code_selection_telemetry",
    "_extract_exception_status_code",
    "_safe_set_request_parsed_body",
    "_emit_auto_agent_alias_route_event",
    "_build_auto_agent_alias_audit_event",
    "_build_auto_agent_alias_audit_events",
    "_persist_auto_agent_alias_audit_only_events_best_effort",
    "_aawm_alias_route_verbose_json_enabled",
    "_aawm_alias_route_healthy_json_enabled",
    "_merge_litellm_metadata",
    "_normalize_low_cardinality_tag_value",
    "_normalize_codex_auto_agent_alias_model",
    "_normalize_anthropic_auto_agent_alias_model",
    "_load_bundled_model_cost_map_for_codex_policy",
    "_get_model_info",
    "_model_cost",
    "_openai_provider_value",
    "_classify_failure",
    "_read_pilot_gate_record",
)


@pytest.fixture(autouse=True)
def _restore_runtime_configuration() -> Iterator[None]:
    """Restore every callback and installed host mirror changed by configure APIs."""
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
        attempt_records,
        audit_build,
        audit_context,
        audit_events,
        audit_persist,
    )

    module_specs = (
        (
            audit_build,
            "_host_globals",
            tuple(audit_build._SEAM_NAMES | set(audit_build._HOST_FUNCTION_NAMES)),
        ),
        (
            audit_events,
            "_host_globals",
            tuple(audit_events._SEAM_NAMES | set(audit_events._HOST_FUNCTION_NAMES)),
        ),
        (
            audit_persist,
            "_host_globals",
            tuple(audit_persist._SEAM_NAMES | set(audit_persist._HOST_FUNCTION_NAMES)),
        ),
        (
            attempt_records,
            "_host_globals_ref",
            tuple(
                set(_ATTEMPT_RECORD_RUNTIME_NAMES)
                | set(attempt_records._HOST_FUNCTION_NAMES)
            ),
        ),
        (audit_context, "_host_globals", tuple(audit_context._SEAM_NAMES)),
    )
    snapshots = []
    for module, host_attr, names in module_specs:
        host_globals = getattr(module, host_attr)
        restore_stacks = getattr(module, "_runtime_restore_stacks", None)
        host_values = (
            {name: host_globals.get(name, _MISSING) for name in names}
            if host_globals is not None
            else {}
        )
        snapshots.append(
            (
                module,
                host_attr,
                names,
                {name: getattr(module, name) for name in names},
                host_globals,
                host_values,
                (
                    {
                        name: list(stack)
                        for name, stack in restore_stacks.items()
                    }
                    if restore_stacks is not None
                    else None
                ),
            )
        )

    try:
        yield
    finally:
        for (
            module,
            host_attr,
            names,
            module_values,
            host_globals,
            host_values,
            restore_stack_values,
        ) in snapshots:
            for name in names:
                setattr(module, name, module_values[name])

            if host_globals is None:
                continue
            for name in names:
                prior_value = host_values[name]
                if prior_value is _MISSING:
                    host_globals.pop(name, None)
                else:
                    host_globals[name] = prior_value
            setattr(module, host_attr, host_globals)
            restore_stacks = getattr(module, "_runtime_restore_stacks", None)
            if restore_stacks is not None and restore_stack_values is not None:
                restore_stacks.clear()
                restore_stacks.update(restore_stack_values)


# ---------------------------------------------------------------------------
# audit_build owner helpers: _auto_agent_alias_int
# ---------------------------------------------------------------------------


class TestAutoAgentAliasInt:
    def test_none_returns_none(self):
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.audit_build import (
            _auto_agent_alias_int,
        )

        assert _auto_agent_alias_int(None) is None

    def test_int_passthrough(self):
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.audit_build import (
            _auto_agent_alias_int,
        )

        assert _auto_agent_alias_int(429) == 429

    def test_string_coercion(self):
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.audit_build import (
            _auto_agent_alias_int,
        )

        assert _auto_agent_alias_int("429") == 429

    def test_float_truncation(self):
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.audit_build import (
            _auto_agent_alias_int,
        )

        assert _auto_agent_alias_int(3.9) == 3

    def test_invalid_string_returns_none(self):
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.audit_build import (
            _auto_agent_alias_int,
        )

        assert _auto_agent_alias_int("not_a_number") is None

    def test_empty_list_returns_none(self):
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.audit_build import (
            _auto_agent_alias_int,
        )

        assert _auto_agent_alias_int([]) is None


# ---------------------------------------------------------------------------
# audit_build owner helpers: _format_auto_agent_alias_timestamp
# ---------------------------------------------------------------------------


class TestFormatAutoAgentAliasTimestamp:
    def test_utc_datetime(self):
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.audit_build import (
            _format_auto_agent_alias_timestamp,
        )

        dt = datetime(2026, 7, 27, 12, 30, 45, tzinfo=timezone.utc)
        assert _format_auto_agent_alias_timestamp(dt) == "2026-07-27T12:30:45Z"

    def test_non_utc_converted(self):
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.audit_build import (
            _format_auto_agent_alias_timestamp,
        )

        eastern = timezone(timedelta(hours=-5))
        dt = datetime(2026, 7, 27, 7, 30, 45, tzinfo=eastern)
        assert _format_auto_agent_alias_timestamp(dt) == "2026-07-27T12:30:45Z"

    def test_microseconds_preserved(self):
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.audit_build import (
            _format_auto_agent_alias_timestamp,
        )

        dt = datetime(2026, 1, 1, 0, 0, 0, 123456, tzinfo=timezone.utc)
        result = _format_auto_agent_alias_timestamp(dt)
        assert result == "2026-01-01T00:00:00.123456Z"


# ---------------------------------------------------------------------------
# audit_build owner helpers: _auto_agent_alias_cooldown_until
# ---------------------------------------------------------------------------


class TestAutoAgentAliasCooldownUntil:
    def test_none_returns_none(self):
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.audit_build import (
            _auto_agent_alias_cooldown_until,
        )

        assert _auto_agent_alias_cooldown_until(None) is None

    def test_positive_seconds_returns_future_z_timestamp(self):
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.audit_build import (
            _auto_agent_alias_cooldown_until,
        )

        result = _auto_agent_alias_cooldown_until(60.0)
        assert result is not None
        assert result.endswith("Z")
        # Parse back and verify it's roughly 60s in the future
        parsed = datetime.fromisoformat(result.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        delta = (parsed - now).total_seconds()
        assert 55.0 <= delta <= 65.0

    def test_negative_seconds_clamped_to_zero(self):
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.audit_build import (
            _auto_agent_alias_cooldown_until,
        )

        result = _auto_agent_alias_cooldown_until(-10.0)
        assert result is not None
        parsed = datetime.fromisoformat(result.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        delta = (parsed - now).total_seconds()
        assert -2.0 <= delta <= 2.0

    def test_zero_seconds(self):
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.audit_build import (
            _auto_agent_alias_cooldown_until,
        )

        result = _auto_agent_alias_cooldown_until(0.0)
        assert result is not None
        assert result.endswith("Z")


# ---------------------------------------------------------------------------
# audit_persist owner helpers: verbosity env checks
# ---------------------------------------------------------------------------


class TestVerboseJsonEnabled:
    @pytest.mark.parametrize(
        "env_val,expected",
        [
            ("1", True),
            ("true", True),
            ("yes", True),
            ("debug", True),
            ("verbose", True),
            ("TRUE", True),
            (" 1 ", True),
            ("0", False),
            ("false", False),
            ("", False),
            ("no", False),
            ("random", False),
        ],
    )
    def test_env_values(self, env_val: str, expected: bool):
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.audit_persist import (
            _aawm_alias_route_verbose_json_enabled,
        )

        with patch.dict(os.environ, {"AAWM_ALIAS_ROUTE_VERBOSE_JSON": env_val}):
            assert _aawm_alias_route_verbose_json_enabled() is expected

    def test_env_unset(self):
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.audit_persist import (
            _aawm_alias_route_verbose_json_enabled,
        )

        with patch.dict(os.environ, {}, clear=True):
            # Remove the key if present
            os.environ.pop("AAWM_ALIAS_ROUTE_VERBOSE_JSON", None)
            assert _aawm_alias_route_verbose_json_enabled() is False


class TestHealthyJsonEnabled:
    @pytest.mark.parametrize(
        "env_val,expected",
        [
            ("1", True),
            ("true", True),
            ("yes", True),
            ("TRUE", True),
            (" 1 ", True),
            ("debug", False),  # NOT in healthy set
            ("verbose", False),  # NOT in healthy set
            ("0", False),
            ("false", False),
            ("", False),
        ],
    )
    def test_env_values(self, env_val: str, expected: bool):
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.audit_persist import (
            _aawm_alias_route_healthy_json_enabled,
        )

        with patch.dict(os.environ, {"AAWM_ALIAS_ROUTE_LOG_HEALTHY": env_val}):
            assert _aawm_alias_route_healthy_json_enabled() is expected

    def test_env_unset(self):
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.audit_persist import (
            _aawm_alias_route_healthy_json_enabled,
        )

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("AAWM_ALIAS_ROUTE_LOG_HEALTHY", None)
            assert _aawm_alias_route_healthy_json_enabled() is False


# ---------------------------------------------------------------------------
# Configure API: None params preserve owner defaults
# ---------------------------------------------------------------------------


class TestConfigureOptionalParams:
    def test_audit_build_configure_none_preserves_defaults(self):
        """configure_audit_build_runtime with None format/to_int/cooldown keeps owner defaults."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing import audit_build as mod

        prev_fmt = mod._format_auto_agent_alias_timestamp
        prev_int = mod._auto_agent_alias_int
        prev_cd = mod._auto_agent_alias_cooldown_until

        mod.configure_audit_build_runtime(
            get_request_context=lambda *a, **kw: {},
            attach_terminal_context_fields=lambda *a, **kw: None,
            format_timestamp=None,
            extract_metadata_value=lambda *a, **kw: None,
            extract_incoming_endpoint=lambda *a, **kw: "/test",
            resolve_outgoing_target=lambda *a, **kw: None,
            to_int=None,
            cooldown_until=None,
        )
        assert mod._format_auto_agent_alias_timestamp is prev_fmt
        assert mod._auto_agent_alias_int is prev_int
        assert mod._auto_agent_alias_cooldown_until is prev_cd

    def test_audit_build_configure_explicit_overrides(self):
        """configure_audit_build_runtime with explicit callables overrides defaults."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing import audit_build as mod

        def custom_fmt(_value: datetime) -> str:
            return "custom"

        def custom_int(_value: Any) -> Optional[int]:
            return 999

        def custom_cd(_value: Optional[float]) -> Optional[str]:
            return "custom_cd"

        mod.configure_audit_build_runtime(
            get_request_context=lambda *a, **kw: {},
            attach_terminal_context_fields=lambda *a, **kw: None,
            format_timestamp=custom_fmt,
            extract_metadata_value=lambda *a, **kw: None,
            extract_incoming_endpoint=lambda *a, **kw: "/test",
            resolve_outgoing_target=lambda *a, **kw: None,
            to_int=custom_int,
            cooldown_until=custom_cd,
        )
        assert mod._format_auto_agent_alias_timestamp is custom_fmt
        assert mod._auto_agent_alias_int is custom_int
        assert mod._auto_agent_alias_cooldown_until is custom_cd

    def test_audit_events_configure_none_preserves_default(self):
        """configure_audit_events_runtime keeps its timestamp owner default."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing import audit_events as mod

        mod._format_auto_agent_alias_timestamp = mod._default_format_timestamp
        previous_format = mod._format_auto_agent_alias_timestamp

        mod.configure_audit_events_runtime(
            get_request_context=lambda *a, **kw: {},
            attach_terminal_context_fields=lambda *a, **kw: None,
            format_timestamp=None,
            extract_metadata_value=lambda *a, **kw: None,
            extract_incoming_endpoint=lambda *a, **kw: "/test",
            resolve_codex_session_key=lambda *a, **kw: None,
            resolve_anthropic_session_key=lambda *a, **kw: None,
            emit_route_event=lambda *a, **kw: None,
            build_audit_events=lambda *a, **kw: [],
            persist_audit_only_events=lambda *a, **kw: "skip_empty",
        )

        assert mod._format_auto_agent_alias_timestamp is previous_format

    def test_audit_persist_configure_none_preserves_defaults(self):
        """configure_audit_persist_runtime with None verbosity params keeps owner defaults."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing import audit_persist as mod

        prev_verbose = mod._aawm_alias_route_verbose_json_enabled
        prev_healthy = mod._aawm_alias_route_healthy_json_enabled

        mod.configure_audit_persist_runtime(
            record_route_status_rollup=lambda *a, **kw: None,
            verbose_json_enabled=None,
            healthy_json_enabled=None,
        )
        assert mod._aawm_alias_route_verbose_json_enabled is prev_verbose
        assert mod._aawm_alias_route_healthy_json_enabled is prev_healthy

    def test_audit_persist_configure_explicit_overrides(self):
        """configure_audit_persist_runtime with explicit callables overrides defaults."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing import audit_persist as mod

        def custom_verbose() -> bool:
            return True

        def custom_healthy() -> bool:
            return False

        mod.configure_audit_persist_runtime(
            record_route_status_rollup=lambda *a, **kw: None,
            verbose_json_enabled=custom_verbose,
            healthy_json_enabled=custom_healthy,
        )
        assert mod._aawm_alias_route_verbose_json_enabled is custom_verbose
        assert mod._aawm_alias_route_healthy_json_enabled is custom_healthy

    def test_attempt_records_configure_none_preserves_defaults(self):
        """configure_attempt_records_runtime keeps both verbosity owner defaults."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
            attempt_records as mod,
        )

        mod._aawm_alias_route_verbose_json_enabled = mod._default_verbose_json_enabled
        mod._aawm_alias_route_healthy_json_enabled = mod._default_healthy_json_enabled
        previous_verbose = mod._aawm_alias_route_verbose_json_enabled
        previous_healthy = mod._aawm_alias_route_healthy_json_enabled

        mod.configure_attempt_records_runtime(
            extract_error_tokens=lambda *a, **kw: set(),
            extract_error_type_and_code=lambda *a, **kw: (None, None),
            parse_header_wait_seconds=lambda *a, **kw: None,
            get_source_error_summary=lambda *a, **kw: None,
            build_kimi_telemetry=lambda *a, **kw: {},
            extract_status_code=lambda *a, **kw: None,
            safe_set_parsed_body=lambda *a, **kw: None,
            emit_route_event=lambda *a, **kw: None,
            build_audit_event=lambda *a, **kw: {},
            build_audit_events=lambda *a, **kw: [],
            persist_audit_only_events=lambda *a, **kw: None,
            verbose_json_enabled=None,
            healthy_json_enabled=None,
            merge_metadata=lambda request_body, **kw: request_body,
            normalize_tag_value=lambda *a, **kw: None,
            normalize_codex_alias_model=lambda *a, **kw: None,
            normalize_anthropic_alias_model=lambda *a, **kw: None,
            load_bundled_model_cost=lambda: {},
            get_model_info=lambda *a, **kw: {},
            model_cost={},
            openai_provider_value="openai",
            classify_failure=lambda *a, **kw: None,
            read_pilot_gate_record=lambda *a, **kw: None,
        )

        assert mod._aawm_alias_route_verbose_json_enabled is previous_verbose
        assert mod._aawm_alias_route_healthy_json_enabled is previous_healthy


# ---------------------------------------------------------------------------
# Configure/install wrapper canonicalization
# ---------------------------------------------------------------------------


class TestInstallWrapperCanonicalization:
    def test_audit_build_preserves_owner_int_behind_host_wrapper(self):
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
            audit_build as mod,
        )

        owner_int = mod._auto_agent_alias_int
        mod.configure_audit_build_runtime(
            get_request_context=lambda *a, **kw: {},
            attach_terminal_context_fields=lambda *a, **kw: None,
            format_timestamp=None,
            extract_metadata_value=lambda *a, **kw: None,
            extract_incoming_endpoint=lambda *a, **kw: "/test",
            resolve_outgoing_target=lambda *a, **kw: None,
            to_int=owner_int,
            cooldown_until=None,
        )

        def host_int(value: Any) -> Optional[int]:
            return mod._auto_agent_alias_int(value)

        host_globals = dict(mod._host_globals or {})
        host_globals["_auto_agent_alias_int"] = host_int
        mod.install(host_globals)

        assert mod._auto_agent_alias_int is owner_int
        assert host_globals["_auto_agent_alias_int"] is host_int
        assert host_globals["_auto_agent_alias_int"]("429") == 429

    def test_audit_events_preserves_owner_timestamp_behind_host_wrapper(self):
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
            audit_events as mod,
        )

        owner_format = mod._default_format_timestamp
        mod.configure_audit_events_runtime(
            get_request_context=lambda *a, **kw: {},
            attach_terminal_context_fields=lambda *a, **kw: None,
            format_timestamp=owner_format,
            extract_metadata_value=lambda *a, **kw: None,
            extract_incoming_endpoint=lambda *a, **kw: "/test",
            resolve_codex_session_key=lambda *a, **kw: None,
            resolve_anthropic_session_key=lambda *a, **kw: None,
            emit_route_event=lambda *a, **kw: None,
            build_audit_events=lambda *a, **kw: [],
            persist_audit_only_events=lambda *a, **kw: "skip_empty",
        )

        def host_format(value: datetime) -> str:
            return mod._format_auto_agent_alias_timestamp(value)

        host_globals = dict(mod._host_globals or {})
        host_globals["_format_auto_agent_alias_timestamp"] = host_format
        mod.install(host_globals)

        value = datetime(2026, 7, 28, 12, 30, tzinfo=timezone.utc)
        assert mod._format_auto_agent_alias_timestamp is owner_format
        assert host_globals["_format_auto_agent_alias_timestamp"] is host_format
        assert host_globals["_format_auto_agent_alias_timestamp"](value).endswith("Z")

    def test_audit_persist_preserves_owner_verbose_behind_host_wrapper(self):
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
            audit_persist as mod,
        )

        def owner_verbose() -> bool:
            return True

        mod.configure_audit_persist_runtime(
            record_route_status_rollup=lambda *a, **kw: None,
            verbose_json_enabled=owner_verbose,
            healthy_json_enabled=None,
        )

        def host_verbose() -> bool:
            return mod._aawm_alias_route_verbose_json_enabled()

        host_globals = dict(mod._host_globals or {})
        host_globals["_aawm_alias_route_verbose_json_enabled"] = host_verbose
        mod.install(host_globals)

        assert mod._aawm_alias_route_verbose_json_enabled is owner_verbose
        assert host_globals["_aawm_alias_route_verbose_json_enabled"] is host_verbose
        assert host_globals["_aawm_alias_route_verbose_json_enabled"]() is True

    def test_attempt_records_preserves_owner_verbose_behind_host_wrapper(self):
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
            attempt_records as mod,
        )

        def owner_verbose() -> bool:
            return True

        mod.configure_attempt_records_runtime(
            extract_error_tokens=lambda *a, **kw: set(),
            extract_error_type_and_code=lambda *a, **kw: (None, None),
            parse_header_wait_seconds=lambda *a, **kw: None,
            get_source_error_summary=lambda *a, **kw: None,
            build_kimi_telemetry=lambda *a, **kw: {},
            extract_status_code=lambda *a, **kw: None,
            safe_set_parsed_body=lambda *a, **kw: None,
            emit_route_event=lambda *a, **kw: None,
            build_audit_event=lambda *a, **kw: {},
            build_audit_events=lambda *a, **kw: [],
            persist_audit_only_events=lambda *a, **kw: None,
            verbose_json_enabled=owner_verbose,
            healthy_json_enabled=None,
            merge_metadata=lambda request_body, **kw: request_body,
            normalize_tag_value=lambda *a, **kw: None,
            normalize_codex_alias_model=lambda *a, **kw: None,
            normalize_anthropic_alias_model=lambda *a, **kw: None,
            load_bundled_model_cost=lambda: {},
            get_model_info=lambda *a, **kw: {},
            model_cost={},
            openai_provider_value="openai",
            classify_failure=lambda *a, **kw: None,
            read_pilot_gate_record=lambda *a, **kw: None,
        )

        def host_verbose() -> bool:
            return mod._aawm_alias_route_verbose_json_enabled()

        host_globals = dict(mod._host_globals_ref or {})
        host_globals["_aawm_alias_route_verbose_json_enabled"] = host_verbose
        mod.install(host_globals)

        assert mod._aawm_alias_route_verbose_json_enabled is owner_verbose
        assert host_globals["_aawm_alias_route_verbose_json_enabled"] is host_verbose
        assert host_globals["_aawm_alias_route_verbose_json_enabled"]() is True


# ---------------------------------------------------------------------------
# Parity: owner defaults match god-module semantics
# ---------------------------------------------------------------------------


class TestParityWithGodModule:
    """Verify owner-concrete functions produce identical output to god-module defs."""

    def test_int_parity(self):
        """Owner _auto_agent_alias_int matches god-module inline definition."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.audit_build import (
            _auto_agent_alias_int,
        )

        # God-module definition:
        def god_int(value: Any) -> Optional[int]:
            if value is None:
                return None
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        cases = [None, 0, 1, -1, 429, "429", "abc", 3.14, [], {}, True]
        for case in cases:
            assert _auto_agent_alias_int(case) == god_int(case), f"Mismatch for {case!r}"

    def test_timestamp_parity(self):
        """Owner _format_auto_agent_alias_timestamp matches god-module inline definition."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.audit_build import (
            _format_auto_agent_alias_timestamp,
        )

        def god_fmt(value: datetime) -> str:
            return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

        cases = [
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            datetime(2026, 7, 27, 23, 59, 59, 999999, tzinfo=timezone.utc),
            datetime(2026, 3, 15, 10, 0, 0, tzinfo=timezone(timedelta(hours=5, minutes=30))),
        ]
        for case in cases:
            assert _format_auto_agent_alias_timestamp(case) == god_fmt(case)

    def test_verbose_json_parity(self):
        """Owner _aawm_alias_route_verbose_json_enabled matches god-module inline definition."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.audit_persist import (
            _aawm_alias_route_verbose_json_enabled,
        )

        def god_verbose() -> bool:
            return os.getenv("AAWM_ALIAS_ROUTE_VERBOSE_JSON", "").strip().lower() in {
                "1", "true", "yes", "debug", "verbose",
            }

        for val in ["1", "true", "yes", "debug", "verbose", "0", "false", "", "no", "DEBUG"]:
            with patch.dict(os.environ, {"AAWM_ALIAS_ROUTE_VERBOSE_JSON": val}):
                assert _aawm_alias_route_verbose_json_enabled() == god_verbose(), f"Mismatch for {val!r}"

    def test_healthy_json_parity(self):
        """Owner _aawm_alias_route_healthy_json_enabled matches god-module inline definition."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.audit_persist import (
            _aawm_alias_route_healthy_json_enabled,
        )

        def god_healthy() -> bool:
            return os.getenv("AAWM_ALIAS_ROUTE_LOG_HEALTHY", "").strip().lower() in {
                "1", "true", "yes",
            }

        for val in ["1", "true", "yes", "debug", "verbose", "0", "false", "", "TRUE"]:
            with patch.dict(os.environ, {"AAWM_ALIAS_ROUTE_LOG_HEALTHY": val}):
                assert _aawm_alias_route_healthy_json_enabled() == god_healthy(), f"Mismatch for {val!r}"


# ---------------------------------------------------------------------------
# attempt_records defaults from audit_persist
# ---------------------------------------------------------------------------


class TestAttemptRecordsVerbosityDefaults:
    def test_defaults_are_audit_persist_functions(self):
        """attempt_records owner defaults retain audit_persist env semantics."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing import attempt_records as ar_mod

        with patch.dict(
            os.environ,
            {
                "AAWM_ALIAS_ROUTE_VERBOSE_JSON": "debug",
                "AAWM_ALIAS_ROUTE_LOG_HEALTHY": "yes",
            },
        ):
            assert ar_mod._default_verbose_json_enabled() is True
            assert ar_mod._default_healthy_json_enabled() is True
