"""Wave 7 D1-591: install() publication surface for observability_metadata.

Verifies that ``install(host_globals)`` publishes exact same-object references
for every symbol in ``_HOST_PUBLISHED_NAMES``, preserving identity, mutable
state sharing, and late runtime configuration behavior.
"""

from __future__ import annotations

import ast
from pathlib import Path

from litellm.proxy.pass_through_endpoints.aawm_request_policy import (
    observability_metadata as om,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_GOD_MODULE_PATH = (
    _REPO_ROOT
    / "litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py"
)


class TestInstallPublishesAllSymbols:
    """install() must publish every name in _HOST_PUBLISHED_NAMES."""

    def test_all_published_names_present_in_host(self):
        host: dict = {}
        om.install(host)
        for name in om._HOST_PUBLISHED_NAMES:
            assert name in host, f"{name} missing from host_globals"

    def test_published_count_matches_inventory(self):
        host: dict = {}
        om.install(host)
        assert len(host) == len(om._HOST_PUBLISHED_NAMES)

    def test_published_count_is_52(self):
        """The god-module assignment block is 52 lines; install replaces all."""
        assert len(om._HOST_PUBLISHED_NAMES) == 52

    def test_published_names_match_god_assignment_order(self):
        """The inventory must be installed via install() call in god module."""
        god_source = _GOD_MODULE_PATH.read_text()
        # After Wave 7 integration, the assignment block is replaced by install()
        assert "_aawm_observability_metadata.install(globals())" in god_source, (
            "God module must call _aawm_observability_metadata.install(globals())"
        )
        # Verify the install publishes all expected names
        host: dict = {}
        om.install(host)
        assert set(host.keys()) == set(om._HOST_PUBLISHED_NAMES)


class TestInstallObjectIdentity:
    """Every published object must be the exact same object (is-check)."""

    def test_function_identity(self):
        host: dict = {}
        om.install(host)
        module_globals = vars(om)
        for name in om._HOST_PUBLISHED_NAMES:
            assert host[name] is module_globals[name], (
                f"{name} is not the same object"
            )

    def test_mutable_set_mutations_propagate_both_directions(self):
        """Both published mutable sets must share owner and host mutations."""
        host: dict = {}
        om.install(host)
        for name in (
            "_PASSTHROUGH_REPOSITORY_PLACEHOLDER_VALUES",
            "_PASSTHROUGH_REPOSITORY_AGENT_ROLE_VALUES",
        ):
            host_values = host[name]
            owner_values = getattr(om, name)
            host_marker = f"wave7-host-{name}"
            owner_marker = f"wave7-owner-{name}"
            try:
                host_values.add(host_marker)
                assert host_marker in owner_values
                owner_values.add(owner_marker)
                assert owner_marker in host_values
            finally:
                owner_values.discard(host_marker)
                owner_values.discard(owner_marker)

    def test_tuple_and_frozenset_identity(self):
        host: dict = {}
        om.install(host)
        assert (
            host["_PASSTHROUGH_SESSION_ID_HEADER_NAMES"]
            is om._PASSTHROUGH_SESSION_ID_HEADER_NAMES
        )
        assert (
            host["_PASSTHROUGH_REPOSITORY_BODY_KEYS"]
            is om._PASSTHROUGH_REPOSITORY_BODY_KEYS
        )


class TestInstallLateRuntimeConfiguration:
    """Late configure_observability_metadata_runtime must affect installed fns."""

    def test_configure_then_call_installed_function(self):
        """Calling an installed function uses late-bound runtime config."""
        host: dict = {}
        om.install(host)

        calls: list[str] = []

        def tracking_env(name: str):
            calls.append(name)
            return None

        om.configure_observability_metadata_runtime(get_env=tracking_env)
        try:
            # _get_passthrough_trace_environment reads env via _get_env
            host["_get_passthrough_trace_environment"]()
            assert len(calls) > 0, "runtime env getter was not invoked"
        finally:
            om.configure_observability_metadata_runtime()


class TestInstallIdempotent:
    """Multiple install calls must not corrupt state."""

    def test_double_install_same_objects(self):
        host: dict = {}
        om.install(host)
        first_snapshot = dict(host)
        om.install(host)
        for name in om._HOST_PUBLISHED_NAMES:
            assert host[name] is first_snapshot[name]


class TestInstallHostState:
    """install() replaces published names and preserves unrelated host state."""

    def test_replaces_stale_entry(self):
        host: dict = {"_merge_litellm_metadata": "stale"}
        om.install(host)
        assert host["_merge_litellm_metadata"] is om._merge_litellm_metadata

    def test_preserves_unrelated_host_key(self):
        sentinel = object()
        host: dict = {"unrelated": sentinel}
        om.install(host)
        assert host["unrelated"] is sentinel


class TestNoGodModuleImport:
    """The owner module must not import the god module at module scope."""

    def test_no_llm_passthrough_import(self):
        tree = ast.parse(Path(om.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "llm_passthrough_endpoints" not in alias.name
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert "llm_passthrough_endpoints" not in module


class TestHostPublishedNamesSubsetOfOwnedSymbols:
    """_HOST_PUBLISHED_NAMES must be a subset of OWNED_SYMBOLS."""

    def test_subset(self):
        owned = set(om.OWNED_SYMBOLS)
        for name in om._HOST_PUBLISHED_NAMES:
            assert name in owned, f"{name} not in OWNED_SYMBOLS"
