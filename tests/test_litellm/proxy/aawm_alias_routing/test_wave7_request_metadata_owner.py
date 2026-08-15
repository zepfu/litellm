"""Wave 7 owner tests for request-metadata and host-attribution helpers (D1-591).

Covers: session-id extraction chain, metadata value extraction, client-product
normalization/label extraction, incoming-endpoint normalization with query
allowlist, and sync/async host attribution fail-closed behavior.
"""

from __future__ import annotations

import ast
import inspect
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    request_metadata,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.codex_oauth import (
    _clean_codex_auth_value,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.request_metadata import (
    _aresolve_auto_agent_alias_route_host_attribution,
    _extract_auto_agent_alias_client_product_label,
    _extract_auto_agent_alias_incoming_endpoint,
    _extract_auto_agent_alias_metadata_value,
    _extract_auto_agent_alias_session_id,
    _normalize_auto_agent_alias_client_product,
    _resolve_auto_agent_alias_route_host_attribution,
    configure_request_metadata_runtime,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NEUTRAL_ATTRIBUTION = {
    "client_ip": None,
    "client_ip_source": None,
    "host_name": None,
    "host_name_source": None,
}

_MISSING = object()
_REQUEST_METADATA_RUNTIME_GLOBALS = (
    *request_metadata._HOST_FUNCTION_NAMES,
    "_extract_passthrough_session_id",
    "_get_codex_auto_agent_header",
)


class _FakeRequest:
    """Minimal stand-in for fastapi.Request in unit tests."""

    def __init__(
        self,
        headers: Optional[dict[str, str]] = None,
        url: str = "",
        path: Optional[str] = None,
    ) -> None:
        self.headers = headers or {}
        self.url = url
        if path is not None:
            self.path = path


def _fake_get_header(headers: dict, header_name: str) -> Optional[str]:
    """Mimics lane_keys._get_codex_auto_agent_header without install()."""
    for key, value in headers.items():
        if not isinstance(key, str) or key.lower() != header_name.lower():
            continue
        cleaned = _clean_codex_auth_value(value)
        if cleaned is not None:
            return cleaned
    return None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _inject_runtime():
    """Provide the cross-module seam for every test."""
    original_functions = {
        name: getattr(request_metadata, name)
        for name in request_metadata._HOST_FUNCTION_NAMES
    }
    original_host_globals = request_metadata._host_globals
    original_host_values = (
        {
            name: original_host_globals.get(name, _MISSING)
            for name in _REQUEST_METADATA_RUNTIME_GLOBALS
        }
        if original_host_globals is not None
        else {}
    )
    original_passthrough = request_metadata._extract_passthrough_session_id
    original_header = request_metadata._get_codex_auto_agent_header
    fake_passthrough = MagicMock(return_value=None)
    configure_request_metadata_runtime(
        extract_passthrough_session_id=fake_passthrough,
        get_codex_auto_agent_header=_fake_get_header,
    )
    yield fake_passthrough
    for name, function in original_functions.items():
        setattr(request_metadata, name, function)
    if original_host_globals is not None:
        for name, value in original_host_values.items():
            if value is _MISSING:
                original_host_globals.pop(name, None)
            else:
                original_host_globals[name] = value
    request_metadata._host_globals = original_host_globals
    request_metadata._extract_passthrough_session_id = original_passthrough
    request_metadata._get_codex_auto_agent_header = original_header


# ---------------------------------------------------------------------------
# configure / contract
# ---------------------------------------------------------------------------


class TestConfigureContract:
    def test_configure_is_sync(self):
        assert not inspect.iscoroutinefunction(configure_request_metadata_runtime)

    def test_session_id_extraction_is_sync(self):
        assert not inspect.iscoroutinefunction(_extract_auto_agent_alias_session_id)

    def test_async_host_attribution_is_coroutine(self):
        assert inspect.iscoroutinefunction(
            _aresolve_auto_agent_alias_route_host_attribution
        )

    def test_sync_host_attribution_is_sync(self):
        assert not inspect.iscoroutinefunction(
            _resolve_auto_agent_alias_route_host_attribution
        )

    def test_pre_install_unconfigured_session_seam_fails_closed(self):
        function_globals = (
            request_metadata._extract_auto_agent_alias_session_id.__globals__
        )
        with patch.dict(
            function_globals,
            {"_extract_passthrough_session_id": None},
        ):
            with pytest.raises(
                RuntimeError,
                match="missing extract_passthrough_session_id",
            ):
                request_metadata._extract_auto_agent_alias_session_id(
                    _FakeRequest(),
                    {},
                )

    def test_pre_install_unconfigured_header_seam_fails_closed(self):
        function_globals = (
            request_metadata._extract_auto_agent_alias_client_product_label.__globals__
        )
        with patch.dict(
            function_globals,
            {"_get_codex_auto_agent_header": None},
        ):
            with pytest.raises(
                RuntimeError,
                match="missing get_codex_auto_agent_header",
            ):
                request_metadata._extract_auto_agent_alias_client_product_label(
                    _FakeRequest(),
                    {},
                )

    def test_no_module_scope_god_import(self):
        tree = ast.parse(inspect.getsource(request_metadata))
        imported_modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported_modules.append(node.module)
        assert not any(
            module.endswith("llm_passthrough_endpoints")
            for module in imported_modules
        )


# ---------------------------------------------------------------------------
# _extract_auto_agent_alias_session_id
# ---------------------------------------------------------------------------


class TestExtractSessionId:
    def test_metadata_session_id_preferred(self):
        request = _FakeRequest()
        body = {"litellm_metadata": {"session_id": "meta-123"}}
        assert _extract_auto_agent_alias_session_id(request, body) == "meta-123"

    def test_metadata_session_id_cleaned(self):
        request = _FakeRequest()
        body = {"litellm_metadata": {"session_id": "  spaced-456  "}}
        assert _extract_auto_agent_alias_session_id(request, body) == "spaced-456"

    def test_passthrough_fallback(self, _inject_runtime):
        _inject_runtime.return_value = "passthrough-789"
        request = _FakeRequest()
        body: dict[str, Any] = {}
        assert _extract_auto_agent_alias_session_id(request, body) == "passthrough-789"
        _inject_runtime.assert_called_once_with(request, body)

    def test_header_fallback(self):
        request = _FakeRequest(headers={"x-session-id": "hdr-abc"})
        body: dict[str, Any] = {}
        assert _extract_auto_agent_alias_session_id(request, body) == "hdr-abc"

    def test_header_fallback_session_id_key(self):
        request = _FakeRequest(headers={"session_id": "hdr-def"})
        body: dict[str, Any] = {}
        assert _extract_auto_agent_alias_session_id(request, body) == "hdr-def"

    def test_header_fallback_dashed_key(self):
        request = _FakeRequest(headers={"session-id": "hdr-ghi"})
        body: dict[str, Any] = {}
        assert _extract_auto_agent_alias_session_id(request, body) == "hdr-ghi"

    def test_none_when_nothing_found(self):
        request = _FakeRequest()
        body: dict[str, Any] = {}
        assert _extract_auto_agent_alias_session_id(request, body) is None

    def test_metadata_not_dict_falls_through(self):
        request = _FakeRequest(headers={"session_id": "hdr-fallback"})
        body: dict[str, Any] = {"litellm_metadata": "not-a-dict"}
        assert _extract_auto_agent_alias_session_id(request, body) == "hdr-fallback"


# ---------------------------------------------------------------------------
# _extract_auto_agent_alias_metadata_value
# ---------------------------------------------------------------------------


class TestExtractMetadataValue:
    def test_first_matching_key(self):
        body = {"litellm_metadata": {"a": "val-a", "b": "val-b"}}
        assert _extract_auto_agent_alias_metadata_value(body, "a", "b") == "val-a"

    def test_second_key_when_first_missing(self):
        body = {"litellm_metadata": {"b": "val-b"}}
        assert _extract_auto_agent_alias_metadata_value(body, "a", "b") == "val-b"

    def test_none_when_no_match(self):
        body = {"litellm_metadata": {"c": "val-c"}}
        assert _extract_auto_agent_alias_metadata_value(body, "a", "b") is None

    def test_none_when_metadata_not_dict(self):
        body: dict[str, Any] = {"litellm_metadata": 42}
        assert _extract_auto_agent_alias_metadata_value(body, "a") is None

    def test_none_when_no_metadata_key(self):
        body: dict[str, Any] = {}
        assert _extract_auto_agent_alias_metadata_value(body, "a") is None

    def test_blank_value_skipped(self):
        body = {"litellm_metadata": {"a": "   ", "b": "real"}}
        assert _extract_auto_agent_alias_metadata_value(body, "a", "b") == "real"


# ---------------------------------------------------------------------------
# _normalize_auto_agent_alias_client_product
# ---------------------------------------------------------------------------


class TestNormalizeClientProduct:
    def test_codex_variants(self):
        for raw in ("codex/1.0", "codex-cli/2.0", "codex_tui/3.0", "Codex-CLI-RS/4.0"):
            result = _normalize_auto_agent_alias_client_product(raw)
            assert result is not None
            assert result.startswith("Codex/")

    def test_claude_variants(self):
        for raw in ("claude/1.0", "claude-cli/2.0", "Claude-Code/3.0"):
            result = _normalize_auto_agent_alias_client_product(raw)
            assert result is not None
            assert result.startswith("Claude/")

    def test_grok_variants(self):
        for raw in ("grok/1.0", "grok-build/2.0", "Grok_Pager/3.0"):
            result = _normalize_auto_agent_alias_client_product(raw)
            assert result is not None
            assert result.startswith("Grok/")

    def test_unknown_product_preserved(self):
        assert _normalize_auto_agent_alias_client_product("mytool/1.0") == "mytool/1.0"

    def test_no_version_passthrough(self):
        assert _normalize_auto_agent_alias_client_product("simple") == "simple"

    def test_none_input(self):
        assert _normalize_auto_agent_alias_client_product(None) is None

    def test_blank_input(self):
        assert _normalize_auto_agent_alias_client_product("   ") is None

    def test_strips_surrounding_parens(self):
        assert _normalize_auto_agent_alias_client_product("(codex/1.0)") == "Codex/1.0"

    def test_takes_first_token_only(self):
        assert (
            _normalize_auto_agent_alias_client_product("codex/1.0 extra-stuff")
            == "Codex/1.0"
        )


# ---------------------------------------------------------------------------
# _extract_auto_agent_alias_client_product_label
# ---------------------------------------------------------------------------


class TestExtractClientProductLabel:
    def test_metadata_client_name_version(self):
        request = _FakeRequest()
        body = {"litellm_metadata": {"client_name_version": "codex/1.2.3"}}
        assert (
            _extract_auto_agent_alias_client_product_label(request, body)
            == "Codex/1.2.3"
        )

    def test_metadata_client_label(self):
        request = _FakeRequest()
        body = {"litellm_metadata": {"client_label": "claude-code/2.0"}}
        assert (
            _extract_auto_agent_alias_client_product_label(request, body)
            == "Claude/2.0"
        )

    def test_metadata_name_plus_version(self):
        request = _FakeRequest()
        body = {
            "litellm_metadata": {
                "client_name": "mytool",
                "client_version": "3.1",
            }
        }
        assert (
            _extract_auto_agent_alias_client_product_label(request, body)
            == "mytool/3.1"
        )

    def test_metadata_name_only(self):
        request = _FakeRequest()
        body = {"litellm_metadata": {"client_name": "bare-tool"}}
        assert (
            _extract_auto_agent_alias_client_product_label(request, body)
            == "bare-tool"
        )

    def test_metadata_name_with_slash_ignores_version(self):
        request = _FakeRequest()
        body = {
            "litellm_metadata": {
                "client_name": "tool/9.9",
                "client_version": "ignored",
            }
        }
        # name already contains "/", so version is not appended
        assert (
            _extract_auto_agent_alias_client_product_label(request, body)
            == "tool/9.9"
        )

    def test_header_fallback_x_aawm_client(self):
        request = _FakeRequest(headers={"x-aawm-client": "grok/0.5"})
        body: dict[str, Any] = {}
        assert (
            _extract_auto_agent_alias_client_product_label(request, body)
            == "Grok/0.5"
        )

    def test_header_fallback_user_agent(self):
        request = _FakeRequest(headers={"user-agent": "codex-cli/7.7"})
        body: dict[str, Any] = {}
        assert (
            _extract_auto_agent_alias_client_product_label(request, body)
            == "Codex/7.7"
        )

    def test_none_when_nothing_found(self):
        request = _FakeRequest()
        body: dict[str, Any] = {}
        assert _extract_auto_agent_alias_client_product_label(request, body) is None

    def test_metadata_priority_over_headers(self):
        request = _FakeRequest(headers={"x-aawm-client": "grok/0.1"})
        body = {"litellm_metadata": {"client_name_version": "codex/2.0"}}
        assert (
            _extract_auto_agent_alias_client_product_label(request, body)
            == "Codex/2.0"
        )


# ---------------------------------------------------------------------------
# _extract_auto_agent_alias_incoming_endpoint
# ---------------------------------------------------------------------------


class TestExtractIncomingEndpoint:
    def test_path_only(self):
        request = _FakeRequest(url="http://localhost/v1/chat/completions")
        assert (
            _extract_auto_agent_alias_incoming_endpoint(request)
            == "/v1/chat/completions"
        )

    def test_allowlisted_query_params_preserved(self):
        request = _FakeRequest(
            url="http://localhost/v1/models?alt=sse&stream=true"
        )
        result = _extract_auto_agent_alias_incoming_endpoint(request)
        assert "alt=sse" in result
        assert "stream=true" in result
        assert result.startswith("/v1/models?")

    def test_non_allowlisted_params_stripped(self):
        request = _FakeRequest(
            url="http://localhost/v1/models?secret=abc&alt=json"
        )
        result = _extract_auto_agent_alias_incoming_endpoint(request)
        assert "secret" not in result
        assert "alt=json" in result

    def test_api_version_and_beta_allowed(self):
        request = _FakeRequest(
            url="http://localhost/v1/x?api-version=2024&beta=feat"
        )
        result = _extract_auto_agent_alias_incoming_endpoint(request)
        assert "api-version=2024" in result
        assert "beta=feat" in result

    def test_empty_url_defaults_to_slash(self):
        request = _FakeRequest(url="")
        assert _extract_auto_agent_alias_incoming_endpoint(request) == "/"

    def test_path_attr_fallback(self):
        request = _FakeRequest(url="", path="/fallback/path")
        assert _extract_auto_agent_alias_incoming_endpoint(request) == "/fallback/path"

    def test_blank_query_values_stripped(self):
        # keep_blank_values=True in parse_qsl, but _clean_codex_auth_value
        # returns None for blank, so blank values are excluded
        request = _FakeRequest(url="http://localhost/v1?alt=")
        result = _extract_auto_agent_alias_incoming_endpoint(request)
        assert result == "/v1"


# ---------------------------------------------------------------------------
# _resolve_auto_agent_alias_route_host_attribution (sync)
# ---------------------------------------------------------------------------


class TestSyncHostAttribution:
    def test_prefers_cached_request_state_attribution(self):
        fake_result = {
            "client_ip": "10.0.0.1",
            "client_ip_source": "x-forwarded-for",
            "host_name": "state-host",
            "host_name_source": "state",
        }
        request = _FakeRequest()
        request.client = SimpleNamespace(host="10.0.0.1")
        request.state = SimpleNamespace(aawm_route_host_attribution=fake_result)

        with patch.dict(
            _resolve_auto_agent_alias_route_host_attribution.__globals__,
            {
                "resolve_aawm_route_host_attribution": MagicMock(
                    side_effect=AssertionError("state must win")
                )
            },
        ):
            result = _resolve_auto_agent_alias_route_host_attribution(request)
            assert result == fake_result

    def test_stale_request_state_delegates_to_resolver(self):
        stale_result = {
            "client_ip": "10.0.0.1",
            "client_ip_source": "request_client",
            "host_name": "stale-host",
            "host_name_source": "state",
        }
        fresh_result = {
            "client_ip": "10.0.0.2",
            "client_ip_source": "request_client",
            "host_name": "fresh-host",
            "host_name_source": "reverse-dns",
        }
        request = _FakeRequest()
        request.client = SimpleNamespace(host="10.0.0.2")
        request.state = SimpleNamespace(aawm_route_host_attribution=stale_result)
        mock_resolve = MagicMock(return_value=fresh_result)

        with patch.dict(
            _resolve_auto_agent_alias_route_host_attribution.__globals__,
            {"resolve_aawm_route_host_attribution": mock_resolve},
        ):
            result = _resolve_auto_agent_alias_route_host_attribution(request)

        assert result == fresh_result
        mock_resolve.assert_called_once_with(
            request,
            allow_blocking_lookup=False,
        )

    def test_delegates_to_resolver(self):
        fake_result = {
            "client_ip": "10.0.0.1",
            "client_ip_source": "x-forwarded-for",
            "host_name": "host.local",
            "host_name_source": "reverse-dns",
        }
        mock_resolve = MagicMock(return_value=fake_result)
        with patch.dict(
            _resolve_auto_agent_alias_route_host_attribution.__globals__,
            {"resolve_aawm_route_host_attribution": mock_resolve},
        ):
            request = _FakeRequest()
            result = _resolve_auto_agent_alias_route_host_attribution(request)
            assert result == fake_result
            mock_resolve.assert_called_once_with(
                request, allow_blocking_lookup=False
            )

    def test_fail_closed_on_exception(self):
        with patch.dict(
            _resolve_auto_agent_alias_route_host_attribution.__globals__,
            {
                "resolve_aawm_route_host_attribution": MagicMock(
                    side_effect=RuntimeError("boom")
                )
            },
        ):
            request = _FakeRequest()
            result = _resolve_auto_agent_alias_route_host_attribution(request)
            assert result == _NEUTRAL_ATTRIBUTION


# ---------------------------------------------------------------------------
# _aresolve_auto_agent_alias_route_host_attribution (async)
# ---------------------------------------------------------------------------


class TestAsyncHostAttribution:
    async def test_prefers_cached_request_state_attribution(self):
        fake_result = {
            "client_ip": "10.0.0.1",
            "client_ip_source": "x-forwarded-for",
            "host_name": "state-host",
            "host_name_source": "state",
        }
        request = _FakeRequest()
        request.client = SimpleNamespace(host="10.0.0.1")
        request.state = SimpleNamespace(aawm_route_host_attribution=fake_result)

        async def fail_async(req, *, allow_blocking_lookup):
            raise AssertionError("state must win")

        with patch.dict(
            _aresolve_auto_agent_alias_route_host_attribution.__globals__,
            {"aresolve_aawm_route_host_attribution": fail_async},
        ):
            result = await _aresolve_auto_agent_alias_route_host_attribution(request)
            assert result == fake_result

    async def test_stale_request_state_delegates_to_async_resolver(self):
        stale_result = {
            "client_ip": "10.0.0.1",
            "client_ip_source": "request_client",
            "host_name": "stale-host",
            "host_name_source": "state",
        }
        fresh_result = {
            "client_ip": "10.0.0.2",
            "client_ip_source": "request_client",
            "host_name": "fresh-host",
            "host_name_source": "reverse-dns",
        }
        request = _FakeRequest()
        request.client = SimpleNamespace(host="10.0.0.2")
        request.state = SimpleNamespace(aawm_route_host_attribution=stale_result)
        calls = []

        async def fake_aresolve(req, *, allow_blocking_lookup):
            calls.append((req, allow_blocking_lookup))
            return fresh_result

        with patch.dict(
            _aresolve_auto_agent_alias_route_host_attribution.__globals__,
            {"aresolve_aawm_route_host_attribution": fake_aresolve},
        ):
            result = await _aresolve_auto_agent_alias_route_host_attribution(request)

        assert result == fresh_result
        assert calls == [(request, True)]

    async def test_delegates_to_async_resolver(self):
        fake_result = {
            "client_ip": "192.168.1.1",
            "client_ip_source": "peer",
            "host_name": "async.local",
            "host_name_source": "dns",
        }

        async def fake_aresolve(req, *, allow_blocking_lookup):
            assert allow_blocking_lookup is True
            return fake_result

        with patch.dict(
            _aresolve_auto_agent_alias_route_host_attribution.__globals__,
            {"aresolve_aawm_route_host_attribution": fake_aresolve},
        ):
            request = _FakeRequest()
            result = await _aresolve_auto_agent_alias_route_host_attribution(request)
            assert result == fake_result

    async def test_fail_closed_on_exception(self):
        async def fake_aresolve_fail(req, *, allow_blocking_lookup):
            raise ConnectionError("dns down")

        with patch.dict(
            _aresolve_auto_agent_alias_route_host_attribution.__globals__,
            {"aresolve_aawm_route_host_attribution": fake_aresolve_fail},
        ):
            request = _FakeRequest()
            result = await _aresolve_auto_agent_alias_route_host_attribution(request)
            assert result == _NEUTRAL_ATTRIBUTION


# ---------------------------------------------------------------------------
# install() contract
# ---------------------------------------------------------------------------


class TestInstall:
    def test_install_publishes_all_host_functions(self):
        host: dict[str, Any] = {}
        request_metadata.install(host)
        for name in request_metadata._HOST_FUNCTION_NAMES:
            assert name in host, f"missing {name} in host_globals"
            assert host[name] is getattr(request_metadata, name)
            assert host[name].__globals__ is host

    def test_install_copies_seam_when_configured(self):
        host: dict[str, Any] = {}
        sentinel = MagicMock()
        request_metadata._extract_passthrough_session_id = sentinel
        request_metadata.install(host)
        assert host.get("_extract_passthrough_session_id") is sentinel

    def test_install_does_not_overwrite_existing_seam(self):
        existing = MagicMock()
        host: dict[str, Any] = {"_extract_passthrough_session_id": existing}
        request_metadata._extract_passthrough_session_id = MagicMock()
        request_metadata.install(host)
        assert host["_extract_passthrough_session_id"] is existing

    def test_install_seeds_unconfigured_seam_keys_as_none(self):
        request_metadata._extract_passthrough_session_id = None
        request_metadata._get_codex_auto_agent_header = None
        host: dict[str, Any] = {}

        request_metadata.install(host)

        assert "_extract_passthrough_session_id" in host
        assert host["_extract_passthrough_session_id"] is None
        assert "_get_codex_auto_agent_header" in host
        assert host["_get_codex_auto_agent_header"] is None

    def test_post_install_unconfigured_calls_raise_runtime_error(self):
        request_metadata._extract_passthrough_session_id = None
        request_metadata._get_codex_auto_agent_header = None
        host: dict[str, Any] = {}
        request_metadata.install(host)

        with pytest.raises(
            RuntimeError,
            match="missing extract_passthrough_session_id",
        ):
            host["_extract_auto_agent_alias_session_id"](_FakeRequest(), {})

        with pytest.raises(
            RuntimeError,
            match="missing get_codex_auto_agent_header",
        ):
            host["_extract_auto_agent_alias_client_product_label"](
                _FakeRequest(),
                {},
            )

    def test_optimized_python_preserves_unconfigured_runtime_error(self):
        script = """
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import request_metadata

request_metadata._extract_passthrough_session_id = None
request_metadata._get_codex_auto_agent_header = None
host = {}
request_metadata.install(host)
try:
    host["_extract_auto_agent_alias_session_id"](object(), {})
except RuntimeError as exc:
    if "missing extract_passthrough_session_id" not in str(exc):
        raise
else:
    raise SystemExit("expected RuntimeError")
"""
        result = subprocess.run(
            [sys.executable, "-O", "-c", script],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr or result.stdout

    def test_install_then_configure_supports_behavioral_host_calls(self):
        passthrough = MagicMock(return_value="session-from-runtime")
        host: dict[str, Any] = {
            "_clean_codex_auth_value": _clean_codex_auth_value,
            "_safe_get_request_headers": request_metadata._safe_get_request_headers,
        }
        request_metadata.install(host)
        configure_request_metadata_runtime(
            extract_passthrough_session_id=passthrough,
            get_codex_auto_agent_header=_fake_get_header,
        )

        request = _FakeRequest(headers={"user-agent": "codex-cli/1.2.3"})
        assert (
            host["_extract_auto_agent_alias_session_id"](request, {})
            == "session-from-runtime"
        )
        assert (
            host["_extract_auto_agent_alias_client_product_label"](request, {})
            == "Codex/1.2.3"
        )
        passthrough.assert_called_once_with(request, {})

    def test_configure_preserves_existing_host_lane_key_helper(self):
        existing_header = MagicMock(return_value="host-header-session")
        configured_header = MagicMock(return_value="configured-header-session")
        passthrough = MagicMock(return_value=None)
        host: dict[str, Any] = {
            "_clean_codex_auth_value": _clean_codex_auth_value,
            "_safe_get_request_headers": request_metadata._safe_get_request_headers,
            "_get_codex_auto_agent_header": existing_header,
        }
        request_metadata.install(host)
        configure_request_metadata_runtime(
            extract_passthrough_session_id=passthrough,
            get_codex_auto_agent_header=configured_header,
        )

        assert host["_get_codex_auto_agent_header"] is existing_header
        assert host["_extract_passthrough_session_id"] is passthrough
        assert (
            host["_extract_auto_agent_alias_session_id"](_FakeRequest(), {})
            == "host-header-session"
        )
        existing_header.assert_called()
        configured_header.assert_not_called()

    async def test_rebound_endpoint_and_attribution_with_minimal_host(self):
        sync_result = {
            "client_ip": "127.0.0.1",
            "client_ip_source": "peer",
            "host_name": "local.test",
            "host_name_source": "test",
        }
        async_result = {
            "client_ip": "127.0.0.2",
            "client_ip_source": "forwarded",
            "host_name": "async.test",
            "host_name_source": "test",
        }
        sync_resolver = MagicMock(return_value=sync_result)

        async def async_resolver(request, *, allow_blocking_lookup):
            assert allow_blocking_lookup is True
            return async_result

        with (
            patch.object(
                request_metadata,
                "resolve_aawm_route_host_attribution",
                sync_resolver,
            ),
            patch.object(
                request_metadata,
                "aresolve_aawm_route_host_attribution",
                async_resolver,
            ),
        ):
            host: dict[str, Any] = {}
            request_metadata.install(host)

        request = _FakeRequest(
            url="http://localhost/v1/messages?stream=true&secret=drop"
        )
        assert (
            host["_extract_auto_agent_alias_incoming_endpoint"](request)
            == "/v1/messages?stream=true"
        )
        assert (
            host["_resolve_auto_agent_alias_route_host_attribution"](request)
            == sync_result
        )
        assert (
            await host["_aresolve_auto_agent_alias_route_host_attribution"](
                request
            )
            == async_result
        )
        sync_resolver.assert_called_once_with(
            request,
            allow_blocking_lookup=False,
        )
