"""Wave 7 owner tests for provider-neutral config-refresh DI runtime.

Validates that ``config_refresh`` works correctly through the injected
``ConfigRefreshRuntime`` seam, preserving response/status/auth semantics
identically to the direct-import fallback path.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any, Optional

import pytest
from fastapi import HTTPException

from litellm.proxy.pass_through_endpoints.aawm_alias_routing import config_refresh
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_refresh import (
    ConfigRefreshRuntime,
    aawm_alias_config_refresh_route,
    configure_config_refresh_runtime,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _CompileError(Exception):
    """Stand-in for ConfigCompileError."""


@dataclass
class _FakeSnapshot:
    config_hash: str
    config_version: int


class _FakeRequest:
    """Minimal async-JSON request stub."""

    def __init__(self, body: Any) -> None:
        self._body = body

    async def json(self) -> Any:
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


def _make_runtime(
    *,
    compile_result: Any = None,
    compile_error: Optional[Exception] = None,
    active_snapshot: Optional[_FakeSnapshot] = None,
    default_yaml: str = "defaults: {}\n",
) -> tuple[ConfigRefreshRuntime, dict[str, Any]]:
    """Build a runtime with controllable fakes; returns (runtime, calls_log)."""
    calls: dict[str, Any] = {"set_snapshot": None, "compile_calls": []}

    def _compile(yaml_str: str) -> Any:
        calls["compile_calls"].append(yaml_str)
        if compile_error is not None:
            raise compile_error
        return compile_result

    state: dict[str, Optional[_FakeSnapshot]] = {"active": active_snapshot}

    def _get_active() -> Optional[_FakeSnapshot]:
        return state["active"]

    def _set_active(snap: Any) -> Optional[Any]:
        prev = state["active"]
        state["active"] = snap
        calls["set_snapshot"] = snap
        return prev

    def _load_default() -> str:
        return default_yaml

    runtime = ConfigRefreshRuntime(
        compile_yaml=_compile,
        get_active_snapshot=_get_active,
        set_active_snapshot=_set_active,
        load_default_source_yaml=_load_default,
        compile_error_types=(_CompileError,),
    )
    return runtime, calls


@pytest.fixture(autouse=True)
def _reset_runtime():
    """Ensure runtime is cleared before and after each test."""
    previous = config_refresh._runtime
    config_refresh._runtime = None
    yield
    config_refresh._runtime = previous


# ---------------------------------------------------------------------------
# Structure / contract tests
# ---------------------------------------------------------------------------


def test_configure_is_sync() -> None:
    assert not inspect.iscoroutinefunction(configure_config_refresh_runtime)


def test_handler_is_async() -> None:
    assert inspect.iscoroutinefunction(aawm_alias_config_refresh_route)


def test_runtime_dataclass_is_frozen() -> None:
    assert ConfigRefreshRuntime.__dataclass_params__.frozen  # type: ignore[attr-defined]


def test_unconfigured_runtime_is_none() -> None:
    assert config_refresh._get_runtime() is None


# ---------------------------------------------------------------------------
# DI path: successful compile + activate
# ---------------------------------------------------------------------------


def test_di_valid_inline_yaml_activates_snapshot() -> None:
    snapshot = _FakeSnapshot(config_hash="abc123", config_version=1)
    runtime, calls = _make_runtime(compile_result=snapshot)
    configure_config_refresh_runtime(runtime=runtime)

    request = _FakeRequest({"yaml": "aliases: []"})
    result = asyncio.run(aawm_alias_config_refresh_route(request))  # type: ignore[arg-type]

    assert result["changed"] is True
    assert result["attempted_config_hash"] == "abc123"
    assert result["active_config_hash"] == "abc123"
    assert result["config_version"] == 1
    assert "activated_at" in result
    assert calls["set_snapshot"] is snapshot


def test_di_noop_refresh_preserves_prior_object() -> None:
    existing = _FakeSnapshot(config_hash="same", config_version=2)
    attempted = _FakeSnapshot(config_hash="same", config_version=2)
    runtime, calls = _make_runtime(
        compile_result=attempted,
        active_snapshot=existing,
    )
    configure_config_refresh_runtime(runtime=runtime)

    request = _FakeRequest({"yaml": "aliases: []"})
    result = asyncio.run(aawm_alias_config_refresh_route(request))  # type: ignore[arg-type]

    assert result["changed"] is False
    assert result["active_config_hash"] == "same"
    # Must NOT replace the snapshot object (in-flight readers hold reference).
    assert calls["set_snapshot"] is None


# ---------------------------------------------------------------------------
# DI path: compile failure -> 400, last-known-good preserved
# ---------------------------------------------------------------------------


def test_di_compile_error_returns_400_with_lkg() -> None:
    lkg = _FakeSnapshot(config_hash="good_hash", config_version=5)
    runtime, _ = _make_runtime(
        compile_error=_CompileError("bad config"),
        active_snapshot=lkg,
    )
    configure_config_refresh_runtime(runtime=runtime)

    request = _FakeRequest({"yaml": "bad: ["})
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(aawm_alias_config_refresh_route(request))  # type: ignore[arg-type]

    assert exc_info.value.status_code == 400
    detail = exc_info.value.detail
    assert "failed to compile" in detail["error"]
    assert detail["active_config_hash"] == "good_hash"
    assert detail["config_version"] == 5


def test_di_compile_error_no_lkg_omits_hash_fields() -> None:
    runtime, _ = _make_runtime(
        compile_error=_CompileError("bad"),
        active_snapshot=None,
    )
    configure_config_refresh_runtime(runtime=runtime)

    request = _FakeRequest({"yaml": "x"})
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(aawm_alias_config_refresh_route(request))  # type: ignore[arg-type]

    detail = exc_info.value.detail
    assert "active_config_hash" not in detail
    assert "config_version" not in detail


# ---------------------------------------------------------------------------
# DI path: request body edge cases
# ---------------------------------------------------------------------------


def test_di_non_string_yaml_field_returns_400() -> None:
    runtime, _ = _make_runtime()
    configure_config_refresh_runtime(runtime=runtime)

    request = _FakeRequest({"yaml": 12345})
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(aawm_alias_config_refresh_route(request))  # type: ignore[arg-type]

    assert exc_info.value.status_code == 400
    assert "must be a string" in exc_info.value.detail["error"]


def test_di_unparseable_json_body_refreshes_default_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _FakeSnapshot(config_hash="from_default", config_version=1)
    runtime, calls = _make_runtime(compile_result=snapshot)
    monkeypatch.setattr(
        config_refresh,
        "_load_full_default_directory_snapshot",
        lambda: (snapshot, ("alpha.yaml", "nested/zulu.yaml")),
    )
    configure_config_refresh_runtime(runtime=runtime)

    # json() raises -> body treated as {} -> inline_yaml is None -> default loader
    request = _FakeRequest(ValueError("not json"))
    result = asyncio.run(aawm_alias_config_refresh_route(request))  # type: ignore[arg-type]

    assert result["changed"] is True
    assert result["files"] == ["alpha.yaml", "nested/zulu.yaml"]
    assert calls["compile_calls"] == []


def test_di_default_directory_error_returns_400(monkeypatch: pytest.MonkeyPatch) -> None:
    def _broken_directory_loader() -> tuple[_FakeSnapshot, tuple[str, ...]]:
        raise config_refresh._AawmAliasConfigCompileError("disk gone")

    lkg = _FakeSnapshot(config_hash="good_hash", config_version=5)
    runtime, calls = _make_runtime(
        active_snapshot=lkg,
        compile_result=_FakeSnapshot(config_hash="unused", config_version=0),
    )
    monkeypatch.setattr(
        config_refresh,
        "_load_full_default_directory_snapshot",
        _broken_directory_loader,
    )
    configure_config_refresh_runtime(runtime=runtime)

    request = _FakeRequest({})
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(aawm_alias_config_refresh_route(request))  # type: ignore[arg-type]

    assert exc_info.value.status_code == 400
    assert "last-known-good" in exc_info.value.detail["error"]
    assert exc_info.value.detail["active_config_hash"] == "good_hash"
    assert exc_info.value.detail["config_version"] == 5
    assert calls["set_snapshot"] is None


# ---------------------------------------------------------------------------
# DI path: secret safety
# ---------------------------------------------------------------------------


def test_di_response_never_echoes_yaml_content() -> None:
    snapshot = _FakeSnapshot(config_hash="h1", config_version=1)
    runtime, _ = _make_runtime(compile_result=snapshot)
    configure_config_refresh_runtime(runtime=runtime)

    secret_yaml = "aliases: []\n# secret: sk-live-SUPERSECRET"
    request = _FakeRequest({"yaml": secret_yaml})
    result = asyncio.run(aawm_alias_config_refresh_route(request))  # type: ignore[arg-type]

    assert "sk-live-SUPERSECRET" not in str(result)


# ---------------------------------------------------------------------------
# Fallback path: unconfigured runtime uses direct imports (smoke)
# ---------------------------------------------------------------------------


def test_fallback_path_still_importable() -> None:
    """The fallback helper exists and is async (backward compat)."""
    assert inspect.iscoroutinefunction(config_refresh._refresh_via_direct_imports)
