"""Direct tests for the Wave 6F Anthropic adapter dispatch gate."""

from __future__ import annotations

import ast
import inspect
from dataclasses import fields
from pathlib import Path
from typing import Any, Optional

import pytest

from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime import (
    anthropic_dispatch,
)
from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.anthropic_dispatch import (
    ANTHROPIC_DISPATCH_SEAM_DISPOSITION,
    AnthropicDispatchRuntime,
    try_dispatch_anthropic_adapter,
)


MODULE_PATH = Path(anthropic_dispatch.__file__).resolve()
GOD_MODULE_PATH = (
    MODULE_PATH.parents[2] / "llm_passthrough_endpoints.py"
)

# Ordered resolver seam names matching the god-module chain priority.
RESOLVER_SEAMS = (
    "resolve_xai_oauth",
    "resolve_grok_native_oauth",
    "resolve_openai_responses",
    "resolve_opencode_zen",
    "resolve_kimi",
    "resolve_alibaba",
    "resolve_nvidia",
    "resolve_openrouter_completion",
    "resolve_openrouter_responses",
)


class _FakeResponse:
    """Stand-in for a FastAPI Response."""

    def __init__(self, name: str) -> None:
        self.name = name


class _Recorder:
    """Builds resolver/handler callables that record invocation order."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.handler_kwargs: dict[str, dict[str, Any]] = {}

    def resolver(self, name: str, result: Optional[str] = None) -> Any:
        def _resolve(request_body: dict, endpoint: str) -> Optional[str]:
            self.calls.append(f"resolve:{name}")
            return result

        return _resolve

    def handler(self, name: str) -> Any:
        async def _handle(**kwargs: Any) -> _FakeResponse:
            self.calls.append(f"handle:{name}")
            self.handler_kwargs[name] = kwargs
            return _FakeResponse(name)

        return _handle


def _make_runtime(
    recorder: _Recorder,
    *,
    match: Optional[str] = None,
    is_responses: bool = False,
) -> AnthropicDispatchRuntime:
    """Build a runtime where exactly ``match`` resolver returns a model."""
    kwargs: dict[str, Any] = {}
    for seam in RESOLVER_SEAMS:
        kwargs[seam] = recorder.resolver(
            seam, result="some-model" if seam == match else None
        )
    kwargs["handle_xai_oauth_responses"] = recorder.handler("xai_responses")
    kwargs["handle_xai_oauth_completion"] = recorder.handler("xai_completion")
    kwargs["handle_grok_native_oauth_responses"] = recorder.handler("grok")
    kwargs["handle_openai_responses"] = recorder.handler("openai")
    kwargs["handle_opencode_zen"] = recorder.handler("opencode_zen")
    kwargs["handle_kimi"] = recorder.handler("kimi")
    kwargs["handle_alibaba"] = recorder.handler("alibaba")
    kwargs["handle_nvidia"] = recorder.handler("nvidia")
    kwargs["handle_openrouter_completion"] = recorder.handler("openrouter_completion")
    kwargs["handle_openrouter_responses"] = recorder.handler("openrouter_responses")
    kwargs["is_oa_xai_responses_model"] = lambda model: is_responses
    return AnthropicDispatchRuntime(**kwargs)


_DISPATCH_KWARGS: dict[str, Any] = {
    "endpoint": "/v1/messages",
    "request": object(),
    "fastapi_response": object(),
    "user_api_key_dict": object(),
    "prepared_request_body": {"model": "claude-3-5-sonnet"},
}


# ---------------------------------------------------------------------------
# Fall-through behavior
# ---------------------------------------------------------------------------


async def test_should_return_none_when_no_adapter_matches() -> None:
    recorder = _Recorder()
    runtime = _make_runtime(recorder, match=None)

    result = await try_dispatch_anthropic_adapter(runtime, **_DISPATCH_KWARGS)

    assert result is None
    # Every resolver consulted, in priority order, no handler invoked.
    assert recorder.calls == [f"resolve:{seam}" for seam in RESOLVER_SEAMS]


# ---------------------------------------------------------------------------
# Representative dispatch, including Kimi and Alibaba recognition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("match", "expected_handler"),
    [
        ("resolve_grok_native_oauth", "grok"),
        ("resolve_openai_responses", "openai"),
        ("resolve_opencode_zen", "opencode_zen"),
        ("resolve_kimi", "kimi"),
        ("resolve_alibaba", "alibaba"),
        ("resolve_nvidia", "nvidia"),
        ("resolve_openrouter_completion", "openrouter_completion"),
        ("resolve_openrouter_responses", "openrouter_responses"),
    ],
)
async def test_should_dispatch_selected_adapter_and_stop_chain(
    match: str, expected_handler: str
) -> None:
    recorder = _Recorder()
    runtime = _make_runtime(recorder, match=match)

    result = await try_dispatch_anthropic_adapter(runtime, **_DISPATCH_KWARGS)

    assert isinstance(result, _FakeResponse)
    assert result.name == expected_handler
    # Chain stops at the matching resolver; later resolvers not consulted.
    match_index = RESOLVER_SEAMS.index(match)
    expected_resolves = [
        f"resolve:{seam}" for seam in RESOLVER_SEAMS[: match_index + 1]
    ]
    assert recorder.calls == expected_resolves + [f"handle:{expected_handler}"]
    # Handler received the adapter_model and common kwargs.
    assert recorder.handler_kwargs[expected_handler]["adapter_model"] == "some-model"
    assert recorder.handler_kwargs[expected_handler]["endpoint"] == "/v1/messages"


async def test_should_recognize_kimi_adapter() -> None:
    recorder = _Recorder()
    runtime = _make_runtime(recorder, match="resolve_kimi")

    result = await try_dispatch_anthropic_adapter(runtime, **_DISPATCH_KWARGS)

    assert isinstance(result, _FakeResponse)
    assert result.name == "kimi"
    assert "handle:kimi" in recorder.calls


async def test_should_recognize_alibaba_adapter() -> None:
    recorder = _Recorder()
    runtime = _make_runtime(recorder, match="resolve_alibaba")

    result = await try_dispatch_anthropic_adapter(runtime, **_DISPATCH_KWARGS)

    assert isinstance(result, _FakeResponse)
    assert result.name == "alibaba"
    assert "handle:alibaba" in recorder.calls


async def test_should_route_xai_to_responses_when_responses_model() -> None:
    recorder = _Recorder()
    runtime = _make_runtime(
        recorder, match="resolve_xai_oauth", is_responses=True
    )

    result = await try_dispatch_anthropic_adapter(runtime, **_DISPATCH_KWARGS)

    assert isinstance(result, _FakeResponse)
    assert result.name == "xai_responses"
    assert recorder.calls == ["resolve:resolve_xai_oauth", "handle:xai_responses"]


async def test_should_route_xai_to_completion_when_not_responses_model() -> None:
    recorder = _Recorder()
    runtime = _make_runtime(
        recorder, match="resolve_xai_oauth", is_responses=False
    )

    result = await try_dispatch_anthropic_adapter(runtime, **_DISPATCH_KWARGS)

    assert isinstance(result, _FakeResponse)
    assert result.name == "xai_completion"
    assert recorder.calls == ["resolve:resolve_xai_oauth", "handle:xai_completion"]


# ---------------------------------------------------------------------------
# Ordering precedence
# ---------------------------------------------------------------------------


async def test_should_respect_priority_order_when_multiple_match() -> None:
    recorder = _Recorder()
    kwargs: dict[str, Any] = {}
    for seam in RESOLVER_SEAMS:
        # Every resolver matches; the earliest (xai) must win.
        kwargs[seam] = recorder.resolver(seam, result="some-model")
    kwargs["handle_xai_oauth_responses"] = recorder.handler("xai_responses")
    kwargs["handle_xai_oauth_completion"] = recorder.handler("xai_completion")
    kwargs["handle_grok_native_oauth_responses"] = recorder.handler("grok")
    kwargs["handle_openai_responses"] = recorder.handler("openai")
    kwargs["handle_opencode_zen"] = recorder.handler("opencode_zen")
    kwargs["handle_kimi"] = recorder.handler("kimi")
    kwargs["handle_alibaba"] = recorder.handler("alibaba")
    kwargs["handle_nvidia"] = recorder.handler("nvidia")
    kwargs["handle_openrouter_completion"] = recorder.handler("openrouter_completion")
    kwargs["handle_openrouter_responses"] = recorder.handler("openrouter_responses")
    kwargs["is_oa_xai_responses_model"] = lambda model: True
    runtime = AnthropicDispatchRuntime(**kwargs)

    result = await try_dispatch_anthropic_adapter(runtime, **_DISPATCH_KWARGS)

    assert isinstance(result, _FakeResponse)
    assert result.name == "xai_responses"
    assert recorder.calls == ["resolve:resolve_xai_oauth", "handle:xai_responses"]


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------


async def test_should_propagate_resolver_error() -> None:
    recorder = _Recorder()

    def _boom(request_body: dict, endpoint: str) -> Optional[str]:
        raise ValueError("resolver failure")

    runtime = _make_runtime(recorder, match=None)
    object.__setattr__(runtime, "resolve_kimi", _boom)

    with pytest.raises(ValueError, match="resolver failure"):
        await try_dispatch_anthropic_adapter(runtime, **_DISPATCH_KWARGS)


async def test_should_propagate_handler_error() -> None:
    recorder = _Recorder()
    runtime = _make_runtime(recorder, match="resolve_alibaba")

    async def _boom(**kwargs: Any) -> _FakeResponse:
        raise RuntimeError("handler failure")

    object.__setattr__(runtime, "handle_alibaba", _boom)

    with pytest.raises(RuntimeError, match="handler failure"):
        await try_dispatch_anthropic_adapter(runtime, **_DISPATCH_KWARGS)


# ---------------------------------------------------------------------------
# Signature / async parity
# ---------------------------------------------------------------------------


def test_try_dispatch_should_be_async_coroutine_function() -> None:
    assert inspect.iscoroutinefunction(try_dispatch_anthropic_adapter)


def test_try_dispatch_signature_should_match_contract() -> None:
    sig = inspect.signature(try_dispatch_anthropic_adapter)
    params = list(sig.parameters)
    assert params[0] == "runtime"
    assert set(params[1:]) == {
        "endpoint",
        "request",
        "fastapi_response",
        "user_api_key_dict",
        "prepared_request_body",
    }
    # All non-runtime params are keyword-only.
    for name in params[1:]:
        assert sig.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


def test_runtime_seam_disposition_should_be_complete_and_explicit() -> None:
    assert (
        anthropic_dispatch.ANTHROPIC_DISPATCH_SEAM_DISPOSITION
        == ANTHROPIC_DISPATCH_SEAM_DISPOSITION
    )
    assert {field.name for field in fields(AnthropicDispatchRuntime)} == set(
        ANTHROPIC_DISPATCH_SEAM_DISPOSITION
    )
    assert all(
        disposition.startswith("runtime.")
        for disposition in ANTHROPIC_DISPATCH_SEAM_DISPOSITION.values()
    )


# ---------------------------------------------------------------------------
# No god-module import at module scope
# ---------------------------------------------------------------------------


def test_module_should_not_import_god_module_at_module_scope() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    imported_modules: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)

    assert not any(
        module.endswith("llm_passthrough_endpoints")
        for module in imported_modules
    )


def test_module_should_not_use_wildcard_imports() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not any(alias.name == "*" for alias in node.names)
