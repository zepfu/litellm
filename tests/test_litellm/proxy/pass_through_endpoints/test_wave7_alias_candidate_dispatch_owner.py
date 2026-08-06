"""Wave 7 module-local tests for alias_candidate_dispatch owner functions.

Write scope: this file only.
"""

from __future__ import annotations

from typing import Any, Generator
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.alias_candidate_dispatch import (
    ALIAS_CANDIDATE_DISPATCH_SEAM_DISPOSITION,
    AliasCandidateDispatchRuntime,
    _dispatch_auto_agent_alias_candidate_request,
    _perform_anthropic_auto_agent_alias_candidate_request,
    install,
)
import litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.alias_candidate_dispatch as _mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_runtime(**overrides: Any) -> AliasCandidateDispatchRuntime:
    """Build a fully-mocked runtime with sensible defaults."""
    defaults: dict[str, Any] = {
        "handle_openai_responses": AsyncMock(return_value="openai_resp"),
        "handle_openrouter_completion": AsyncMock(return_value="or_completion_resp"),
        "handle_openrouter_responses": AsyncMock(return_value="or_responses_resp"),
        "handle_xai_oauth_responses": AsyncMock(return_value="xai_resp"),
        "handle_grok_native_oauth_responses": AsyncMock(return_value="grok_resp"),
        "handle_opencode_zen": AsyncMock(return_value="opencode_resp"),
        "handle_kimi_chat_completions": AsyncMock(return_value="kimi_resp"),
        "handle_alibaba_token_plan": AsyncMock(return_value="alibaba_resp"),
        "normalize_native_model_alias": MagicMock(
            side_effect=lambda body: (body, None)
        ),
        "prepare_context_1m_native": MagicMock(
            side_effect=lambda *, request, request_body, custom_headers: (
                request_body,
                custom_headers,
                None,
            )
        ),
        "safe_set_request_parsed_body": MagicMock(),
        "perform_native_passthrough": AsyncMock(return_value="native_resp"),
        "provider_native": "codex_native",
        "provider_openrouter": "openrouter",
        "provider_xai": "xai",
        "provider_opencode": "opencode",
        "provider_kimi": "kimi_code",
        "provider_alibaba": "alibaba_token_plan",
        "anthropic_beta_header_name": "anthropic-beta",
    }
    defaults.update(overrides)
    return AliasCandidateDispatchRuntime(**defaults)


def _install_runtime(rt: AliasCandidateDispatchRuntime) -> None:
    """Install runtime into the module-level slot."""
    _mod._runtime = rt


def _clear_runtime() -> None:
    _mod._runtime = None


@pytest.fixture(autouse=True)
def _restore_alias_candidate_dispatch_runtime() -> Generator[None, None, None]:
    saved = _mod._runtime
    _mod._runtime = None
    yield
    _mod._runtime = saved


@pytest.fixture
def opus_5_stable_effort_metadata(monkeypatch):
    """Deterministic claude-opus-5 metadata mirroring the CFG-013 Body A
    catalog entry, so Body C dispatch tests pass regardless of catalog state.
    """
    import litellm

    entry = {
        "litellm_provider": "anthropic",
        "mode": "chat",
        "max_input_tokens": 1000000,
        "max_output_tokens": 128000,
        "max_tokens": 128000,
        "supports_reasoning": True,
        "supports_max_reasoning_effort": True,
    }
    monkeypatch.setitem(litellm.model_cost, "claude-opus-5", entry)
    monkeypatch.setitem(litellm.model_cost, "anthropic/claude-opus-5", entry)
    return entry


# ---------------------------------------------------------------------------
# _dispatch_auto_agent_alias_candidate_request
# ---------------------------------------------------------------------------


class TestDispatchAutoAgentAliasCandidateRequest:
    @pytest.mark.asyncio
    async def test_provider_handler_match(self):
        handler = AsyncMock(return_value="provider_hit")
        default = AsyncMock(return_value="default")
        result = await _dispatch_auto_agent_alias_candidate_request(
            candidate={"provider": "prov_a", "route_family": ""},
            provider_handlers={"prov_a": handler},
            default_handler=default,
        )
        assert result == "provider_hit"
        handler.assert_awaited_once()
        default.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_default_handler_fallback(self):
        default = AsyncMock(return_value="default_resp")
        result = await _dispatch_auto_agent_alias_candidate_request(
            candidate={"provider": "unknown", "route_family": ""},
            provider_handlers={"prov_a": AsyncMock()},
            default_handler=default,
        )
        assert result == "default_resp"
        default.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_route_family_specific_match(self):
        family_handler = AsyncMock(return_value="family_hit")
        wildcard = AsyncMock(return_value="wildcard")
        provider_handler = AsyncMock(return_value="provider")
        default = AsyncMock(return_value="default")
        result = await _dispatch_auto_agent_alias_candidate_request(
            candidate={"provider": "prov_x", "route_family": "specific_route"},
            provider_handlers={"prov_x": provider_handler},
            default_handler=default,
            route_family_handlers={
                "prov_x": {"specific_route": family_handler, "*": wildcard},
            },
        )
        assert result == "family_hit"
        family_handler.assert_awaited_once()
        wildcard.assert_not_awaited()
        provider_handler.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_route_family_wildcard_fallback(self):
        wildcard = AsyncMock(return_value="wildcard_hit")
        default = AsyncMock(return_value="default")
        result = await _dispatch_auto_agent_alias_candidate_request(
            candidate={"provider": "prov_x", "route_family": "unknown_route"},
            provider_handlers={},
            default_handler=default,
            route_family_handlers={"prov_x": {"*": wildcard}},
        )
        assert result == "wildcard_hit"
        wildcard.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_route_family_provider_not_in_map_falls_to_provider_handlers(self):
        provider_handler = AsyncMock(return_value="provider_hit")
        default = AsyncMock(return_value="default")
        result = await _dispatch_auto_agent_alias_candidate_request(
            candidate={"provider": "prov_b", "route_family": "some"},
            provider_handlers={"prov_b": provider_handler},
            default_handler=default,
            route_family_handlers={"prov_x": {"*": AsyncMock()}},
        )
        assert result == "provider_hit"

    @pytest.mark.asyncio
    async def test_empty_provider_and_route_family_uses_default(self):
        default = AsyncMock(return_value="default_resp")
        result = await _dispatch_auto_agent_alias_candidate_request(
            candidate={},
            provider_handlers={},
            default_handler=default,
        )
        assert result == "default_resp"

    @pytest.mark.asyncio
    async def test_none_route_family_handlers_skips_family_lookup(self):
        default = AsyncMock(return_value="default_resp")
        result = await _dispatch_auto_agent_alias_candidate_request(
            candidate={"provider": "x", "route_family": "y"},
            provider_handlers={},
            default_handler=default,
            route_family_handlers=None,
        )
        assert result == "default_resp"


# ---------------------------------------------------------------------------
# _perform_anthropic_auto_agent_alias_candidate_request
# ---------------------------------------------------------------------------


class TestPerformAnthropicAutoAgentAliasCandidateRequest:
    @pytest.mark.asyncio
    async def test_fails_closed_without_runtime(self):
        with pytest.raises(RuntimeError, match="runtime not installed"):
            await _perform_anthropic_auto_agent_alias_candidate_request(
                endpoint="/v1/messages",
                request=MagicMock(),
                fastapi_response=MagicMock(),
                user_api_key_dict=MagicMock(),
                candidate={"model": "m", "provider": "codex_native"},
                candidate_body={"model": "m"},
                target_url="https://api.anthropic.com",
                custom_headers={},
            )

    @pytest.mark.asyncio
    async def test_openai_provider_dispatches_to_openai_handler(self):
        rt = _make_runtime()
        _install_runtime(rt)
        result = await _perform_anthropic_auto_agent_alias_candidate_request(
            endpoint="/v1/messages",
            request=MagicMock(),
            fastapi_response=MagicMock(),
            user_api_key_dict=MagicMock(),
            candidate={"model": "gpt-4o", "provider": "codex_native", "route_family": ""},
            candidate_body={"model": "gpt-4o"},
            target_url="https://api.anthropic.com",
            custom_headers={},
        )
        assert result == "openai_resp"
        rt.handle_openai_responses.assert_awaited_once()
        kw = rt.handle_openai_responses.call_args.kwargs
        assert kw["adapter_model"] == "gpt-4o"
        assert kw["use_alias_candidate_probe"] is True

    @pytest.mark.asyncio
    async def test_openrouter_completion_route_family(self):
        rt = _make_runtime()
        _install_runtime(rt)
        result = await _perform_anthropic_auto_agent_alias_candidate_request(
            endpoint="/v1/messages",
            request=MagicMock(),
            fastapi_response=MagicMock(),
            user_api_key_dict=MagicMock(),
            candidate={
                "model": "or-model",
                "provider": "openrouter",
                "route_family": "anthropic_openrouter_completion_adapter",
            },
            candidate_body={"model": "or-model"},
            target_url="https://api.anthropic.com",
            custom_headers={},
        )
        assert result == "or_completion_resp"
        rt.handle_openrouter_completion.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_openrouter_wildcard_uses_responses_handler(self):
        rt = _make_runtime()
        _install_runtime(rt)
        result = await _perform_anthropic_auto_agent_alias_candidate_request(
            endpoint="/v1/messages",
            request=MagicMock(),
            fastapi_response=MagicMock(),
            user_api_key_dict=MagicMock(),
            candidate={
                "model": "or-model",
                "provider": "openrouter",
                "route_family": "some_other_family",
            },
            candidate_body={"model": "or-model"},
            target_url="https://api.anthropic.com",
            custom_headers={},
        )
        assert result == "or_responses_resp"
        rt.handle_openrouter_responses.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_xai_oauth_route_family(self):
        rt = _make_runtime()
        _install_runtime(rt)
        result = await _perform_anthropic_auto_agent_alias_candidate_request(
            endpoint="/v1/messages",
            request=MagicMock(),
            fastapi_response=MagicMock(),
            user_api_key_dict=MagicMock(),
            candidate={
                "model": "grok-3",
                "provider": "xai",
                "route_family": "anthropic_xai_oauth_responses_adapter",
            },
            candidate_body={"model": "grok-3"},
            target_url="https://api.anthropic.com",
            custom_headers={},
        )
        assert result == "xai_resp"
        rt.handle_xai_oauth_responses.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_xai_wildcard_uses_grok_native(self):
        rt = _make_runtime()
        _install_runtime(rt)
        result = await _perform_anthropic_auto_agent_alias_candidate_request(
            endpoint="/v1/messages",
            request=MagicMock(),
            fastapi_response=MagicMock(),
            user_api_key_dict=MagicMock(),
            candidate={
                "model": "grok-3",
                "provider": "xai",
                "route_family": "grok_native_oauth",
            },
            candidate_body={"model": "grok-3"},
            target_url="https://api.anthropic.com",
            custom_headers={},
        )
        assert result == "grok_resp"
        rt.handle_grok_native_oauth_responses.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_opencode_provider(self):
        rt = _make_runtime()
        _install_runtime(rt)
        result = await _perform_anthropic_auto_agent_alias_candidate_request(
            endpoint="/v1/messages",
            request=MagicMock(),
            fastapi_response=MagicMock(),
            user_api_key_dict=MagicMock(),
            candidate={"model": "oc-model", "provider": "opencode", "route_family": ""},
            candidate_body={"model": "oc-model"},
            target_url="https://api.anthropic.com",
            custom_headers={},
        )
        assert result == "opencode_resp"

    @pytest.mark.asyncio
    async def test_kimi_provider(self):
        rt = _make_runtime()
        _install_runtime(rt)
        result = await _perform_anthropic_auto_agent_alias_candidate_request(
            endpoint="/v1/messages",
            request=MagicMock(),
            fastapi_response=MagicMock(),
            user_api_key_dict=MagicMock(),
            candidate={"model": "kimi-k2", "provider": "kimi_code", "route_family": ""},
            candidate_body={"model": "kimi-k2"},
            target_url="https://api.anthropic.com",
            custom_headers={},
        )
        assert result == "kimi_resp"

    @pytest.mark.asyncio
    async def test_alibaba_provider(self):
        rt = _make_runtime()
        _install_runtime(rt)
        result = await _perform_anthropic_auto_agent_alias_candidate_request(
            endpoint="/v1/messages",
            request=MagicMock(),
            fastapi_response=MagicMock(),
            user_api_key_dict=MagicMock(),
            candidate={
                "model": "qwen-max",
                "provider": "alibaba_token_plan",
                "route_family": "",
            },
            candidate_body={"model": "qwen-max"},
            target_url="https://api.anthropic.com",
            custom_headers={},
        )
        assert result == "alibaba_resp"

    @pytest.mark.asyncio
    async def test_unknown_provider_falls_to_native(self):
        rt = _make_runtime()
        _install_runtime(rt)
        req = MagicMock()
        result = await _perform_anthropic_auto_agent_alias_candidate_request(
            endpoint="/v1/messages",
            request=req,
            fastapi_response=MagicMock(),
            user_api_key_dict=MagicMock(),
            candidate={"model": "claude-x", "provider": "anthropic", "route_family": ""},
            candidate_body={"model": "claude-x"},
            target_url="https://api.anthropic.com",
            custom_headers={"x-api-key": "k"},
        )
        assert result == "native_resp"
        rt.perform_native_passthrough.assert_awaited_once()
        rt.safe_set_request_parsed_body.assert_called_once()

    @pytest.mark.asyncio
    async def test_native_context_1m_blocks_beta_header(self):
        rt = _make_runtime(
            prepare_context_1m_native=MagicMock(
                side_effect=lambda *, request, request_body, custom_headers: (
                    request_body,
                    custom_headers,
                    "claude-sonnet-4-20250514[1m]",
                )
            ),
        )
        _install_runtime(rt)
        await _perform_anthropic_auto_agent_alias_candidate_request(
            endpoint="/v1/messages",
            request=MagicMock(),
            fastapi_response=MagicMock(),
            user_api_key_dict=MagicMock(),
            candidate={"model": "claude-x", "provider": "unknown", "route_family": ""},
            candidate_body={"model": "claude-x"},
            target_url="https://api.anthropic.com",
            custom_headers={},
        )
        kw = rt.perform_native_passthrough.call_args.kwargs
        assert kw["blocked_pass_through_prefixed_headers"] == ["anthropic-beta"]

    @pytest.mark.asyncio
    async def test_native_no_context_1m_no_blocked_headers(self):
        rt = _make_runtime()
        _install_runtime(rt)
        await _perform_anthropic_auto_agent_alias_candidate_request(
            endpoint="/v1/messages",
            request=MagicMock(),
            fastapi_response=MagicMock(),
            user_api_key_dict=MagicMock(),
            candidate={"model": "claude-x", "provider": "unknown", "route_family": ""},
            candidate_body={"model": "claude-x"},
            target_url="https://api.anthropic.com",
            custom_headers={},
        )
        kw = rt.perform_native_passthrough.call_args.kwargs
        assert kw["blocked_pass_through_prefixed_headers"] is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("effort", "expected_thinking_type"),
        [
            ("high", "enabled"),
            ("xhigh", "enabled"),
            ("max", "enabled"),
            ("none", None),
        ],
    )
    async def test_native_reasoning_effort_maps_schema_valid_thinking(
        self, effort, expected_thinking_type
    ):
        """CFG-006: native path normalizes effort, strips raw field, maps valid thinking."""
        rt = _make_runtime()
        _install_runtime(rt)
        candidate_body = {
            "model": "claude-sonnet-4-5",
            "reasoning_effort": effort,
        }
        await _perform_anthropic_auto_agent_alias_candidate_request(
            endpoint="/v1/messages",
            request=MagicMock(),
            fastapi_response=MagicMock(),
            user_api_key_dict=MagicMock(),
            candidate={
                "model": "claude-sonnet-4-5",
                "provider": "unknown",
                "route_family": "",
            },
            candidate_body=candidate_body,
            target_url="https://api.anthropic.com",
            custom_headers={},
        )
        sent_body = rt.safe_set_request_parsed_body.call_args.args[1]
        assert "reasoning_effort" not in sent_body
        if expected_thinking_type is None:
            assert "thinking" not in sent_body
        else:
            assert sent_body["thinking"]["type"] == expected_thinking_type
            assert isinstance(sent_body["thinking"].get("budget_tokens"), int)
        assert candidate_body["reasoning_effort"] == effort  # original not mutated


# ---------------------------------------------------------------------------
# install()
# ---------------------------------------------------------------------------


class TestInstall:
    def test_install_publishes_owned_symbols(self):
        host: dict[str, Any] = {}
        install(host)
        assert "_dispatch_auto_agent_alias_candidate_request" in host
        assert "_perform_anthropic_auto_agent_alias_candidate_request" in host
        assert "AliasCandidateDispatchRuntime" in host

    def test_install_with_runtime_activates_executor(self):
        rt = _make_runtime()
        host: dict[str, Any] = {}
        install(host, runtime=rt)
        assert _mod._runtime is rt

    def test_install_without_runtime_leaves_slot_none(self):
        _clear_runtime()
        host: dict[str, Any] = {}
        install(host)
        assert _mod._runtime is None


# ---------------------------------------------------------------------------
# Seam disposition completeness
# ---------------------------------------------------------------------------


class TestSeamDisposition:
    def test_disposition_covers_all_runtime_fields(self):
        import dataclasses

        field_names = {f.name for f in dataclasses.fields(AliasCandidateDispatchRuntime)}
        assert field_names == set(ALIAS_CANDIDATE_DISPATCH_SEAM_DISPOSITION.keys())

    def test_disposition_values_reference_runtime(self):
        for key, value in ALIAS_CANDIDATE_DISPATCH_SEAM_DISPOSITION.items():
            assert value == f"runtime.{key}"


# ---------------------------------------------------------------------------
# CFG-013 Body C: native alias-candidate dispatch integration
# ---------------------------------------------------------------------------


class TestCfg013BodyCNativeDispatch:
    """Native Anthropic alias-candidate shaping for stable-effort models.

    Canonical ``claude-opus-5`` with ``reasoning_effort: max`` must egress as
    adaptive ``thinking`` plus merged ``output_config.effort: max`` via the
    shared Body B mapper, without mutating the input payload, and without
    leaking transformed Opus state into a later Terra attempt.
    """

    async def _run_native(
        self,
        candidate_body: dict[str, Any],
        *,
        candidate_model: str = "claude-opus-5",
        provider: str = "anthropic",
        **runtime_overrides: Any,
    ) -> dict[str, Any]:
        rt = _make_runtime(**runtime_overrides)
        _install_runtime(rt)
        await _perform_anthropic_auto_agent_alias_candidate_request(
            endpoint="/v1/messages",
            request=MagicMock(),
            fastapi_response=MagicMock(),
            user_api_key_dict=MagicMock(),
            candidate={
                "model": candidate_model,
                "provider": provider,
                "route_family": "",
            },
            candidate_body=candidate_body,
            target_url="https://api.anthropic.com",
            custom_headers={},
        )
        return rt.safe_set_request_parsed_body.call_args.args[1]

    @pytest.mark.asyncio
    async def test_opus_5_max_shapes_adaptive_thinking_and_output_config(
        self, opus_5_stable_effort_metadata
    ):
        """Opus 5 max egress: top-level reasoning_effort removed, adaptive
        thinking plus output_config.effort=max emitted via shared mapper."""
        sent_body = await self._run_native(
            {"model": "claude-opus-5", "reasoning_effort": "max"}
        )
        assert "reasoning_effort" not in sent_body
        assert sent_body["thinking"] == {"type": "adaptive"}
        assert sent_body["output_config"]["effort"] == "max"

    @pytest.mark.asyncio
    async def test_opus_5_max_merges_into_existing_output_config(
        self, opus_5_stable_effort_metadata
    ):
        """Unrelated existing output_config fields survive the effort merge."""
        sent_body = await self._run_native(
            {
                "model": "claude-opus-5",
                "reasoning_effort": "max",
                "output_config": {"verbosity": "high"},
            }
        )
        assert sent_body["output_config"] == {"verbosity": "high", "effort": "max"}

    @pytest.mark.asyncio
    async def test_opus_5_max_does_not_mutate_input_payload(
        self, opus_5_stable_effort_metadata
    ):
        """The input candidate body is never mutated in place."""
        original_output_config = {"verbosity": "high"}
        candidate_body = {
            "model": "claude-opus-5",
            "reasoning_effort": "max",
            "output_config": original_output_config,
        }
        await self._run_native(candidate_body)
        assert candidate_body == {
            "model": "claude-opus-5",
            "reasoning_effort": "max",
            "output_config": {"verbosity": "high"},
        }
        assert original_output_config == {"verbosity": "high"}

    @pytest.mark.asyncio
    async def test_no_raw_effort_leaks_into_a_later_terra_attempt(
        self, opus_5_stable_effort_metadata
    ):
        """A fresh Terra payload built from the original body after an Opus
        attempt carries no transformed Opus state (thinking/output_config)."""
        candidate_body = {
            "model": "claude-opus-5",
            "reasoning_effort": "max",
        }
        await self._run_native(candidate_body)
        terra_payload = dict(candidate_body)
        terra_payload["model"] = "gpt-5.6-terra"
        assert "thinking" not in terra_payload
        assert "output_config" not in terra_payload
        assert terra_payload.get("reasoning_effort") == "max"

    @pytest.mark.asyncio
    async def test_non_anthropic_candidates_unchanged(self):
        """Non-native providers receive candidate_body as prepared, with no
        Anthropic effort shaping."""
        rt = _make_runtime()
        _install_runtime(rt)
        candidate_body = {"model": "gpt-5.6-terra", "reasoning_effort": "max"}
        await _perform_anthropic_auto_agent_alias_candidate_request(
            endpoint="/v1/responses",
            request=MagicMock(),
            fastapi_response=MagicMock(),
            user_api_key_dict=MagicMock(),
            candidate={
                "model": "gpt-5.6-terra",
                "provider": "codex_native",
                "route_family": "",
            },
            candidate_body=candidate_body,
            target_url="https://api.anthropic.com",
            custom_headers={},
        )
        kw = rt.handle_openai_responses.call_args.kwargs
        assert kw["prepared_request_body"] is candidate_body
        assert candidate_body == {"model": "gpt-5.6-terra", "reasoning_effort": "max"}
        assert "thinking" not in candidate_body
        assert "output_config" not in candidate_body

    @pytest.mark.asyncio
    async def test_unsupported_anthropic_max_falls_back_to_high_thinking(
        self, opus_5_stable_effort_metadata
    ):
        """Anthropic models without max-effort support keep the legacy
        behavior: max is clamped to high budget thinking, no output_config."""
        sent_body = await self._run_native(
            {"model": "claude-sonnet-4-5", "reasoning_effort": "max"},
            candidate_model="claude-sonnet-4-5",
        )
        assert "reasoning_effort" not in sent_body
        assert sent_body["thinking"]["type"] == "enabled"
        assert isinstance(sent_body["thinking"].get("budget_tokens"), int)
        assert "output_config" not in sent_body

    @pytest.mark.asyncio
    async def test_no_reasoning_effort_leaves_payload_untouched(
        self, opus_5_stable_effort_metadata
    ):
        """Missing reasoning_effort leaves the payload otherwise untouched."""
        sent_body = await self._run_native({"model": "claude-opus-5"})
        assert "reasoning_effort" not in sent_body
        assert "thinking" not in sent_body
        assert "output_config" not in sent_body
