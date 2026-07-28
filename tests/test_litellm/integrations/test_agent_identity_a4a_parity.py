"""Wave A4A golden-output parity tests.

Pins CURRENT behavior of the four A4A target bands so the engineer's
behavior-preserving extraction can be verified by re-running these tests
post-move.  Every test here is GREEN on pre-engineer develop.

Target modules and their ORIGINAL line bands (per
.analysis/aawm-agent-identity-and-oversized-units-decomposition-2026-07-23.md):

  usage_extract.py      7759-8348, 10094-10267
  provider_normalize.py 8349-8742, 14797-15306
  request_signals.py    8743-9539
  prompt_overhead.py    10268-11019
"""

from __future__ import annotations

from typing import Any

import pytest

from litellm.integrations.aawm_agent_identity import (
    # --- usage_extract band (7759-8348 + 10094-10267) ---
    _extract_prompt_tokens,
    _extract_completion_tokens,
    _extract_total_tokens,
    _extract_prompt_tokens_details,
    _extract_completion_tokens_details,
    _extract_cache_read_input_tokens,
    _extract_cache_creation_input_tokens,
    _coerce_usage_object_to_dict,
    _extract_reported_reasoning_tokens,
    _determine_reasoning_tokens_source,
    _coerce_rerank_text,
    # --- provider_normalize band (8349-8742 + 14797-15306) ---
    _normalize_session_history_provider_name,
    _sanitize_session_history_api_base,
    _is_local_session_history_api_base,
    _is_completion_call_type,
    _is_embedding_call_type,
    _strip_local_provider_model_prefix,
    _session_history_provider_from_api_base,
    # --- request_signals band (8743-9539) ---
    _empty_structured_output_state,
    _invalid_tool_call_error_text_seen,
    _request_payload_contains,
    _classify_compact_summary_state,
    _is_codex_compact_context,
    _is_claude_code_compact_context,
    _is_gemini_cli_compact_context,
    _base_gemini_compact_prompt_id,
    # --- prompt_overhead band (10268-11019) ---
    _fallback_text_token_estimate,
    _empty_prompt_overhead_breakdown,
    _serialize_prompt_overhead_component,
    _positive_int_or_none,
    _usage_has_positive_tokens,
)


# =========================================================================
# usage_extract goldens (original 7759-8348 + 10094-10267)
# =========================================================================


class TestUsageExtractGoldens:
    def test_extract_prompt_tokens_openai(self) -> None:
        assert _extract_prompt_tokens({"prompt_tokens": 1234}) == 1234

    def test_extract_prompt_tokens_anthropic(self) -> None:
        assert _extract_prompt_tokens({"input_tokens": 567}) == 567

    def test_extract_prompt_tokens_fallback_zero(self) -> None:
        assert _extract_prompt_tokens({}) == 0
        assert _extract_prompt_tokens(None) == 0

    def test_extract_completion_tokens_openai(self) -> None:
        assert _extract_completion_tokens({"completion_tokens": 89}) == 89

    def test_extract_completion_tokens_anthropic(self) -> None:
        assert _extract_completion_tokens({"output_tokens": 42}) == 42

    def test_extract_total_tokens_explicit(self) -> None:
        assert _extract_total_tokens({"total_tokens": 999}, 100, 50) == 999

    def test_extract_total_tokens_derived(self) -> None:
        assert _extract_total_tokens({}, 100, 50) == 150

    def test_extract_cache_read_input_tokens_direct(self) -> None:
        assert _extract_cache_read_input_tokens({"cache_read_input_tokens": 300}) == 300

    def test_extract_cache_read_input_tokens_from_details(self) -> None:
        usage = {"prompt_tokens_details": {"cached_tokens": 200}}
        assert _extract_cache_read_input_tokens(usage) == 200

    def test_extract_cache_creation_input_tokens(self) -> None:
        assert _extract_cache_creation_input_tokens({"cache_creation_input_tokens": 150}) == 150

    def test_extract_prompt_tokens_details(self) -> None:
        details = {"cached_tokens": 10}
        assert _extract_prompt_tokens_details({"prompt_tokens_details": details}) is details

    def test_extract_completion_tokens_details(self) -> None:
        details = {"reasoning_tokens": 5}
        assert _extract_completion_tokens_details({"completion_tokens_details": details}) is details

    def test_coerce_usage_object_to_dict(self) -> None:
        assert _coerce_usage_object_to_dict({"prompt_tokens": 1}) == {"prompt_tokens": 1}
        assert _coerce_usage_object_to_dict(None) is None
        assert _coerce_usage_object_to_dict("garbage") is None

    def test_extract_reported_reasoning_tokens(self) -> None:
        assert _extract_reported_reasoning_tokens({"reasoning_tokens": 42}) == 42
        assert _extract_reported_reasoning_tokens({}) is None

    def test_determine_reasoning_tokens_source(self) -> None:
        # Both provider-reported and reported present -> "provider_reported"
        assert _determine_reasoning_tokens_source(
            provider_reported_reasoning_tokens=5,
            reported_reasoning_tokens=10,
            estimated_reasoning_tokens=None,
            reasoning_present=True,
        ) == "provider_reported"
        # No reasoning signal at all -> "not_applicable"
        assert _determine_reasoning_tokens_source(
            provider_reported_reasoning_tokens=None,
            reported_reasoning_tokens=None,
            estimated_reasoning_tokens=None,
            reasoning_present=False,
        ) == "not_applicable"

    def test_coerce_rerank_text(self) -> None:
        assert _coerce_rerank_text("hello") == "hello"
        assert _coerce_rerank_text(None) == ""
        assert _coerce_rerank_text(123) == "123"


# =========================================================================
# provider_normalize goldens (original 8349-8742 + 14797-15306)
# =========================================================================


class TestProviderNormalizeGoldens:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("openai", "openai"),
            ("anthropic", "anthropic"),
            ("google", "gemini"),
            ("google_code_assist", None),
            ("google-code-assist", None),
            ("grok", "xai"),
            ("antigravity", None),
            ("agy", None),
            ("google-antigravity", None),
            ("nvidia", "nvidia_nim"),
            ("opencode", "opencode_zen"),
            ("unknown", None),
            ("", None),
            (None, None),
            ("litellm", None),
            ("local_embed", "local_embed"),
            ("local-embed", "local_embed"),
            ("SomeCustomProvider", "somecustomprovider"),
        ],
    )
    def test_normalize_provider_name(self, raw: Any, expected: Any) -> None:
        assert _normalize_session_history_provider_name(raw) == expected

    def test_sanitize_api_base(self) -> None:
        assert _sanitize_session_history_api_base("https://api.openai.com/v1") == "https://api.openai.com/v1"
        assert _sanitize_session_history_api_base(None) is None
        assert _sanitize_session_history_api_base("") is None

    @pytest.mark.parametrize(
        "url,expected",
        [
            ("http://localhost:4000", True),
            ("http://127.0.0.1:8080", True),
            ("http://192.168.1.1:4000", True),
            ("http://host.docker.internal:4000", True),
            ("https://api.openai.com", False),
            ("https://api.anthropic.com", False),
            (None, False),
        ],
    )
    def test_is_local_api_base(self, url: Any, expected: bool) -> None:
        assert _is_local_session_history_api_base(url) == expected

    def test_is_completion_call_type(self) -> None:
        assert _is_completion_call_type("completion") is True
        assert _is_completion_call_type("embedding") is False

    def test_is_embedding_call_type(self) -> None:
        assert _is_embedding_call_type("embedding", None) is True
        assert _is_embedding_call_type("completion", None) is False

    def test_strip_local_provider_model_prefix(self) -> None:
        assert _strip_local_provider_model_prefix("local_embed/some-model") == "some-model"
        assert _strip_local_provider_model_prefix("gpt-4o") == "gpt-4o"

    def test_session_history_provider_from_api_base(self) -> None:
        assert _session_history_provider_from_api_base("https://api.openai.com/v1") == "openai"
        assert (
            _session_history_provider_from_api_base(
                "https://generativelanguage.googleapis.com/v1beta"
            )
            == "gemini"
        )
        assert (
            _session_history_provider_from_api_base(
                "https://us-central1-aiplatform.googleapis.com/v1"
            )
            == "gemini"
        )
        assert (
            _session_history_provider_from_api_base(
                "https://daily-cloudcode-pa.googleapis.com/v1internal:streamGenerateContent"
            )
            is None
        )


# =========================================================================
# request_signals goldens (original 8743-9539)
# =========================================================================


class TestRequestSignalsGoldens:
    def test_empty_structured_output_state(self) -> None:
        state = _empty_structured_output_state()
        assert isinstance(state, dict)
        assert state["structured_output_attempted"] is False
        assert state["structured_output_failed"] is False

    def test_invalid_tool_call_error_text_seen(self) -> None:
        assert _invalid_tool_call_error_text_seen("invalid tool call detected") is True
        assert _invalid_tool_call_error_text_seen("InputValidationError: bad") is True
        assert _invalid_tool_call_error_text_seen("everything is fine") is False
        assert _invalid_tool_call_error_text_seen(None) is False

    def test_request_payload_contains(self) -> None:
        payload = {"a": {"cache_control": {"type": "ephemeral"}}}
        assert _request_payload_contains(payload, lambda d: "cache_control" in d) is True
        assert _request_payload_contains(payload, lambda d: "missing_key" in d) is False

    def test_base_gemini_compact_prompt_id(self) -> None:
        assert _base_gemini_compact_prompt_id("compress-abc123") == "compress-abc123"
        assert _base_gemini_compact_prompt_id("compress-abc123-verify") == "compress-abc123"

    def test_is_codex_compact_context(self) -> None:
        assert _is_codex_compact_context({"client_name": "codex-tui"}) is True
        assert _is_codex_compact_context({"trace_name": "codex.something"}) is True
        assert _is_codex_compact_context({"client_name": "claude-cli"}) is False

    def test_is_claude_code_compact_context(self) -> None:
        assert _is_claude_code_compact_context({"client_name": "claude-cli"}) is True
        assert _is_claude_code_compact_context({"trace_name": "claude-code.ops"}) is True
        assert _is_claude_code_compact_context({"client_name": "codex-tui"}) is False

    def test_is_gemini_cli_compact_context(self) -> None:
        assert _is_gemini_cli_compact_context({"client_name": "gemini-cli"}) is True
        assert _is_gemini_cli_compact_context({"client_user_agent": "geminicli-tui/1.0"}) is True
        assert _is_gemini_cli_compact_context({"client_name": "codex-tui"}) is False

    def test_classify_compact_summary_state_codex_event(self) -> None:
        result = _classify_compact_summary_state(
            metadata={"client_name": "codex-tui"},
            request_body={"messages": [{"role": "user", "content": "Context checkpoint compaction happened"}]},
            output_payload=None,
            session_id="sess-1",
            litellm_call_id="call-1",
            trace_id="trace-1",
        )
        assert result["is_compact_summary"] is True
        assert result["compact_summary_source"] == "codex"
        assert result["compact_summary_role"] == "event"

    def test_classify_compact_summary_state_not_compact(self) -> None:
        result = _classify_compact_summary_state(
            metadata={"client_name": "unknown"},
            request_body={"messages": [{"role": "user", "content": "hello"}]},
            output_payload=None,
            session_id=None,
            litellm_call_id=None,
            trace_id=None,
        )
        assert result["is_compact_summary"] is False
        assert result["compact_summary_source"] is None


# =========================================================================
# prompt_overhead goldens (original 10268-11019)
# =========================================================================


class TestPromptOverheadGoldens:
    def test_fallback_text_token_estimate(self) -> None:
        assert _fallback_text_token_estimate("") == 0
        assert _fallback_text_token_estimate("   ") == 0
        assert _fallback_text_token_estimate("hello world") == 3  # (11+3)//4

    def test_empty_prompt_overhead_breakdown(self) -> None:
        bd = _empty_prompt_overhead_breakdown()
        assert isinstance(bd, dict)
        assert all(v == 0 for v in bd.values())
        assert "input_system_tokens_estimated" in bd
        assert "system_safety_tokens_estimated" in bd

    def test_serialize_prompt_overhead_component(self) -> None:
        assert _serialize_prompt_overhead_component(None) == ""
        assert _serialize_prompt_overhead_component("text") == "text"
        assert _serialize_prompt_overhead_component(42) == "42"
        result = _serialize_prompt_overhead_component({"b": 2, "a": 1})
        assert result == '{"a":1,"b":2}'

    def test_positive_int_or_none(self) -> None:
        assert _positive_int_or_none(5) == 5
        assert _positive_int_or_none(0) is None
        assert _positive_int_or_none(-1) is None
        assert _positive_int_or_none(None) is None

    def test_usage_has_positive_tokens(self) -> None:
        assert _usage_has_positive_tokens({"prompt_tokens": 10}) is True
        assert _usage_has_positive_tokens({"prompt_tokens": 0}) is False
        assert _usage_has_positive_tokens({}) is False
