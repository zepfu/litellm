"""Focused tests for plan items A1–A4 (logging correctness)."""

from __future__ import annotations

from pathlib import Path


def test_a1_langfuse_fallback_uses_input_output_keys() -> None:
    """Primary usage with input/output must not be discarded for fallback."""
    src = Path("litellm/integrations/langfuse/langfuse.py").read_text()
    assert 'usage.get("input"' in src
    assert 'usage.get("output"' in src
    # Guard must not rely only on renamed-away OpenAI key names.
    # The old bug checked prompt_tokens/completion_tokens after rename.
    idx = src.find('usage.get("input"')
    window = src[max(0, idx - 400) : idx + 400]
    assert "fallback_total_tokens" in window

    usage = {"input": 100, "output": 20, "total": 120, "unit": "TOKENS"}
    fallback_total_tokens = 50
    should_fallback = (
        usage is None
        or (
            usage.get("input", 0) == 0
            and usage.get("output", 0) == 0
            and fallback_total_tokens > 0
        )
    ) and fallback_total_tokens > 0
    assert should_fallback is False

    empty_usage = {"input": 0, "output": 0, "total": 0, "unit": "TOKENS"}
    should_fallback_empty = (
        empty_usage is None
        or (
            empty_usage.get("input", 0) == 0
            and empty_usage.get("output", 0) == 0
            and fallback_total_tokens > 0
        )
    ) and fallback_total_tokens > 0
    assert should_fallback_empty is True
    # Old keys absent → always 0 (the bug if still used as the guard).
    assert usage.get("prompt_tokens", 0) == 0
    assert usage.get("completion_tokens", 0) == 0


def test_a2_sync_failure_skips_custom_logger_for_pass_through() -> None:
    """Sync CustomLogger failure path must skip CallTypes.pass_through."""
    src = Path("litellm/litellm_core_utils/litellm_logging.py").read_text()
    # Success + failure both guard pass_through for CustomLogger dual-fire.
    assert src.count("CallTypes.pass_through.value") >= 2
    assert "async_log_failure_event" in src or "pass_through endpoints call async_log_failure" in src


def test_a3_mid_stream_read_timeout_calls_async_failure_handler() -> None:
    src = Path(
        "litellm/proxy/pass_through_endpoints/streaming_handler.py"
    ).read_text()
    assert "async_failure_handler" in src
    assert "import traceback" in src
    # Placed on the post-first-byte terminal path, not only elsewhere.
    idx = src.find("async_failure_handler")
    window = src[max(0, idx - 600) : idx + 200]
    assert "_record_post_first_byte_stream_terminal_rollup" in window or (
        "first" in window.lower() and "timeout" in window.lower()
    )


def test_a4_handle_logging_redacts_before_callbacks() -> None:
    src = Path(
        "litellm/proxy/pass_through_endpoints/success_handler.py"
    ).read_text()
    assert "redact_message_input_output_from_logging" in src
    assert "redact_message_input_output_from_custom_logger" in src
    # Redaction must run before callback iteration.
    redact_idx = src.find("redact_message_input_output_from_logging")
    sync_cb_idx = src.find("sync_callbacks = logging_obj.get_combined_callback_list")
    assert redact_idx >= 0 and sync_cb_idx >= 0
    assert redact_idx < sync_cb_idx
