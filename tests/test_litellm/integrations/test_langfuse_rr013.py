"""RR-013: Langfuse usage-key fallback + avoid duplicate full-payload size serialization."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from litellm.integrations.langfuse import langfuse as langfuse_mod


def _small_generation_params() -> Dict[str, Any]:
    return {
        "id": "generation-rr013-small",
        "name": "litellm-completion",
        "model": "gpt-test",
        "input": "hello",
        "output": "world",
        "metadata": {"repository": "litellm"},
    }


def test_rr013_fit_reuses_measured_size_without_reserializing_under_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under-target payloads must not call _json_size_bytes when size is pre-measured."""
    generation_params = _small_generation_params()
    measured = 42
    call_count = {"n": 0}
    real_json_size = langfuse_mod._json_size_bytes

    def counting_json_size(value: Any) -> int:
        call_count["n"] += 1
        return real_json_size(value)

    monkeypatch.setattr(langfuse_mod, "_json_size_bytes", counting_json_size)

    fitted, fit_summary = langfuse_mod._fit_langfuse_generation_params_to_event_size(
        generation_params,
        max_event_size_bytes=10_000,
        measured_size_bytes=measured,
    )

    assert fitted is generation_params
    assert fit_summary is None
    assert call_count["n"] == 0


def test_rr013_payload_size_summary_reuses_measured_total_without_full_serialize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit threshold gate must accept a pre-measured total and skip full dumps."""
    generation_params = _small_generation_params()
    call_args: List[Any] = []
    real_json_size = langfuse_mod._json_size_bytes

    def counting_json_size(value: Any) -> int:
        call_args.append(value)
        return real_json_size(value)

    monkeypatch.setattr(langfuse_mod, "_json_size_bytes", counting_json_size)

    # Below 0.9 * max: early return. measured total prevents serializing generation_params.
    summary = langfuse_mod._build_langfuse_payload_size_summary(
        generation_params,
        trace_id="trace-rr013",
        call_type="completion",
        max_event_size_bytes=10_000,
        measured_total_size_bytes=100,
    )
    assert summary is None
    assert generation_params not in call_args


def test_rr013_payload_size_summary_prefers_fit_summary_final_total(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generation_params = {
        "id": "generation-rr013-fit-total",
        "name": "aawm.large",
        "model": "gpt-test",
        "input": "x" * 50,
        "output": "y" * 50,
        "metadata": {"k": "v"},
    }
    call_args: List[Any] = []
    real_json_size = langfuse_mod._json_size_bytes

    def counting_json_size(value: Any) -> int:
        call_args.append(value)
        return real_json_size(value)

    monkeypatch.setattr(langfuse_mod, "_json_size_bytes", counting_json_size)

    fit_summary = {
        "event_fit_target_bytes": 100,
        "field_truncations": [],
        "truncated_fields": [],
        "omitted_fields": [],
        "final_total_size_bytes": 250,
        "event_fit_failed": False,
    }
    # With fit_summary present, summary is always built (even if under audit threshold).
    # total_size_bytes must come from fit_summary, not a new full-payload dump.
    summary = langfuse_mod._build_langfuse_payload_size_summary(
        generation_params,
        trace_id="trace-rr013-fit",
        call_type="completion",
        max_event_size_bytes=10_000,
        input_truncation_summary=fit_summary,
    )
    assert summary is not None
    assert summary["total_size_bytes"] == 250
    assert generation_params not in call_args


def test_rr013_fit_and_audit_single_full_payload_measure_for_small_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hot path: fit + audit share one full-payload serialization for small events."""
    generation_params = _small_generation_params()
    full_payload_measures = {"n": 0}
    real_json_size = langfuse_mod._json_size_bytes

    def counting_json_size(value: Any) -> int:
        # Count only full event-dict serializations (the expensive duplicate cost).
        if isinstance(value, dict) and "input" in value and "output" in value:
            full_payload_measures["n"] += 1
        return real_json_size(value)

    monkeypatch.setattr(langfuse_mod, "_json_size_bytes", counting_json_size)
    monkeypatch.setattr(
        langfuse_mod, "record_langfuse_enqueue_size_audit", lambda *_a, **_k: None
    )

    fitted, fit_summary = langfuse_mod._fit_and_audit_langfuse_event_params(
        generation_params,
        trace_id="trace-rr013-hot",
        call_type="completion",
        max_event_size_bytes=10_000,
    )

    assert fitted is generation_params
    assert fit_summary is None
    # One shared measure for fit early-return + audit threshold gate.
    assert full_payload_measures["n"] == 1


def test_rr013_fit_and_audit_reuses_fit_summary_size_when_truncation_occurs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When fitting runs, audit total must reuse fit summary final_total_size_bytes."""
    # Force a tiny limit so fitting is required.
    generation_params = {
        "id": "generation-rr013-over",
        "name": "aawm.large",
        "model": "gpt-test",
        "input": "secret prompt " * 200,
        "output": "response text " * 50,
        "metadata": {"blob": "z" * 200},
    }
    # Spy on summary builder to ensure measured_total_size_bytes is passed.
    captured: Dict[str, Any] = {}
    original_build = langfuse_mod._build_langfuse_payload_size_summary

    def wrapping_build(*args: Any, **kwargs: Any):
        captured["kwargs"] = dict(kwargs)
        if args:
            captured["generation_params"] = args[0]
        return original_build(*args, **kwargs)

    monkeypatch.setattr(langfuse_mod, "_build_langfuse_payload_size_summary", wrapping_build)
    monkeypatch.setattr(
        langfuse_mod, "record_langfuse_enqueue_size_audit", lambda *_a, **_k: None
    )

    fitted, fit_summary = langfuse_mod._fit_and_audit_langfuse_event_params(
        generation_params,
        trace_id="trace-rr013-over",
        call_type="completion",
        max_event_size_bytes=400,
    )

    assert fit_summary is not None
    assert isinstance(fit_summary.get("final_total_size_bytes"), int)
    assert captured["kwargs"].get("measured_total_size_bytes") == fit_summary[
        "final_total_size_bytes"
    ]
    assert fitted is not generation_params or fit_summary.get("truncated_fields") is not None


def test_rr013_primary_usage_input_output_keys_not_discarded_by_fallback() -> None:
    """Issue #1 guard: primary usage with input/output must not be overwritten.

    Replicates the production guard condition against Langfuse-preferred keys.
    """
    # Successful primary extraction shape (post-rename).
    usage: Dict[str, Any] = {
        "input": 80,  # prompt_tokens - cache_read
        "output": 20,
        "total": 120,
        "unit": "TOKENS",
        "total_cost": 0.01,
    }
    fallback_total_tokens = 120

    should_fallback = (
        usage is None
        or (
            usage.get("input", 0) == 0
            and usage.get("output", 0) == 0
            and fallback_total_tokens > 0
        )
    ) and fallback_total_tokens > 0

    assert should_fallback is False

    # Zero primary tokens with positive fallback still allows fallback.
    empty_primary = {"input": 0, "output": 0, "total": 0, "unit": "TOKENS"}
    should_fallback_empty = (
        empty_primary is None
        or (
            empty_primary.get("input", 0) == 0
            and empty_primary.get("output", 0) == 0
            and fallback_total_tokens > 0
        )
    ) and fallback_total_tokens > 0
    assert should_fallback_empty is True

    # Old buggy guard on prompt_tokens/completion_tokens would incorrectly fire.
    buggy_should_fallback = (
        usage is None
        or (
            usage.get("prompt_tokens", 0) == 0
            and usage.get("completion_tokens", 0) == 0
            and fallback_total_tokens > 0
        )
    ) and fallback_total_tokens > 0
    assert buggy_should_fallback is True  # documents the pre-fix bug shape


def test_rr013_usage_fallback_guard_present_in_source() -> None:
    """Ensure production source keeps the input/output fallback guard (not prompt_tokens)."""
    source_path = langfuse_mod.__file__
    assert source_path is not None
    source = open(source_path, encoding="utf-8").read()
    assert 'usage.get("input", 0) == 0' in source
    assert 'usage.get("output", 0) == 0' in source
    # The old keys must not remain as the fallback guard condition on usage.
    # (standard_logging_object may still use prompt_tokens/completion_tokens.)
    assert (
        'usage.get("prompt_tokens", 0) == 0'
        not in source
    )
    assert (
        'usage.get("completion_tokens", 0) == 0'
        not in source
    )


def test_rr013_direct_log_helper_accepts_measured_total(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generation_params = _small_generation_params()
    seen: Dict[str, Any] = {}

    def fake_build(**kwargs: Any) -> None:
        seen.update(kwargs)
        return None

    monkeypatch.setattr(langfuse_mod, "_build_langfuse_payload_size_summary", fake_build)

    langfuse_mod._log_langfuse_payload_size_if_needed(
        generation_params,
        trace_id="t",
        call_type="completion",
        measured_total_size_bytes=123,
        max_event_size_bytes=999,
    )
    assert seen["measured_total_size_bytes"] == 123
    assert seen["max_event_size_bytes"] == 999
