from __future__ import annotations

import inspect
from typing import Any

import pytest

from litellm.proxy.pass_through_endpoints.aawm_alias_routing import lane_keys
from litellm.proxy.pass_through_endpoints.aawm_request_policy import (
    alias_guidance as guidance,
)

EXPECTED_CODEX_GUIDANCE = """Codex auto-agent completion contract:
- Always produce a non-empty final answer after completing or stopping the task; do not end a successful request with only reasoning, tool calls, or no visible assistant text.
- Do not return internal planning text as the final answer. Complete the requested work, or state the exact blocker and the next concrete step.
- If a required tool is unavailable or blocked, state the exact observed tool/platform error and continue with bounded evidence from available context; do not claim tools or filesystem are unavailable unless a tool/platform error proves it.
- If the user requested code or artifact changes, either make the scoped change or explicitly say no files were modified and why. Do not answer with a generic explanation of the function or file when implementation or verification was requested.
- If verification could not be run, name the command or check that was not run and why.
- For a coding or file-edit task, a design summary, plan, or statement that edits are about to begin is not a valid final answer. Do not stop until the edit tool has returned and the requested checks have run, or an explicit blocker has been proven.
- Never claim `apply_patch` failed, aborted, or cannot edit a linked `/tmp` worktree unless the client returned an explicit tool error. Absolute paths to writable linked worktrees are supported. If no tool result is visible, retry the tool call instead of switching editing methods or finalizing.
- Preserve the caller's editing contract. Do not replace `apply_patch` with Python, `sed`, or another file-mutation mechanism when the task or repository requires `apply_patch`.
- A successful coding-task final answer must name the changed paths and requested verification results."""

EXPECTED_READ_GUIDANCE = """AAWM read-only agent contract:
- Treat the delegated task as exploration, audit, review, or investigation unless the prompt explicitly authorizes file edits for this worker.
- Do not edit files, create files, apply patches, or run commands that modify the worktree.
- If a fix is needed, describe the patch only. Do not claim the patch was implemented unless the prompt explicitly authorized edits and the files were actually changed.
- If the delegated prompt requires the exact final phrase `No files were modified.`, include that phrase truthfully in the final answer.
- Return findings, evidence, coverage gaps, and recommended next steps. Do not return implementation summaries for read-only work."""


class CallbackRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def build_span(
        self,
        *,
        name: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append(("span", (name, dict(metadata))))
        return {"name": name, "metadata": dict(metadata)}

    def merge_metadata(
        self,
        request_body: dict[str, Any],
        *,
        tags_to_add: list[str],
        extra_fields: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append(
            (
                "merge",
                (request_body, list(tags_to_add), dict(extra_fields)),
            )
        )
        updated_body = dict(request_body)
        existing_metadata = request_body.get("litellm_metadata")
        merged_metadata = (
            dict(existing_metadata) if isinstance(existing_metadata, dict) else {}
        )
        existing_tags = merged_metadata.get("tags")
        merged_metadata["tags"] = [
            *(existing_tags if isinstance(existing_tags, list) else []),
            *tags_to_add,
        ]
        merged_metadata.update(extra_fields)
        updated_body["litellm_metadata"] = merged_metadata
        return updated_body

    @property
    def callbacks(self) -> guidance.AliasGuidanceCallbacks:
        return guidance.AliasGuidanceCallbacks(
            merge_litellm_metadata=self.merge_metadata,
            build_langfuse_span_descriptor=self.build_span,
        )


@pytest.fixture(autouse=True)
def _reset_runtime_callbacks():
    """Ensure runtime callbacks are reset before and after each test."""
    guidance.configure_alias_guidance_runtime(callbacks=None)
    yield
    guidance.configure_alias_guidance_runtime(callbacks=None)


# ── Constant identity tests ────────────────────────────────────────


def test_should_source_codex_constants_from_lane_keys_same_object() -> None:
    config = guidance.DEFAULT_ALIAS_GUIDANCE_CONFIG
    assert config.codex_auto_agent_prevention_guidance_policy_name is (
        lane_keys._CODEX_AUTO_AGENT_PREVENTION_GUIDANCE_POLICY_NAME
    )
    assert config.codex_auto_agent_prevention_guidance_policy_version is (
        lane_keys._CODEX_AUTO_AGENT_PREVENTION_GUIDANCE_POLICY_VERSION
    )
    assert config.codex_auto_agent_prevention_guidance_prompt is (
        lane_keys._CODEX_AUTO_AGENT_PREVENTION_GUIDANCE_PROMPT
    )


def test_should_source_read_constants_from_lane_keys_same_object() -> None:
    config = guidance.DEFAULT_ALIAS_GUIDANCE_CONFIG
    assert config.aawm_read_agent_guidance_policy_name is (
        lane_keys._AAWM_READ_AGENT_GUIDANCE_POLICY_NAME
    )
    assert config.aawm_read_agent_guidance_policy_version is (
        lane_keys._AAWM_READ_AGENT_GUIDANCE_POLICY_VERSION
    )
    assert config.aawm_read_agent_guidance_prompt is (
        lane_keys._AAWM_READ_AGENT_GUIDANCE_PROMPT
    )


# ── Signature / call-site parity tests ─────────────────────────────


def test_should_accept_god_call_site_signature_for_read_guidance() -> None:
    """God module calls: _apply_aawm_read_agent_guidance_to_request_body(body, alias_model=..., target_field=...)"""
    body = {"instructions": "existing"}
    result = guidance._apply_aawm_read_agent_guidance_to_request_body(
        body,
        alias_model="aawm-read",
        target_field="instructions",
    )
    assert isinstance(result, tuple)
    assert len(result) == 2
    updated_body, metadata = result
    assert updated_body["instructions"] == f"existing\n\n{EXPECTED_READ_GUIDANCE}"
    assert metadata["aawm_read_agent_guidance_applied"] is True


def test_should_accept_god_call_site_signature_for_codex_guidance() -> None:
    """God module calls: _apply_codex_auto_agent_prevention_guidance_to_request_body(body)"""
    body = {"instructions": "existing"}
    result = guidance._apply_codex_auto_agent_prevention_guidance_to_request_body(body)
    assert isinstance(result, tuple)
    assert len(result) == 2
    updated_body, metadata = result
    assert updated_body["instructions"] == f"existing\n\n{EXPECTED_CODEX_GUIDANCE}"
    assert metadata["codex_auto_agent_prevention_guidance_applied"] is True


def test_should_have_callbacks_optional_in_both_signatures() -> None:
    read_sig = inspect.signature(
        guidance._apply_aawm_read_agent_guidance_to_request_body
    )
    codex_sig = inspect.signature(
        guidance._apply_codex_auto_agent_prevention_guidance_to_request_body
    )
    assert read_sig.parameters["callbacks"].default is None
    assert codex_sig.parameters["callbacks"].default is None


def test_should_accept_explicit_callbacks_kwarg() -> None:
    recorder = CallbackRecorder()
    body = {"instructions": "existing"}
    updated_body, metadata = (
        guidance._apply_codex_auto_agent_prevention_guidance_to_request_body(
            body,
            callbacks=recorder.callbacks,
        )
    )
    assert metadata["codex_auto_agent_prevention_guidance_applied"] is True
    assert [name for name, _ in recorder.calls] == ["span", "merge"]


# ── Configured runtime seam tests ──────────────────────────────────


def test_should_use_configured_runtime_callbacks_when_no_explicit() -> None:
    recorder = CallbackRecorder()
    guidance.configure_alias_guidance_runtime(callbacks=recorder.callbacks)

    body = {"instructions": "existing"}
    updated_body, metadata = (
        guidance._apply_codex_auto_agent_prevention_guidance_to_request_body(body)
    )
    assert metadata["codex_auto_agent_prevention_guidance_applied"] is True
    assert [name for name, _ in recorder.calls] == ["span", "merge"]


def test_should_prefer_explicit_callbacks_over_configured_runtime() -> None:
    runtime_recorder = CallbackRecorder()
    explicit_recorder = CallbackRecorder()
    guidance.configure_alias_guidance_runtime(callbacks=runtime_recorder.callbacks)

    body = {"instructions": "existing"}
    guidance._apply_codex_auto_agent_prevention_guidance_to_request_body(
        body,
        callbacks=explicit_recorder.callbacks,
    )
    assert runtime_recorder.calls == []
    assert [name for name, _ in explicit_recorder.calls] == ["span", "merge"]


def test_should_restore_defaults_when_runtime_reset_to_none() -> None:
    recorder = CallbackRecorder()
    guidance.configure_alias_guidance_runtime(callbacks=recorder.callbacks)
    guidance.configure_alias_guidance_runtime(callbacks=None)

    body = {"instructions": "existing"}
    updated_body, metadata = (
        guidance._apply_codex_auto_agent_prevention_guidance_to_request_body(body)
    )
    assert metadata["codex_auto_agent_prevention_guidance_applied"] is True
    assert recorder.calls == []
    # Default callbacks produce litellm_metadata with tags
    assert "litellm_metadata" in updated_body
    tags = updated_body["litellm_metadata"]["tags"]
    assert "codex-auto-agent-prevention-guidance" in tags


def test_should_use_default_callbacks_without_any_configuration() -> None:
    """Without configure_alias_guidance_runtime, default observability_metadata callbacks apply."""
    body = {"instructions": "existing"}
    updated_body, metadata = (
        guidance._apply_aawm_read_agent_guidance_to_request_body(
            body,
            alias_model="aawm-read",
            target_field="instructions",
        )
    )
    assert metadata["aawm_read_agent_guidance_applied"] is True
    assert "litellm_metadata" in updated_body
    tags = updated_body["litellm_metadata"]["tags"]
    assert "aawm-read-agent-guidance" in tags
    assert "aawm-read-agent-guidance:2026-06-06.v1" in tags
    assert "aawm-read-agent-guidance-alias:aawm-read" in tags
    spans = updated_body["litellm_metadata"]["langfuse_spans"]
    assert len(spans) == 1
    assert spans[0]["name"] == "aawm.read_agent_guidance"


# ── Default configuration pin tests ────────────────────────────────


def test_should_pin_exact_default_guidance_configuration() -> None:
    config = guidance.DEFAULT_ALIAS_GUIDANCE_CONFIG

    assert config.codex_auto_agent_prevention_guidance_policy_name == (
        "codex_auto_agent_prevention_guidance"
    )
    assert config.codex_auto_agent_prevention_guidance_policy_version == (
        "2026-07-21.v2"
    )
    assert config.codex_auto_agent_prevention_guidance_prompt == EXPECTED_CODEX_GUIDANCE
    assert config.aawm_read_agent_guidance_policy_name == "aawm_read_agent_guidance"
    assert config.aawm_read_agent_guidance_policy_version == "2026-06-06.v1"
    assert config.aawm_read_agent_guidance_prompt == EXPECTED_READ_GUIDANCE
    assert config.codex_aawm_read_alias == "aawm-read"
    assert config.anthropic_aawm_read_alias == "aawm-read-anthropic"


# ── Append helper tests ────────────────────────────────────────────


@pytest.mark.parametrize("value", [None, "", "   "])
def test_should_use_codex_guidance_for_empty_instructions(value: str | None) -> None:
    assert (
        guidance._append_codex_auto_agent_prevention_guidance_to_instructions(value)
        == EXPECTED_CODEX_GUIDANCE
    )


def test_should_append_codex_guidance_after_trimmed_existing_instructions() -> None:
    assert guidance._append_codex_auto_agent_prevention_guidance_to_instructions(
        "  existing instructions  "
    ) == f"existing instructions\n\n{EXPECTED_CODEX_GUIDANCE}"


def test_should_make_codex_guidance_idempotent_and_preserve_original_order() -> None:
    value = f"  prefix\n\n{EXPECTED_CODEX_GUIDANCE}  "

    first = guidance._append_codex_auto_agent_prevention_guidance_to_instructions(
        value
    )
    second = guidance._append_codex_auto_agent_prevention_guidance_to_instructions(
        first
    )

    assert first == f"prefix\n\n{EXPECTED_CODEX_GUIDANCE}"
    assert second == first
    assert first.count(EXPECTED_CODEX_GUIDANCE) == 1


@pytest.mark.parametrize(
    ("alias_model", "expected"),
    [
        ("aawm-read", True),
        ("aawm-read-anthropic", True),
        ("AAWM-READ", False),
        ("aawm-read ", False),
        ("read", False),
        (None, False),
        (1, False),
    ],
)
def test_should_match_only_exact_read_alias_strings(
    alias_model: Any,
    expected: bool,
) -> None:
    assert guidance._is_aawm_read_agent_alias_model(alias_model) is expected


@pytest.mark.parametrize("value", [None, "", "   "])
def test_should_use_read_guidance_for_empty_text(value: str | None) -> None:
    assert (
        guidance._append_aawm_read_agent_guidance_to_text(value)
        == EXPECTED_READ_GUIDANCE
    )


def test_should_append_read_guidance_after_trimmed_text_once() -> None:
    first = guidance._append_aawm_read_agent_guidance_to_text("  existing  ")
    second = guidance._append_aawm_read_agent_guidance_to_text(first)

    assert first == f"existing\n\n{EXPECTED_READ_GUIDANCE}"
    assert second == first
    assert first.count(EXPECTED_READ_GUIDANCE) == 1


@pytest.mark.parametrize(
    ("system_value", "expected_value", "expected_changed", "expected_chars"),
    [
        (None, EXPECTED_READ_GUIDANCE, True, 0),
        ("", EXPECTED_READ_GUIDANCE, True, 0),
        (
            "  existing  ",
            f"existing\n\n{EXPECTED_READ_GUIDANCE}",
            True,
            len("  existing  "),
        ),
        ({"type": "text"}, {"type": "text"}, False, 0),
        (42, 42, False, 0),
    ],
)
def test_should_shape_scalar_anthropic_system_values_exactly(
    system_value: Any,
    expected_value: Any,
    expected_changed: bool,
    expected_chars: int,
) -> None:
    updated, changed, original_chars = (
        guidance._append_aawm_read_agent_guidance_to_anthropic_system(system_value)
    )

    assert updated == expected_value
    assert changed is expected_changed
    assert original_chars == expected_chars


def test_should_append_read_guidance_after_all_anthropic_system_blocks() -> None:
    system_value = [
        "alpha",
        {"type": "text", "text": "beta"},
        {"type": "image", "source": "ignored"},
        7,
    ]

    updated, changed, original_chars = (
        guidance._append_aawm_read_agent_guidance_to_anthropic_system(system_value)
    )

    assert updated == [
        *system_value,
        {"type": "text", "text": EXPECTED_READ_GUIDANCE},
    ]
    assert changed is True
    assert original_chars == len("alpha") + len("beta")
    assert updated is not system_value


def test_should_leave_anthropic_system_list_unchanged_when_guidance_exists() -> None:
    system_value = [
        {"type": "text", "text": "alpha"},
        f"prefix {EXPECTED_READ_GUIDANCE} suffix",
        {"type": "text", "text": "not counted after duplicate"},
    ]

    updated, changed, original_chars = (
        guidance._append_aawm_read_agent_guidance_to_anthropic_system(system_value)
    )

    assert updated is system_value
    assert changed is False
    assert original_chars == len("alpha") + len(system_value[1])


# ── Apply with explicit callbacks (existing behavior) ──────────────


def test_should_noop_read_guidance_for_non_alias_invalid_field_and_malformed_values() -> (
    None
):
    recorder = CallbackRecorder()
    body = {"instructions": ["malformed"], "system": {"malformed": True}}

    non_alias_result = guidance._apply_aawm_read_agent_guidance_to_request_body(
        body,
        alias_model="other",
        target_field="instructions",
        callbacks=recorder.callbacks,
    )
    invalid_field_result = guidance._apply_aawm_read_agent_guidance_to_request_body(
        body,
        alias_model="aawm-read",
        target_field="other",
        callbacks=recorder.callbacks,
    )
    malformed_instructions_result = (
        guidance._apply_aawm_read_agent_guidance_to_request_body(
            body,
            alias_model="aawm-read",
            target_field="instructions",
            callbacks=recorder.callbacks,
        )
    )
    malformed_system_result = guidance._apply_aawm_read_agent_guidance_to_request_body(
        body,
        alias_model="aawm-read-anthropic",
        target_field="system",
        callbacks=recorder.callbacks,
    )

    for returned_body, metadata in (
        non_alias_result,
        invalid_field_result,
        malformed_instructions_result,
        malformed_system_result,
    ):
        assert returned_body is body
        assert metadata == {}
    assert recorder.calls == []


@pytest.mark.parametrize(
    ("alias_model", "target_field", "field_name", "original_value"),
    [
        ("aawm-read", "instructions", "instructions", "  existing  "),
        (
            "aawm-read-anthropic",
            "system",
            "system",
            ["alpha", {"type": "text", "text": "beta"}],
        ),
    ],
)
def test_should_apply_read_guidance_with_exact_metadata_and_callback_order(
    alias_model: str,
    target_field: str,
    field_name: str,
    original_value: Any,
) -> None:
    recorder = CallbackRecorder()
    body = {field_name: original_value, "untouched": True}

    updated_body, metadata = (
        guidance._apply_aawm_read_agent_guidance_to_request_body(
            body,
            alias_model=alias_model,
            target_field=target_field,
            callbacks=recorder.callbacks,
        )
    )

    assert updated_body is not body
    assert body == {field_name: original_value, "untouched": True}
    if target_field == "instructions":
        assert updated_body[field_name] == f"existing\n\n{EXPECTED_READ_GUIDANCE}"
        expected_original_chars = len("  existing  ")
    else:
        assert updated_body[field_name] == [
            *original_value,
            {"type": "text", "text": EXPECTED_READ_GUIDANCE},
        ]
        expected_original_chars = len("alpha") + len("beta")
    assert metadata == {
        "aawm_read_agent_guidance_policy_name": "aawm_read_agent_guidance",
        "aawm_read_agent_guidance_policy_version": "2026-06-06.v1",
        "aawm_read_agent_guidance_applied": True,
        "aawm_read_agent_guidance_alias": alias_model,
        "aawm_read_agent_guidance_target_field": target_field,
        "aawm_read_agent_guidance_original_chars": expected_original_chars,
        "aawm_read_agent_guidance_prompt_chars": len(EXPECTED_READ_GUIDANCE),
    }
    assert [call_name for call_name, _ in recorder.calls] == ["span", "merge"]
    merge_body, tags, extra_fields = recorder.calls[1][1]
    assert merge_body[field_name] == updated_body[field_name]
    assert tags == [
        "aawm-read-agent-guidance",
        "aawm-read-agent-guidance:2026-06-06.v1",
        f"aawm-read-agent-guidance-alias:{alias_model}",
    ]
    assert extra_fields == {
        **metadata,
        "langfuse_spans": [
            {
                "name": "aawm.read_agent_guidance",
                "metadata": metadata,
            }
        ],
    }


def test_should_make_applied_read_guidance_idempotent() -> None:
    recorder = CallbackRecorder()
    body = {"instructions": "existing"}

    first_body, first_metadata = (
        guidance._apply_aawm_read_agent_guidance_to_request_body(
            body,
            alias_model="aawm-read",
            target_field="instructions",
            callbacks=recorder.callbacks,
        )
    )
    second_body, second_metadata = (
        guidance._apply_aawm_read_agent_guidance_to_request_body(
            first_body,
            alias_model="aawm-read",
            target_field="instructions",
            callbacks=recorder.callbacks,
        )
    )

    assert first_metadata
    assert second_body is first_body
    assert second_metadata == {}
    assert first_body["instructions"].count(EXPECTED_READ_GUIDANCE) == 1
    assert [call_name for call_name, _ in recorder.calls] == ["span", "merge"]


def test_should_noop_codex_guidance_for_malformed_or_existing_instructions() -> None:
    recorder = CallbackRecorder()
    malformed_body = {"instructions": ["malformed"]}
    existing_body = {"instructions": EXPECTED_CODEX_GUIDANCE}

    malformed_result = (
        guidance._apply_codex_auto_agent_prevention_guidance_to_request_body(
            malformed_body,
            callbacks=recorder.callbacks,
        )
    )
    existing_result = (
        guidance._apply_codex_auto_agent_prevention_guidance_to_request_body(
            existing_body,
            callbacks=recorder.callbacks,
        )
    )

    assert malformed_result == (malformed_body, {})
    assert malformed_result[0] is malformed_body
    assert existing_result == (existing_body, {})
    assert existing_result[0] is existing_body
    assert recorder.calls == []


def test_should_apply_codex_guidance_with_exact_metadata_and_callback_order() -> None:
    recorder = CallbackRecorder()
    body = {"instructions": "  existing  ", "untouched": True}

    updated_body, metadata = (
        guidance._apply_codex_auto_agent_prevention_guidance_to_request_body(
            body,
            callbacks=recorder.callbacks,
        )
    )

    assert updated_body["instructions"] == f"existing\n\n{EXPECTED_CODEX_GUIDANCE}"
    assert body == {"instructions": "  existing  ", "untouched": True}
    assert metadata == {
        "codex_auto_agent_prevention_guidance_policy_name": (
            "codex_auto_agent_prevention_guidance"
        ),
        "codex_auto_agent_prevention_guidance_policy_version": "2026-07-21.v2",
        "codex_auto_agent_prevention_guidance_applied": True,
        "codex_auto_agent_prevention_guidance_original_instruction_chars": len(
            "  existing  "
        ),
        "codex_auto_agent_prevention_guidance_prompt_chars": len(
            EXPECTED_CODEX_GUIDANCE
        ),
    }
    assert [call_name for call_name, _ in recorder.calls] == ["span", "merge"]
    merge_body, tags, extra_fields = recorder.calls[1][1]
    assert merge_body["instructions"] == updated_body["instructions"]
    assert tags == [
        "codex-auto-agent-prevention-guidance",
        "codex-auto-agent-prevention-guidance:2026-07-21.v2",
    ]
    assert extra_fields == {
        **metadata,
        "langfuse_spans": [
            {
                "name": "codex.auto_agent_prevention_guidance",
                "metadata": metadata,
            }
        ],
    }


def test_should_make_applied_codex_guidance_idempotent() -> None:
    recorder = CallbackRecorder()
    body = {"instructions": "existing"}

    first_body, first_metadata = (
        guidance._apply_codex_auto_agent_prevention_guidance_to_request_body(
            body,
            callbacks=recorder.callbacks,
        )
    )
    second_body, second_metadata = (
        guidance._apply_codex_auto_agent_prevention_guidance_to_request_body(
            first_body,
            callbacks=recorder.callbacks,
        )
    )

    assert first_metadata
    assert second_body is first_body
    assert second_metadata == {}
    assert first_body["instructions"].count(EXPECTED_CODEX_GUIDANCE) == 1
    assert [call_name for call_name, _ in recorder.calls] == ["span", "merge"]


# ── Exception behavior tests ───────────────────────────────────────


def test_should_preserve_malformed_request_body_exception_behavior() -> None:
    recorder = CallbackRecorder()

    non_alias_body, metadata = (
        guidance._apply_aawm_read_agent_guidance_to_request_body(
            None,  # type: ignore[arg-type]
            alias_model="other",
            target_field="instructions",
            callbacks=recorder.callbacks,
        )
    )
    assert non_alias_body is None
    assert metadata == {}

    with pytest.raises(TypeError):
        guidance._apply_aawm_read_agent_guidance_to_request_body(
            None,  # type: ignore[arg-type]
            alias_model="aawm-read",
            target_field="instructions",
            callbacks=recorder.callbacks,
        )

    with pytest.raises(AttributeError):
        guidance._apply_codex_auto_agent_prevention_guidance_to_request_body(
            None,  # type: ignore[arg-type]
            callbacks=recorder.callbacks,
        )


def test_should_propagate_span_and_merge_exceptions_in_original_order() -> None:
    calls: list[str] = []
    span_error = RuntimeError("span failed")

    def raise_span(
        *,
        name: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        calls.append("span")
        raise span_error

    def unexpected_merge(
        request_body: dict[str, Any],
        *,
        tags_to_add: list[str],
        extra_fields: dict[str, Any],
    ) -> dict[str, Any]:
        calls.append("merge")
        return request_body

    callbacks = guidance.AliasGuidanceCallbacks(
        merge_litellm_metadata=unexpected_merge,
        build_langfuse_span_descriptor=raise_span,
    )
    with pytest.raises(RuntimeError) as span_exc:
        guidance._apply_codex_auto_agent_prevention_guidance_to_request_body(
            {},
            callbacks=callbacks,
        )
    assert span_exc.value is span_error
    assert calls == ["span"]

    calls.clear()
    merge_error = LookupError("merge failed")

    def build_span(
        *,
        name: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        calls.append("span")
        return {"name": name, "metadata": metadata}

    def raise_merge(
        request_body: dict[str, Any],
        *,
        tags_to_add: list[str],
        extra_fields: dict[str, Any],
    ) -> dict[str, Any]:
        calls.append("merge")
        raise merge_error

    callbacks = guidance.AliasGuidanceCallbacks(
        merge_litellm_metadata=raise_merge,
        build_langfuse_span_descriptor=build_span,
    )
    with pytest.raises(LookupError) as merge_exc:
        guidance._apply_aawm_read_agent_guidance_to_request_body(
            {},
            alias_model="aawm-read",
            target_field="instructions",
            callbacks=callbacks,
        )
    assert merge_exc.value is merge_error
    assert calls == ["span", "merge"]
