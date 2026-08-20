"""CFG-025: repetitive visible-text guard on OpenAI passthrough Responses.

Provider-neutral, config-driven detector. Initial selector is resolved xAI
identity on POST /openai_passthrough/v1/responses (and /openai_passthrough/responses).
Do not hardcode model names. Do not hook direct /grok/v1/responses.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime import (
    repetitive_output as ro,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    output_guard_config as ogc,
)
from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
    grok_proxy_route,
)


LOOP_CLAUSE = (
    "the warehouse layout still needs another careful pass later today"
)
assert len(LOOP_CLAUSE.split()) == 10

REPETITIVE_OUTPUT_CODE = "aawm_repetitive_output_loop"
FAILURE_KIND = "repetitive_output_loop"

REPO_ROOT = Path(__file__).resolve().parents[4]
DETECTOR_PATH = (
    REPO_ROOT
    / "litellm/proxy/pass_through_endpoints/aawm_adapter_runtime/repetitive_output.py"
)
CONFIG_MODULE_PATH = (
    REPO_ROOT
    / "litellm/proxy/pass_through_endpoints/aawm_alias_routing/output_guard_config.py"
)
DEFAULT_YAML_PATH = (
    REPO_ROOT
    / "litellm/proxy/pass_through_endpoints/aawm_alias_routing/output_guards.yaml"
)
GROK_ROUTE_MODULE_PATH = (
    REPO_ROOT
    / "litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py"
)
PASS_THROUGH_REQUEST_PATH = (
    REPO_ROOT
    / "litellm/proxy/pass_through_endpoints/pass_through_endpoints.py"
)
CODEX_CANDIDATE_CALLS_PATH = (
    REPO_ROOT
    / "litellm/proxy/pass_through_endpoints/aawm_adapter_runtime/codex_candidate_calls.py"
)
ALIAS_STREAMING_PATH = (
    REPO_ROOT
    / "litellm/proxy/pass_through_endpoints/aawm_alias_routing/streaming.py"
)


def _looping_text(*, repeats: int = 12, corrupt: bool = False) -> str:
    parts: list[str] = []
    for index in range(repeats):
        clause = LOOP_CLAUSE
        if corrupt:
            if index % 3 == 1:
                clause = (
                    "the the warehouse layout still needs another careful pass later today"
                )
            elif index % 3 == 2:
                clause = (
                    "the warehouse layout still need another careful pass later today"
                )
        parts.append(clause)
    return " ".join(parts)


def _default_policy(**overrides: Any) -> ogc.OutputGuardPolicy:
    payload = {
        "name": "xai_visible_text_v1",
        "inspect": "visible_output_text",
        "window_words": 400,
        "min_words": 80,
        "min_ngram": 8,
        "min_repeats": 6,
        "min_coverage": 0.72,
        "growth_words": 120,
        "max_novelty": 0.18,
        "max_ngram_edit_distance": 2,
    }
    payload.update(overrides)
    return ogc.OutputGuardPolicy(**payload)


def _xai_passthrough_context(**overrides: Any) -> ogc.OutputGuardRequestContext:
    payload = {
        "ingress_path": "/openai_passthrough/v1/responses",
        "method": "POST",
        "custom_llm_provider": "xai",
        "egress_credential_family": "xai",
        "route_family": "codex_xai_oauth_responses_adapter",
        "resolved_model": "grok-native-placeholder",
    }
    payload.update(overrides)
    return ogc.OutputGuardRequestContext(**payload)


def _sse_event(event_type: str, payload: dict[str, Any]) -> bytes:
    body = {"type": event_type, **payload}
    return (
        f"event: {event_type}\ndata: "
        + json.dumps(body, separators=(",", ":"))
        + "\n\n"
    ).encode("utf-8")


def _output_text_delta(text: str, *, item_id: str = "msg_1") -> bytes:
    return _sse_event(
        "response.output_text.delta",
        {"item_id": item_id, "output_index": 0, "content_index": 0, "delta": text},
    )


def _decode_sse_events(chunks: list[bytes]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    blob = b"".join(chunks).decode("utf-8")
    for block in blob.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        data_lines = [
            line[len("data:") :].strip()
            for line in block.splitlines()
            if line.startswith("data:")
        ]
        if not data_lines:
            continue
        data_text = "\n".join(data_lines)
        if data_text == "[DONE]":
            events.append({"type": "[DONE]"})
            continue
        events.append(json.loads(data_text))
    return events


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


def test_exact_repetition_past_thresholds_matches() -> None:
    detector = ro.VisibleTextRepetitionDetector(_default_policy())
    match = detector.feed(_looping_text(repeats=12, corrupt=False))
    assert match is not None
    assert match.policy_name == "xai_visible_text_v1"
    assert match.word_count >= 80
    assert match.repeat_count >= 6
    assert match.coverage >= 0.72
    assert match.novelty <= 0.18
    assert LOOP_CLAUSE not in json.dumps(match.diagnostics())
    assert "triggering_ngram_hash" in match.diagnostics()


def test_corrupted_near_repetition_still_matches() -> None:
    detector = ro.VisibleTextRepetitionDetector(_default_policy())
    match = detector.feed(_looping_text(repeats=14, corrupt=True))
    assert match is not None
    assert match.repeat_count >= 6
    assert LOOP_CLAUSE not in json.dumps(match.diagnostics())


def test_legitimate_list_and_code_do_not_match() -> None:
    numbered = "\n".join(
        f"{index}. unique pantry item {index} about spice rack {index} layout"
        for index in range(1, 40)
    )
    code = "\n".join(
        [
            "def accumulate(values):",
            "    total = 0",
            "    for index, value in enumerate(values):",
            "        total += value",
            *[f"        bucket_{index} += 1" for index in range(40)],
            "    return total",
        ]
    )
    policy = _default_policy()
    assert ro.VisibleTextRepetitionDetector(policy).feed(numbered) is None
    assert ro.VisibleTextRepetitionDetector(policy).feed(code) is None


def test_short_visible_text_never_arms() -> None:
    detector = ro.VisibleTextRepetitionDetector(_default_policy())
    short = " ".join([LOOP_CLAUSE] * 4)  # 40 words
    assert detector.feed(short) is None


def test_reasoning_and_tool_argument_events_are_excluded() -> None:
    detector = ro.VisibleTextRepetitionDetector(_default_policy())
    looping = _looping_text(repeats=12)
    reasoning_event = {
        "type": "response.reasoning_summary_text.delta",
        "delta": looping,
    }
    tool_event = {
        "type": "response.function_call_arguments.delta",
        "delta": looping,
        "arguments": looping,
    }
    mcp_event = {
        "type": "response.mcp_call_arguments.done",
        "arguments": looping,
    }
    encrypted_event = {
        "type": "response.output_item.done",
        "item": {
            "type": "reasoning",
            "encrypted_content": looping,
        },
    }
    visible_event = {
        "type": "response.output_text.delta",
        "delta": looping,
        "item_id": "msg_1",
    }

    assert ro.extract_visible_text_from_responses_event(reasoning_event) is None
    assert ro.extract_visible_text_from_responses_event(tool_event) is None
    assert ro.extract_visible_text_from_responses_event(mcp_event) is None
    assert ro.extract_visible_text_from_responses_event(encrypted_event) is None
    assert ro.extract_visible_text_from_responses_event(visible_event) == looping

    extractor = ro.VisibleOutputTextExtractor()
    assert extractor.consume_event(reasoning_event) is None
    assert extractor.consume_event(tool_event) is None
    assert detector.feed_event(reasoning_event) is None
    assert detector.feed_event(tool_event) is None
    match = detector.feed_event(visible_event)
    assert match is not None


def test_output_text_done_is_ignored_when_deltas_already_accumulated() -> None:
    extractor = ro.VisibleOutputTextExtractor()
    delta = {
        "type": "response.output_text.delta",
        "item_id": "msg_1",
        "delta": "hello world",
    }
    done = {
        "type": "response.output_text.done",
        "item_id": "msg_1",
        "text": "hello world",
    }
    assert extractor.consume_event(delta) == "hello world"
    assert extractor.consume_event(done) is None


def test_nonstream_body_extracts_only_message_output_text() -> None:
    body = {
        "status": "completed",
        "output": [
            {
                "type": "reasoning",
                "content": [{"type": "output_text", "text": LOOP_CLAUSE}],
                "encrypted_content": LOOP_CLAUSE,
            },
            {
                "type": "function_call",
                "arguments": LOOP_CLAUSE,
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "visible one "},
                    {"type": "text", "text": "visible two"},
                    {"type": "refusal", "text": "hidden"},
                ],
            },
        ],
    }
    assert ro.extract_visible_output_text_from_response_body(body) == (
        "visible one visible two"
    )


def test_detector_keeps_only_bounded_normalized_words() -> None:
    policy = _default_policy(window_words=12)
    detector = ro.VisibleTextRepetitionDetector(policy)
    detector.feed("alpha beta gamma delta epsilon zeta eta theta iota kappa")
    detector.feed("lambda mu nu")
    words = detector.normalized_words()
    assert words == [
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "zeta",
        "eta",
        "theta",
        "iota",
        "kappa",
        "lambda",
        "mu",
        "nu",
    ]
    assert len(words) <= 12


# ---------------------------------------------------------------------------
# Config / selector
# ---------------------------------------------------------------------------


def test_default_policy_thresholds_come_from_yaml_not_model_names() -> None:
    config = ogc.load_output_guard_config(DEFAULT_YAML_PATH)
    policy = config.policies["xai_visible_text_v1"]
    assert policy.window_words == 400
    assert policy.min_words == 80
    assert policy.min_ngram == 8
    assert policy.min_repeats == 6
    assert policy.min_coverage == pytest.approx(0.72)
    assert policy.growth_words == 120
    assert policy.max_novelty == pytest.approx(0.18)
    assert policy.max_ngram_edit_distance == 2
    assert policy.inspect == "visible_output_text"

    yaml_text = DEFAULT_YAML_PATH.read_text(encoding="utf-8")
    module_text = CONFIG_MODULE_PATH.read_text(encoding="utf-8")
    detector_text = DETECTOR_PATH.read_text(encoding="utf-8")
    for forbidden in ("grok-4.6", "oa_xai/", "oa_xai"):
        assert forbidden not in yaml_text
        assert forbidden not in module_text
        assert forbidden not in detector_text


def test_selector_enables_resolved_xai_on_openai_passthrough_responses() -> None:
    config = ogc.load_output_guard_config(DEFAULT_YAML_PATH)
    for context in (
        _xai_passthrough_context(route_family="codex_xai_oauth_responses_adapter"),
        _xai_passthrough_context(
            custom_llm_provider="openai",
            egress_credential_family="xai",
            route_family="openai_responses",
        ),
        _xai_passthrough_context(
            custom_llm_provider="xai",
            egress_credential_family=None,
            route_family="codex_auto_agent_xai_oauth_responses",
            ingress_path="/openai_passthrough/responses",
        ),
        _xai_passthrough_context(
            custom_llm_provider="openai",
            egress_credential_family=None,
            route_family="codex_auto_agent_grok_native_responses",
        ),
    ):
        selected = ogc.resolve_output_guard_policy(context, config=config)
        assert selected is not None
        assert selected.name == "xai_visible_text_v1"


def test_selector_off_for_openai_provider_and_direct_grok() -> None:
    config = ogc.load_output_guard_config(DEFAULT_YAML_PATH)
    openai_context = _xai_passthrough_context(
        custom_llm_provider="openai",
        egress_credential_family="openai",
        route_family="codex_responses",
    )
    grok_context = _xai_passthrough_context(
        ingress_path="/grok/v1/responses",
        custom_llm_provider="xai",
        egress_credential_family="xai",
        route_family="grok_cli_chat_proxy",
    )
    chat_context = _xai_passthrough_context(
        ingress_path="/openai_passthrough/v1/chat/completions",
    )
    anthropic_context = _xai_passthrough_context(
        ingress_path="/v1/messages",
        custom_llm_provider="anthropic",
        egress_credential_family="anthropic",
        route_family="anthropic_messages",
    )
    assert ogc.resolve_output_guard_policy(openai_context, config=config) is None
    assert ogc.resolve_output_guard_policy(grok_context, config=config) is None
    assert ogc.resolve_output_guard_policy(chat_context, config=config) is None
    assert ogc.resolve_output_guard_policy(anthropic_context, config=config) is None


def test_custom_yaml_thresholds_are_used_by_detector(tmp_path: Path) -> None:
    yaml_path = tmp_path / "output_guards.yaml"
    yaml_path.write_text(
        """
output_guards:
  policies:
    xai_visible_text_v1:
      inspect: visible_output_text
      window_words: 40
      min_words: 8
      min_ngram: 3
      min_repeats: 3
      min_coverage: 0.5
      growth_words: 8
      max_novelty: 0.5
      max_ngram_edit_distance: 2
  selectors:
    - match:
        ingress: openai_passthrough_responses
        provider: xai
      policy: xai_visible_text_v1
""".strip()
        + "\n",
        encoding="utf-8",
    )
    config = ogc.load_output_guard_config(yaml_path)
    policy = ogc.resolve_output_guard_policy(
        _xai_passthrough_context(),
        config=config,
    )
    assert policy is not None
    assert policy.min_words == 8
    assert policy.min_ngram == 3
    match = ro.VisibleTextRepetitionDetector(policy).feed(
        "red cat sat red cat sat red cat sat red cat sat"
    )
    assert match is not None
    assert match.policy_name == "xai_visible_text_v1"


def test_failure_kind_is_not_capacity_quota_or_cooldown() -> None:
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.failure_actions import (
        DEFAULT_FAILURE_ACTION_POLICY,
        FAILURE_ACTION_ENFORCEMENT_ENABLED,
    )
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.failure_vocabulary import (
        FailureEvent,
        is_coolable,
    )

    assert ro.REPETITIVE_OUTPUT_FAILURE_KIND == FAILURE_KIND
    assert ro.REPETITIVE_OUTPUT_ERROR_CODE == REPETITIVE_OUTPUT_CODE
    action = DEFAULT_FAILURE_ACTION_POLICY.action_for(FAILURE_KIND)
    assert action == "terminal"
    assert FAILURE_ACTION_ENFORCEMENT_ENABLED is False
    event = FailureEvent(
        class_name=FAILURE_KIND,
        origin="upstream",
        confidence="structured",
        provider="xai",
        scope="lane",
        retryable=False,
        evidence={"code": REPETITIVE_OUTPUT_CODE},
    )
    # Content-loop abort must not cool a candidate even if origin is upstream.
    assert action not in {"cooldown", "retry_same", "failover", "redispatch"}
    shadow_coolable = is_coolable(event) and action == "cooldown"
    assert shadow_coolable is False


# ---------------------------------------------------------------------------
# Streaming / non-stream hooks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_match_emits_failed_closes_upstream_and_does_not_retry() -> None:
    policy = _default_policy()
    looping = _looping_text(repeats=12)
    pieces = [looping[index : index + 24] for index in range(0, len(looping), 24)]
    remaining_after_match = _output_text_delta("SHOULD_NOT_BE_FORWARDED")
    completed = _sse_event(
        "response.completed",
        {"response": {"id": "resp_loop", "status": "completed", "output": []}},
    )

    class _Upstream:
        def __init__(self) -> None:
            self.aclose = AsyncMock()
            self.close = MagicMock()

        async def aiter_bytes(self):
            for piece in pieces:
                yield _output_text_delta(piece)
            yield remaining_after_match
            yield completed

    upstream = _Upstream()
    retry_fn = MagicMock(name="retry_upstream")
    failover_fn = MagicMock(name="failover_alias")

    chunks: list[bytes] = []
    async for chunk in ro.wrap_responses_sse_with_repetitive_output_guard(
        upstream.aiter_bytes(),
        policy=policy,
        request_context=_xai_passthrough_context(),
        upstream_response=upstream,
        retry_fn=retry_fn,
        failover_fn=failover_fn,
    ):
        chunks.append(chunk if isinstance(chunk, bytes) else str(chunk).encode("utf-8"))

    events = _decode_sse_events(chunks)
    assert any(event.get("type") == "response.output_text.delta" for event in events)
    failed = next(event for event in events if event.get("type") == "response.failed")
    error = failed["response"]["error"]
    assert failed["response"]["status"] == "failed"
    assert error["code"] == REPETITIVE_OUTPUT_CODE
    assert error["type"] == "proxy_stream_terminal_error"
    assert events[-1]["type"] == "[DONE]"
    forwarded = b"".join(chunks)
    assert b"SHOULD_NOT_BE_FORWARDED" not in forwarded
    assert LOOP_CLAUSE.encode() not in json.dumps(failed).encode()
    metadata = failed["response"]["metadata"]
    assert metadata["failure_kind"] == FAILURE_KIND
    assert metadata["policy"] == "xai_visible_text_v1"
    assert metadata["termination_action"] == "response.failed"
    assert metadata["visible_delta_forwarded"] is True
    assert metadata["provider"] == "xai"
    assert metadata["route_family"] == "codex_xai_oauth_responses_adapter"
    assert "triggering_ngram_hash" in metadata
    assert upstream.aclose.await_count == 1
    retry_fn.assert_not_called()
    failover_fn.assert_not_called()


@pytest.mark.asyncio
async def test_selector_off_leaves_openai_and_grok_streams_unchanged() -> None:
    looping = _looping_text(repeats=12)
    source_chunks = [
        _output_text_delta(looping),
        _sse_event(
            "response.completed",
            {"response": {"id": "resp_ok", "status": "completed", "output": []}},
        ),
        b"data: [DONE]\n\n",
    ]

    async def _gen():
        for chunk in source_chunks:
            yield chunk

    openai_chunks: list[bytes] = []
    async for chunk in ro.maybe_wrap_passthrough_responses_stream(
        _gen(),
        request_context=_xai_passthrough_context(
            custom_llm_provider="openai",
            egress_credential_family="openai",
            route_family="codex_responses",
        ),
        upstream_response=None,
    ):
        openai_chunks.append(chunk)
    assert openai_chunks == source_chunks

    grok_chunks: list[bytes] = []
    async for chunk in ro.maybe_wrap_passthrough_responses_stream(
        _gen(),
        request_context=_xai_passthrough_context(
            ingress_path="/grok/v1/responses",
            route_family="grok_cli_chat_proxy",
        ),
        upstream_response=None,
    ):
        grok_chunks.append(chunk)
    assert grok_chunks == source_chunks


def test_nonstream_looped_output_text_is_rejected() -> None:
    body = {
        "object": "response",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": _looping_text(repeats=12)},
                ],
            }
        ],
    }
    with pytest.raises(HTTPException) as exc_info:
        ro.reject_nonstream_responses_body_if_repetitive(
            body,
            policy=_default_policy(),
            request_context=_xai_passthrough_context(),
        )
    assert exc_info.value.status_code == 502
    detail = exc_info.value.detail
    error = detail["error"] if isinstance(detail, dict) else {}
    assert error["code"] == REPETITIVE_OUTPUT_CODE
    assert body["status"] == "completed"


def test_maybe_reject_nonstream_is_noop_when_selector_off() -> None:
    body = {
        "status": "completed",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": _looping_text(repeats=12)}],
            }
        ],
    }
    result = ro.maybe_reject_passthrough_responses_body(
        body,
        request_context=_xai_passthrough_context(
            custom_llm_provider="openai",
            egress_credential_family="openai",
            route_family="codex_responses",
        ),
    )
    assert result is body


def test_grok_proxy_route_does_not_require_streaming_abort_path() -> None:
    source = GROK_ROUTE_MODULE_PATH.read_text(encoding="utf-8")
    module = ast.parse(source)
    grok_fn = next(
        node
        for node in module.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "grok_proxy_route"
    )
    called_names: set[str] = set()
    for node in ast.walk(grok_fn):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                called_names.add(func.id)
            elif isinstance(func, ast.Attribute):
                called_names.add(func.attr)
    assert "wrap_responses_sse_with_repetitive_output_guard" not in called_names
    assert "maybe_wrap_passthrough_responses_stream" not in called_names
    assert "reject_nonstream_responses_body_if_repetitive" not in called_names
    grok_source = ast.get_source_segment(source, grok_fn) or ""
    assert "repetitive_output" not in grok_source
    assert inspect.getsource(grok_proxy_route)
    assert "repetitive_output" not in inspect.getsource(grok_proxy_route)


def _function_source(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(source, node) or ""
    raise AssertionError(f"{name} not found in {path}")


def test_pass_through_request_installs_live_forward_repetitive_output_hooks() -> None:
    source = _function_source(PASS_THROUGH_REQUEST_PATH, "pass_through_request")
    assert "maybe_wrap_passthrough_responses_stream" in source
    assert "maybe_reject_passthrough_responses_body" in source
    assert "output_guard_context_from_passthrough" in source
    wrap_count = source.count("maybe_wrap_passthrough_responses_stream")
    assert wrap_count >= 2


def test_alias_stream_sites_install_live_forward_wrapper_when_bypassing_pass_through_request() -> None:
    wrap_name = "maybe_wrap_passthrough_responses_stream"
    extra_sources = [
        CODEX_CANDIDATE_CALLS_PATH.read_text(encoding="utf-8"),
        ALIAS_STREAMING_PATH.read_text(encoding="utf-8"),
    ]
    assert any(wrap_name in source for source in extra_sources)
    for source in extra_sources:
        if wrap_name in source:
            assert "output_guard_context_from_passthrough" in source or (
                "OutputGuardRequestContext" in source
            )


def test_diagnostics_hash_is_stable_and_content_free() -> None:
    detector = ro.VisibleTextRepetitionDetector(_default_policy())
    match = detector.feed(_looping_text(repeats=12))
    assert match is not None
    diagnostics = match.diagnostics()
    expected = hashlib.sha256(
        match.triggering_ngram_fingerprint.encode("utf-8")
    ).hexdigest()
    assert diagnostics["triggering_ngram_hash"] == expected
    dumped = json.dumps(diagnostics)
    assert LOOP_CLAUSE not in dumped
    for word in LOOP_CLAUSE.split():
        assert word not in dumped
