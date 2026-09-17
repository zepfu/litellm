from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import patch

import pytest

from litellm.llms.cursor_agent.connect import CursorConnectError
from litellm.llms.custom_httpx.async_client_cleanup import (
    close_litellm_async_clients,
)
from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime import (
    codex_candidate_calls,
)


@pytest.fixture(autouse=True)
def _clear_replay_registry() -> None:
    codex_candidate_calls._clear_cursor_replay_registry()
    yield
    codex_candidate_calls._clear_cursor_replay_registry()


class _AcloseCountingSession:
    def __init__(self) -> None:
        self.close_calls = 0
        self.aclose_calls = 0

    def close(self) -> None:
        self.close_calls += 1

    async def aclose(self) -> None:
        self.aclose_calls += 1


def _continuation_failure() -> CursorConnectError:
    continuation_exc = CursorConnectError(
        "missing retained session",
        status_code=409,
    )
    setattr(
        continuation_exc,
        codex_candidate_calls._CURSOR_SESSION_CONTINUATION_FAILURE_MARKER,
        True,
    )
    return continuation_exc


def _stock_message(*, role: str, text: str, item_id: str) -> dict[str, Any]:
    content_type = "output_text" if role == "assistant" else "input_text"
    return {
        "type": "message",
        "id": item_id,
        "role": role,
        "content": [{"type": content_type, "text": text}],
    }


def _stock_function_call(*, call_id: str, cmd: str) -> dict[str, Any]:
    return {
        "type": "function_call",
        "id": f"fc_{call_id}",
        "name": "exec_command",
        "arguments": f'{{"cmd":"{cmd}"}}',
        "call_id": call_id,
    }


def _stock_function_call_output(*, call_id: str, output: str) -> dict[str, Any]:
    return {
        "type": "function_call_output",
        "id": f"fco_{uuid.uuid4()}",
        "call_id": call_id,
        "output": output,
    }


def _stock_multi_call_body(
    pairs: list[tuple[str, str, str]],
    *,
    extra_items: Optional[list[dict[str, Any]]] = None,
    tools: Optional[list[Any]] = None,
) -> dict[str, Any]:
    input_items: list[dict[str, Any]] = [
        _stock_message(
            role="user",
            text="Run the requested commands, then continue.",
            item_id="msg_01a06269-1827-79d2-b3a5-41ed4566fa70",
        )
    ]
    for call_id, cmd, output in pairs:
        input_items.append(_stock_function_call(call_id=call_id, cmd=cmd))
        input_items.append(_stock_function_call_output(call_id=call_id, output=output))
    if extra_items:
        input_items.extend(extra_items)
    return {
        "model": "work",
        "tools": [] if tools is None else tools,
        "input": input_items,
    }


def test_stock_parser_keeps_single_call_replay() -> None:
    body = _stock_multi_call_body([("call-pwd00001", "pwd", "/workspace")])
    result = codex_candidate_calls._cursor_replay_stock_codex_full_history_input(body)

    assert result.rejection is None
    assert result.value[-2:] == [
        {
            "type": "function_call",
            "call_id": "call-pwd00001",
            "name": "exec_command",
            "arguments": '{"cmd":"pwd"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call-pwd00001",
            "output": "/workspace",
        },
    ]


def test_stock_parser_accepts_bounded_parallel_and_sequential_calls() -> None:
    sequential = _stock_multi_call_body(
        [
            ("call-pwd00001", "pwd", "/workspace"),
            ("call-ls0000002", "ls", "README.md"),
        ]
    )
    sequential_result = (
        codex_candidate_calls._cursor_replay_stock_codex_full_history_input(sequential)
    )
    assert sequential_result.rejection is None
    assert [
        item["call_id"]
        for item in sequential_result.value
        if item.get("type") in {"function_call", "function_call_output"}
    ] == [
        "call-pwd00001",
        "call-pwd00001",
        "call-ls0000002",
        "call-ls0000002",
    ]

    parallel = _stock_multi_call_body([])
    parallel["input"].extend(
        [
            _stock_function_call(call_id="call-pwd00001", cmd="pwd"),
            _stock_function_call(call_id="call-ls0000002", cmd="ls"),
            _stock_function_call_output(
                call_id="call-ls0000002",
                output="README.md",
            ),
            _stock_function_call_output(
                call_id="call-pwd00001",
                output="/workspace",
            ),
        ]
    )
    parallel_result = (
        codex_candidate_calls._cursor_replay_stock_codex_full_history_input(parallel)
    )
    assert parallel_result.rejection is None
    rebuilt = codex_candidate_calls._build_cursor_replay_safe_fresh_dispatch_body(
        parallel,
        continuation_exc=_continuation_failure(),
    )
    assert rebuilt is not None
    assert [item["call_id"] for item in rebuilt["input"][1:]] == [
        "call-pwd00001",
        "call-ls0000002",
        "call-ls0000002",
        "call-pwd00001",
    ]


@pytest.mark.parametrize(
    ("mutate_body", "expected_reason"),
    [
        (
            lambda body: body["input"][3].update(
                {"call_id": "call-pwd00001", "id": "fc_call-pwd00001"}
            ),
            "call_graph",
        ),
        (
            lambda body: body["input"].insert(
                2,
                _stock_function_call_output(
                    call_id="call-ls0000002",
                    output="README.md",
                ),
            ),
            "unresolved_call_id",
        ),
        (
            lambda body: body["input"][4].update({"call_id": "call-unknown"}),
            "unresolved_call_id",
        ),
        (
            lambda body: body["input"].pop(),
            "unresolved_call_id",
        ),
        (
            lambda body: body["input"].append(
                {
                    "type": "reasoning",
                    "id": "rs_provider-owned-state",
                    "encrypted_content": "opaque-provider-state",
                }
            ),
            "item_type",
        ),
        (
            lambda body: body["input"][1].update(
                {"previous_response_id": "resp-nested"}
            ),
            "cursor_continuation_identifier",
        ),
    ],
    ids=[
        "duplicate-call-id",
        "output-before-matching-call",
        "unresolved-output-id",
        "missing-output",
        "unsupported-opaque-item",
        "nested-continuation-identifier",
    ],
)
def test_stock_parser_fails_closed_on_ambiguous_or_opaque_graphs(
    mutate_body: Any,
    expected_reason: str,
) -> None:
    body = _stock_multi_call_body(
        [
            ("call-pwd00001", "pwd", "/workspace"),
            ("call-ls0000002", "ls", "README.md"),
        ]
    )
    mutate_body(body)
    result = codex_candidate_calls._cursor_replay_stock_codex_full_history_input(body)
    assert result.rejection is not None
    assert result.rejection.reason == expected_reason


def test_stock_parser_rejects_extra_function_calls_beyond_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        codex_candidate_calls,
        "_CURSOR_REPLAY_MAX_STOCK_FUNCTION_CALLS",
        1,
    )
    body = _stock_multi_call_body(
        [
            ("call-pwd00001", "pwd", "/workspace"),
            ("call-ls0000002", "ls", "README.md"),
        ]
    )
    result = codex_candidate_calls._cursor_replay_stock_codex_full_history_input(body)
    assert result.rejection is not None
    assert result.rejection.reason == "function_call_count"


def test_replay_registry_rejects_oversized_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(codex_candidate_calls, "_CURSOR_REPLAY_MAX_ENTRY_BYTES", 32)
    with pytest.raises(CursorConnectError, match="per-entry replay bound"):
        codex_candidate_calls._store_cursor_replay_state(
            "resp-too-large",
            messages=[{"role": "user", "content": "x" * 64}],
            tools=[],
        )
    assert "resp-too-large" not in codex_candidate_calls._CURSOR_REPLAY_REGISTRY


def test_replay_registry_evicts_to_stay_within_byte_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(codex_candidate_calls, "_CURSOR_REPLAY_MAX_ENTRY_BYTES", 1024)
    monkeypatch.setattr(codex_candidate_calls, "_CURSOR_REPLAY_MAX_TOTAL_BYTES", 220)
    first_messages = [{"role": "user", "content": "first-entry"}]
    second_messages = [{"role": "user", "content": "second-entry"}]
    first_bytes = codex_candidate_calls._cursor_replay_entry_payload_bytes(
        messages=first_messages,
        tools=[],
        pending_call_ids=[],
        continuation_outcome=None,
        owner_scope=None,
    )
    second_bytes = codex_candidate_calls._cursor_replay_entry_payload_bytes(
        messages=second_messages,
        tools=[],
        pending_call_ids=[],
        continuation_outcome=None,
        owner_scope=None,
    )
    assert first_bytes is not None and second_bytes is not None
    assert first_bytes + second_bytes > 220

    codex_candidate_calls._store_cursor_replay_state(
        "resp-old-bytes",
        messages=first_messages,
        tools=[],
    )
    codex_candidate_calls._store_cursor_replay_state(
        "resp-new-bytes",
        messages=second_messages,
        tools=[],
    )

    with pytest.raises(CursorConnectError, match="missing"):
        codex_candidate_calls._take_cursor_replay_state("resp-old-bytes")
    assert (
        codex_candidate_calls._take_cursor_replay_state("resp-new-bytes")["messages"]
        == second_messages
    )
    assert codex_candidate_calls._cursor_replay_registry_total_bytes() == 0


def test_replay_registry_replacement_keeps_prior_entry_if_construction_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_deepcopy = codex_candidate_calls.copy.deepcopy
    session = object()
    codex_candidate_calls._store_cursor_replay_state(
        "resp-atomic",
        messages=[{"role": "user", "content": "keep-me"}],
        tools=[],
        retained_session=session,
    )

    def _failing_deepcopy(value: Any) -> Any:
        raise RuntimeError("copy failed")

    monkeypatch.setattr(codex_candidate_calls.copy, "deepcopy", _failing_deepcopy)
    with pytest.raises(RuntimeError, match="copy failed"):
        codex_candidate_calls._store_cursor_replay_state(
            "resp-atomic",
            messages=[{"role": "user", "content": "replacement"}],
            tools=[],
            retained_session=object(),
        )

    monkeypatch.setattr(codex_candidate_calls.copy, "deepcopy", original_deepcopy)
    state = codex_candidate_calls._peek_cursor_replay_state("resp-atomic")
    assert state["messages"] == [{"role": "user", "content": "keep-me"}]
    assert state["retained_session"] is session


def test_replay_registry_replacement_keeps_prior_if_schedule_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = object()
    replacement_session = object()
    codex_candidate_calls._store_cursor_replay_state(
        "resp-schedule-atomic",
        messages=[{"role": "user", "content": "keep-me"}],
        tools=[],
        retained_session=session,
    )

    def _failing_schedule(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("schedule failed")

    monkeypatch.setattr(
        codex_candidate_calls,
        "_schedule_cursor_replay_expiry",
        _failing_schedule,
    )
    with pytest.raises(RuntimeError, match="schedule failed"):
        codex_candidate_calls._store_cursor_replay_state(
            "resp-schedule-atomic",
            messages=[{"role": "user", "content": "replacement"}],
            tools=[],
            retained_session=replacement_session,
        )

    state = codex_candidate_calls._peek_cursor_replay_state("resp-schedule-atomic")
    assert state["messages"] == [{"role": "user", "content": "keep-me"}]
    assert state["retained_session"] is session
    assert (
        state["payload_bytes"]
        == codex_candidate_calls._cursor_replay_registry_total_bytes()
    )


@pytest.mark.asyncio
async def test_replay_registry_disposes_with_bounded_aclose_not_sync_close() -> None:
    session = _AcloseCountingSession()
    codex_candidate_calls._store_cursor_replay_state(
        "resp-async-close",
        messages=[{"role": "user", "content": "close me"}],
        tools=[],
        retained_session=session,
    )
    codex_candidate_calls._consume_cursor_replay_state("resp-async-close")
    await codex_candidate_calls._await_cursor_replay_disposal_tasks()

    assert session.aclose_calls == 1
    assert session.close_calls == 0


@pytest.mark.asyncio
async def test_close_litellm_async_clients_acloses_replay_registry_once() -> None:
    import litellm

    session = _AcloseCountingSession()
    codex_candidate_calls._store_cursor_replay_state(
        "resp-shutdown",
        messages=[{"role": "user", "content": "shutdown"}],
        tools=[],
        retained_session=session,
    )
    cache = SimpleNamespace(cache_dict={}, _evicted_clients_retained=[])

    with patch.object(litellm, "in_memory_llm_clients_cache", cache), patch.object(
        litellm, "base_llm_aiohttp_handler", None
    ):
        await close_litellm_async_clients()
        await close_litellm_async_clients()

    assert session.aclose_calls == 1
    assert "resp-shutdown" not in codex_candidate_calls._CURSOR_REPLAY_REGISTRY


def test_replay_validation_receipt_reuses_unchanged_body_and_invalidates_on_mutation() -> (
    None
):
    body = _stock_multi_call_body(
        [
            ("call-pwd00001", "pwd", "/workspace"),
            ("call-ls0000002", "ls", "README.md"),
        ]
    )
    rebuilt = codex_candidate_calls._build_cursor_replay_safe_fresh_dispatch_body(
        body,
        continuation_exc=_continuation_failure(),
    )
    assert rebuilt is not None
    receipt = codex_candidate_calls._get_cursor_replay_request_receipt(rebuilt)
    assert receipt is not None
    assert receipt.history == rebuilt["input"]
    assert receipt.tools == rebuilt["tools"]
    assert receipt.opaque_state_rejected is True
    assert codex_candidate_calls._CURSOR_REPLAY_VALIDATION_RUN_COUNT == 1
    assert codex_candidate_calls._CURSOR_REPLAY_RECEIPT_REUSE_COUNT == 0

    reused = codex_candidate_calls._build_cursor_replay_safe_fresh_dispatch_body(
        rebuilt,
        continuation_exc=_continuation_failure(),
    )
    assert reused is rebuilt
    assert codex_candidate_calls._get_cursor_replay_request_receipt(rebuilt) is receipt
    assert codex_candidate_calls._CURSOR_REPLAY_VALIDATION_RUN_COUNT == 1
    assert codex_candidate_calls._CURSOR_REPLAY_RECEIPT_REUSE_COUNT == 1

    replaced = dict(rebuilt)
    assert codex_candidate_calls._get_cursor_replay_request_receipt(replaced) is None
    rebuilt["input"] = list(rebuilt["input"])
    rebuilt["input"].append(
        {
            "type": "reasoning",
            "id": "rs_mutated",
        }
    )
    assert codex_candidate_calls._get_cursor_replay_request_receipt(rebuilt) is None
    mutated = codex_candidate_calls._build_cursor_replay_safe_fresh_dispatch_body(
        rebuilt,
        continuation_exc=_continuation_failure(),
    )
    assert mutated is None
    assert codex_candidate_calls._CURSOR_REPLAY_VALIDATION_RUN_COUNT == 2


def test_replay_validation_receipt_adopts_onto_recursive_selection_body() -> None:
    body = _stock_multi_call_body([("call-pwd00001", "pwd", "/workspace")])
    rebuilt = codex_candidate_calls._build_cursor_replay_safe_fresh_dispatch_body(
        body,
        continuation_exc=_continuation_failure(),
    )
    assert rebuilt is not None
    source_receipt = codex_candidate_calls._get_cursor_replay_request_receipt(rebuilt)
    assert source_receipt is not None

    merged = dict(rebuilt)
    merged["litellm_metadata"] = {"aawm_redispatch_ordinal": 1}
    adopted = codex_candidate_calls._adopt_cursor_replay_request_receipt(
        rebuilt,
        merged,
    )

    assert adopted is not None
    assert adopted.body_ref is merged
    assert adopted.history is source_receipt.history
    assert adopted.tools is source_receipt.tools
    assert adopted.call_graph is source_receipt.call_graph
    assert codex_candidate_calls._get_cursor_replay_request_receipt(merged) is adopted
    assert codex_candidate_calls._get_cursor_replay_request_receipt(rebuilt) is None
    reused = codex_candidate_calls._build_cursor_replay_safe_fresh_dispatch_body(
        merged,
        continuation_exc=_continuation_failure(),
    )
    assert reused is merged
    assert codex_candidate_calls._CURSOR_REPLAY_VALIDATION_RUN_COUNT == 1
    assert codex_candidate_calls._CURSOR_REPLAY_RECEIPT_REUSE_COUNT == 1
