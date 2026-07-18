"""RR-054 regression: tool-call cache scope_key propagation.

Google Code Assist tool-name/argument repair cache helpers accept an optional
session/tenant ``scope_key`` so the same ``tool_call_id`` can safely coexist
across concurrent sessions. Production call chains that remember and later look
up those entries must propagate a stable per-session scope; otherwise the last
writer wins and tool-pair repair can re-emit the wrong name/arguments.

This module is tests-only and production-read-only. It exercises the real
restore → ensure / build call chains and requires stable isolation for the same
``tool_call_id`` across two sessions.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi import Request

from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe


SHARED_TOOL_CALL_ID = "call_shared_rr054_scope"
SESSION_A = "session-a-rr054-tool-cache"
SESSION_B = "session-b-rr054-tool-cache"


def _clear_tool_call_caches() -> None:
    lpe._codex_google_code_assist_tool_call_name_cache.clear()
    lpe._codex_google_code_assist_tool_call_arguments_cache.clear()


def _dual_function_tools() -> list[dict[str, Any]]:
    """Two advertised tools so ensure/build cannot fall back to single-tool inference."""
    return [
        {
            "type": "function",
            "function": {
                "name": "Read",
                "description": "read a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "Write",
                "description": "write a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            },
        },
    ]


def _response_with_tool_call(
    *,
    tool_call_id: str,
    function_name: str,
    function_arguments: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=[
                        SimpleNamespace(
                            id=tool_call_id,
                            function=SimpleNamespace(
                                name=function_name,
                                arguments=function_arguments,
                            ),
                        )
                    ]
                ),
                delta=None,
            )
        ]
    )


def _mock_request(session_id: str) -> MagicMock:
    request = MagicMock(spec=Request)
    request.headers = {"session_id": session_id}
    return request


def _assistant_tool_call_names(messages: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            if isinstance(name, str) and name:
                names.append(name)
    return names


def _assistant_tool_call_arguments(messages: list[dict[str, Any]]) -> list[str]:
    args: list[str] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                args.append(arguments)
    return args


def _tool_result_only_kwargs(*, tool_call_id: str, tool_content: str) -> dict[str, Any]:
    return {
        "model": "claude-sonnet-4-6",
        "max_tokens": 32,
        "messages": [
            {"role": "user", "content": "continue after tools"},
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": tool_content,
            },
        ],
        "tools": _dual_function_tools(),
    }


def test_rr054_tool_cache_helpers_isolate_same_tool_call_id_by_scope_key() -> None:
    """Helper layer: same tool_call_id under two scopes must not collide."""
    _clear_tool_call_caches()
    try:
        lpe._remember_codex_google_code_assist_tool_call_name(
            SHARED_TOOL_CALL_ID,
            "Read",
            '{"path":"a.txt"}',
            scope_key=SESSION_A,
        )
        lpe._remember_codex_google_code_assist_tool_call_name(
            SHARED_TOOL_CALL_ID,
            "Write",
            '{"path":"b.txt","content":"x"}',
            scope_key=SESSION_B,
        )

        assert (
            lpe._lookup_codex_google_code_assist_tool_call_name(
                SHARED_TOOL_CALL_ID, scope_key=SESSION_A
            )
            == "Read"
        )
        assert (
            lpe._lookup_codex_google_code_assist_tool_call_name(
                SHARED_TOOL_CALL_ID, scope_key=SESSION_B
            )
            == "Write"
        )
        assert (
            lpe._lookup_codex_google_code_assist_tool_call_arguments(
                SHARED_TOOL_CALL_ID, scope_key=SESSION_A
            )
            == '{"path":"a.txt"}'
        )
        assert (
            lpe._lookup_codex_google_code_assist_tool_call_arguments(
                SHARED_TOOL_CALL_ID, scope_key=SESSION_B
            )
            == '{"path":"b.txt","content":"x"}'
        )
        thought_id = f"{SHARED_TOOL_CALL_ID}__thought__sig"
        assert (
            lpe._lookup_codex_google_code_assist_tool_call_name(
                thought_id, scope_key=SESSION_A
            )
            == "Read"
        )
        assert (
            lpe._lookup_codex_google_code_assist_tool_call_name(
                thought_id, scope_key=SESSION_B
            )
            == "Write"
        )
        assert (
            lpe._lookup_codex_google_code_assist_tool_call_name(
                SHARED_TOOL_CALL_ID, scope_key="session-unrelated"
            )
            is None
        )
        assert (
            lpe._lookup_codex_google_code_assist_tool_call_name(SHARED_TOOL_CALL_ID)
            is None
        )
    finally:
        _clear_tool_call_caches()


def test_rr054_tool_cache_scope_key_format_is_stable() -> None:
    """Cache keys must stably prefix cleaned scope ahead of tool_call_id."""
    key_a = lpe._codex_google_code_assist_tool_call_cache_key(
        SHARED_TOOL_CALL_ID, scope_key=f"  {SESSION_A}  "
    )
    key_b = lpe._codex_google_code_assist_tool_call_cache_key(
        SHARED_TOOL_CALL_ID, scope_key=SESSION_B
    )
    key_none = lpe._codex_google_code_assist_tool_call_cache_key(SHARED_TOOL_CALL_ID)
    assert key_a == f"{SESSION_A}:{SHARED_TOOL_CALL_ID}"
    assert key_b == f"{SESSION_B}:{SHARED_TOOL_CALL_ID}"
    assert key_none == SHARED_TOOL_CALL_ID
    assert key_a != key_b
    assert key_a != key_none


def test_rr054_tool_cache_restore_forwards_scope_key_into_remember() -> None:
    """Production restore must forward session scope into remember().

    Two sessions share a tool_call_id; each restore must remember under its own
    scope so concurrent sessions do not overwrite each other.
    """
    _clear_tool_call_caches()
    remembered: list[dict[str, Any]] = []

    original_remember = lpe._remember_codex_google_code_assist_tool_call_name

    def _capturing_remember(
        tool_call_id: Any,
        function_name: Any,
        function_arguments: Any = None,
        *,
        scope_key: Any = None,
    ) -> None:
        remembered.append(
            {
                "tool_call_id": tool_call_id,
                "function_name": function_name,
                "function_arguments": function_arguments,
                "scope_key": scope_key,
            }
        )
        original_remember(
            tool_call_id,
            function_name,
            function_arguments,
            scope_key=scope_key,
        )

    try:
        with patch.object(
            lpe,
            "_remember_codex_google_code_assist_tool_call_name",
            side_effect=_capturing_remember,
        ):
            # Preferred contract: restore accepts and forwards scope_key.
            try:
                lpe._restore_google_adapter_tool_call_names(
                    _response_with_tool_call(
                        tool_call_id=SHARED_TOOL_CALL_ID,
                        function_name="Read",
                        function_arguments='{"path":"a.txt"}',
                    ),
                    {},
                    scope_key=SESSION_A,  # type: ignore[call-arg]
                )
                lpe._restore_google_adapter_tool_call_names(
                    _response_with_tool_call(
                        tool_call_id=SHARED_TOOL_CALL_ID,
                        function_name="Write",
                        function_arguments='{"path":"b.txt","content":"x"}',
                    ),
                    {},
                    scope_key=SESSION_B,  # type: ignore[call-arg]
                )
            except TypeError as exc:
                pytest.fail(
                    "RR-054 scope propagation gap: "
                    "_restore_google_adapter_tool_call_names does not accept "
                    f"scope_key ({exc}). Production restore must take a session "
                    "scope and forward it into "
                    "_remember_codex_google_code_assist_tool_call_name so the "
                    "same tool_call_id stays isolated across sessions."
                )

        assert len(remembered) == 2, remembered
        scopes = [entry.get("scope_key") for entry in remembered]
        assert scopes == [SESSION_A, SESSION_B], (
            "restore must forward distinct session scope_key values into "
            f"remember(); got {scopes!r}"
        )
        assert (
            lpe._lookup_codex_google_code_assist_tool_call_name(
                SHARED_TOOL_CALL_ID, scope_key=SESSION_A
            )
            == "Read"
        )
        assert (
            lpe._lookup_codex_google_code_assist_tool_call_name(
                SHARED_TOOL_CALL_ID, scope_key=SESSION_B
            )
            == "Write"
        )
    finally:
        _clear_tool_call_caches()


def test_rr054_tool_cache_ensure_forwards_scope_key_into_lookup() -> None:
    """Production ensure/repair must look up tool cache under session scope."""
    _clear_tool_call_caches()
    lookup_scopes: list[Any] = []
    original_lookup_name = lpe._lookup_codex_google_code_assist_tool_call_name
    original_lookup_args = lpe._lookup_codex_google_code_assist_tool_call_arguments

    def _capturing_lookup_name(tool_call_id: Any, *, scope_key: Any = None) -> Any:
        lookup_scopes.append(("name", scope_key))
        return original_lookup_name(tool_call_id, scope_key=scope_key)

    def _capturing_lookup_args(tool_call_id: Any, *, scope_key: Any = None) -> Any:
        lookup_scopes.append(("args", scope_key))
        return original_lookup_args(tool_call_id, scope_key=scope_key)

    try:
        lpe._remember_codex_google_code_assist_tool_call_name(
            SHARED_TOOL_CALL_ID,
            "Read",
            '{"path":"a.txt"}',
            scope_key=SESSION_A,
        )
        lpe._remember_codex_google_code_assist_tool_call_name(
            SHARED_TOOL_CALL_ID,
            "Write",
            '{"path":"b.txt","content":"x"}',
            scope_key=SESSION_B,
        )

        kwargs_a = _tool_result_only_kwargs(
            tool_call_id=SHARED_TOOL_CALL_ID,
            tool_content="file-a-contents",
        )
        kwargs_b = _tool_result_only_kwargs(
            tool_call_id=SHARED_TOOL_CALL_ID,
            tool_content="wrote-b",
        )

        with (
            patch.object(
                lpe,
                "_lookup_codex_google_code_assist_tool_call_name",
                side_effect=_capturing_lookup_name,
            ),
            patch.object(
                lpe,
                "_lookup_codex_google_code_assist_tool_call_arguments",
                side_effect=_capturing_lookup_args,
            ),
        ):
            try:
                updated_a, changes_a = (
                    lpe._ensure_codex_google_code_assist_tool_results_have_calls(
                        kwargs_a,
                        scope_key=SESSION_A,  # type: ignore[call-arg]
                    )
                )
                updated_b, changes_b = (
                    lpe._ensure_codex_google_code_assist_tool_results_have_calls(
                        kwargs_b,
                        scope_key=SESSION_B,  # type: ignore[call-arg]
                    )
                )
            except TypeError as exc:
                pytest.fail(
                    "RR-054 scope propagation gap: "
                    "_ensure_codex_google_code_assist_tool_results_have_calls "
                    f"does not accept scope_key ({exc}). Ensure/repair must take "
                    "the session scope and pass it into tool-cache lookups so "
                    "the same tool_call_id stays isolated across sessions."
                )

        name_scopes = [scope for kind, scope in lookup_scopes if kind == "name"]
        args_scopes = [scope for kind, scope in lookup_scopes if kind == "args"]
        assert SESSION_A in name_scopes, lookup_scopes
        assert SESSION_B in name_scopes, lookup_scopes
        assert SESSION_A in args_scopes, lookup_scopes
        assert SESSION_B in args_scopes, lookup_scopes
        assert None not in name_scopes, (
            "ensure looked up tool names without scope_key; "
            f"scopes seen={name_scopes!r}"
        )
        assert None not in args_scopes, (
            "ensure looked up tool arguments without scope_key; "
            f"scopes seen={args_scopes!r}"
        )

        names_a = _assistant_tool_call_names(list(updated_a.get("messages") or []))
        names_b = _assistant_tool_call_names(list(updated_b.get("messages") or []))
        args_a = _assistant_tool_call_arguments(list(updated_a.get("messages") or []))
        args_b = _assistant_tool_call_arguments(list(updated_b.get("messages") or []))

        assert changes_a.get("google_adapter_codex_inserted_missing_tool_call_count") == 1
        assert changes_b.get("google_adapter_codex_inserted_missing_tool_call_count") == 1
        assert names_a == ["Read"], names_a
        assert names_b == ["Write"], names_b
        assert args_a == ['{"path":"a.txt"}'], args_a
        assert args_b == ['{"path":"b.txt","content":"x"}'], args_b
    finally:
        _clear_tool_call_caches()


@pytest.mark.asyncio
async def test_rr054_tool_cache_build_chain_isolates_two_sessions() -> None:
    """Full Google Code Assist build path must not cross-contaminate sessions.

    Two sessions restore the same tool_call_id under distinct scopes, then each
    runs a tool-result-only follow-up through the production builder. Multi-tool
    advertisement prevents single-tool fallback from masking missing scope
    propagation.
    """
    _clear_tool_call_caches()
    try:
        # Seed via helpers with explicit scopes (production restore is asserted
        # separately for scope_key acceptance/forwarding).
        lpe._remember_codex_google_code_assist_tool_call_name(
            SHARED_TOOL_CALL_ID,
            "Read",
            '{"path":"session-a.txt"}',
            scope_key=SESSION_A,
        )
        lpe._remember_codex_google_code_assist_tool_call_name(
            SHARED_TOOL_CALL_ID,
            "Write",
            '{"path":"session-b.txt","content":"payload-b"}',
            scope_key=SESSION_B,
        )

        async def _build_for_session(
            session_id: str,
        ) -> tuple[list[str], list[str], dict[str, Any]]:
            kwargs = _tool_result_only_kwargs(
                tool_call_id=SHARED_TOOL_CALL_ID,
                tool_content=f"result-for-{session_id}",
            )
            try:
                (
                    _wrapped,
                    _headers,
                    completion_messages,
                    _optional,
                    _litellm_params,
                    changes,
                ) = await lpe._build_google_code_assist_request_from_completion_kwargs(
                    completion_kwargs=kwargs,
                    adapter_model="claude-sonnet-4-6",
                    project="test-project",
                    request=_mock_request(session_id),
                    completion_kwargs_are_openai_chat=True,
                    scope_key=session_id,  # type: ignore[call-arg]
                )
            except TypeError as exc:
                # Builder may derive scope from request session_id instead of an
                # explicit kwarg. Fall back to request-only build, then assert
                # isolation on the resulting messages.
                if "scope_key" not in str(exc) and "unexpected keyword" not in str(exc):
                    raise
                (
                    _wrapped,
                    _headers,
                    completion_messages,
                    _optional,
                    _litellm_params,
                    changes,
                ) = await lpe._build_google_code_assist_request_from_completion_kwargs(
                    completion_kwargs=kwargs,
                    adapter_model="claude-sonnet-4-6",
                    project="test-project",
                    request=_mock_request(session_id),
                    completion_kwargs_are_openai_chat=True,
                )
            return (
                _assistant_tool_call_names(completion_messages),
                _assistant_tool_call_arguments(completion_messages),
                changes,
            )

        names_a, args_a, changes_a = await _build_for_session(SESSION_A)
        names_b, args_b, changes_b = await _build_for_session(SESSION_B)

        inserted_a = changes_a.get(
            "google_adapter_codex_inserted_missing_tool_call_count"
        )
        inserted_b = changes_b.get(
            "google_adapter_codex_inserted_missing_tool_call_count"
        )
        assert names_a == ["read_file"], (
            "build chain for session A must insert Read from scoped cache; "
            f"got names={names_a!r}, inserted_count={inserted_a!r}, "
            f"changes_keys={sorted(changes_a)}. Production ensure/lookup inside "
            "the builder must use the request session as scope_key for the same "
            "tool_call_id."
        )
        assert names_b == ["write_file"], (
            "build chain for session B must insert Write from scoped cache; "
            f"got names={names_b!r}, inserted_count={inserted_b!r}, "
            f"changes_keys={sorted(changes_b)}."
        )
        assert args_a == ['{"path":"session-a.txt"}'], args_a
        assert args_b == ['{"path":"session-b.txt","content":"payload-b"}'], args_b
        assert inserted_a == 1, changes_a
        assert inserted_b == 1, changes_b
    finally:
        _clear_tool_call_caches()


def test_rr054_tool_cache_function_call_args_lookup_is_scope_isolated() -> None:
    """Claude functionCall args helper must resolve arguments under scope_key."""
    _clear_tool_call_caches()
    try:
        lpe._remember_codex_google_code_assist_tool_call_name(
            SHARED_TOOL_CALL_ID,
            "Read",
            '{"path":"scoped-a"}',
            scope_key=SESSION_A,
        )
        lpe._remember_codex_google_code_assist_tool_call_name(
            SHARED_TOOL_CALL_ID,
            "Write",
            '{"path":"scoped-b","content":"y"}',
            scope_key=SESSION_B,
        )

        try:
            args_a = lpe._google_code_assist_function_call_args_for_id(
                SHARED_TOOL_CALL_ID,
                scope_key=SESSION_A,  # type: ignore[call-arg]
            )
            args_b = lpe._google_code_assist_function_call_args_for_id(
                SHARED_TOOL_CALL_ID,
                scope_key=SESSION_B,  # type: ignore[call-arg]
            )
        except TypeError as exc:
            pytest.fail(
                "RR-054 scope propagation gap: "
                "_google_code_assist_function_call_args_for_id does not accept "
                f"scope_key ({exc}). Args lookup used when inserting Claude "
                "functionCall pairs must be session-scoped for the same "
                "tool_call_id."
            )

        assert args_a == {"path": "scoped-a"}, args_a
        assert args_b == {"path": "scoped-b", "content": "y"}, args_b
    finally:
        _clear_tool_call_caches()


def test_rr054_tool_cache_unscoped_restore_overwrites_same_tool_call_id() -> None:
    """Unscoped production restore currently last-writer-wins (defect baseline).

    This is not the desired multi-session contract; it pins the unscoped path so
    the scoped restore/ensure/build tests remain the acceptance gate.
    """
    _clear_tool_call_caches()
    try:
        lpe._restore_google_adapter_tool_call_names(
            _response_with_tool_call(
                tool_call_id=SHARED_TOOL_CALL_ID,
                function_name="Read",
                function_arguments='{"path":"a"}',
            ),
            {},
        )
        lpe._restore_google_adapter_tool_call_names(
            _response_with_tool_call(
                tool_call_id=SHARED_TOOL_CALL_ID,
                function_name="Write",
                function_arguments='{"path":"b","content":"z"}',
            ),
            {},
        )
        assert (
            lpe._lookup_codex_google_code_assist_tool_call_name(SHARED_TOOL_CALL_ID)
            == "Write"
        )
        # Unscoped remember merges argument deltas for streaming fragments, so
        # consecutive full JSON payloads for the same id may concatenate rather
        # than replace. Name still last-writer-wins under a single unscoped key.
        unscoped_args = lpe._lookup_codex_google_code_assist_tool_call_arguments(
            SHARED_TOOL_CALL_ID
        )
        assert unscoped_args in (
            '{"path":"b","content":"z"}',
            '{"path":"a"}{"path":"b","content":"z"}',
        ), unscoped_args
        # Only one unscoped key may exist for this id.
        unscoped_keys = [
            key
            for key in lpe._codex_google_code_assist_tool_call_name_cache
            if key == SHARED_TOOL_CALL_ID or key.endswith(f":{SHARED_TOOL_CALL_ID}")
        ]
        assert SHARED_TOOL_CALL_ID in unscoped_keys
        assert all(
            not key.startswith(f"{SESSION_A}:") and not key.startswith(f"{SESSION_B}:")
            for key in unscoped_keys
        )
    finally:
        _clear_tool_call_caches()
