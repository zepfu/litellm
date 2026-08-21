"""RR-095: Cursor Agent advertised OpenAI params must reach the wire or reject.

AgentRunRequest (CURSOR-001) has conversation_state, action.user_message_action,
mcp_tools, and custom_system_prompt. It does not have OpenAI temperature,
max_tokens, or tool_choice as top-level fields. LiteLLM must not spawn the
Cursor CLI at request time.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple, Union
from unittest.mock import patch

import pytest

from litellm.exceptions import UnsupportedParamsError
from litellm.llms.cursor_agent.chat.transformation import CursorAgentConfig
from litellm.llms.cursor_agent.common_utils import CursorAgentError, build_run_request

MODEL = "composer-2.5"
FULL_MODEL = "cursor_agent/composer-2.5"

# Verified AgentRunRequest field names from CURSOR-001. OpenAI sampling /
# tool_choice keys are not among them and must not be invented on the wire.
VERIFIED_AGENT_RUN_REQUEST_FIELDS = {
    "conversation_state",
    "action",
    "model_details",
    "mcp_tools",
    "conversation_id",
    "mcp_file_system_options",
    "skill_options",
    "custom_system_prompt",
    "requested_model",
    "suggest_next_prompt",
    "subagent_type_name",
    "exclude_workspace_context",
    "harness",
    "selected_subagent_models",
    "selected_subagent_model_details",
    "conversation_group_id",
    "pre_fetched_blobs",
    "dev_raw_model_slug",
    "client_supports_inline_images",
    "subagent_model_overrides",
    "can_create_cloud_subagents",
    "suppress_subagent_progress_update_tool",
    "client_supports_send_to_user",
    "computer_use_coordinate_mode",
    "run_id",
    "agent_session_id",
    "client_supports_prompt_context_usage_rpc",
    "client_supports_routed_model_update",
}

OPENAI_ONLY_KEYS = {"temperature", "max_tokens", "tool_choice", "tools"}

LOOKUP_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_ticket",
        "description": "Look up a ticket by id",
        "parameters": {
            "type": "object",
            "properties": {"ticket_id": {"type": "string"}},
            "required": ["ticket_id"],
        },
    },
}

EARLIER_USER = "remember the project marker is RR-095-EARLIER-TURN"
ASSISTANT_PRIOR = "acknowledged RR-095 prior context ASSISTANT-PRIOR-TURN"
LAST_USER = "LAST-USER-TURN what is the project marker?"
SYSTEM_PROMPT = "SYSTEM-PROMPT-RR095 stay in the named worktree"

_REJECT_TYPES = (CursorAgentError, UnsupportedParamsError, ValueError, TypeError)


def _config() -> CursorAgentConfig:
    return CursorAgentConfig()


def _transform_or_reject(
    *,
    messages: List[Dict[str, Any]],
    optional_params: Dict[str, Any],
    model: str = MODEL,
) -> Tuple[str, Union[Dict[str, Any], BaseException]]:
    try:
        payload = _config().transform_request(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            optional_params=dict(optional_params),
            litellm_params={},
            headers={},
        )
    except _REJECT_TYPES as exc:
        return "reject", exc
    return "ok", payload


def _run_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    if "run_request" in payload and isinstance(payload["run_request"], dict):
        return payload["run_request"]
    return payload


def _dump(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, default=str)


def _contains(payload: Any, needle: str) -> bool:
    return needle in _dump(payload)


def _top_level_keys(run: Dict[str, Any]) -> set:
    return set(run.keys())


def _assert_no_invented_openai_fields(payload: Dict[str, Any]) -> None:
    run = _run_request(payload)
    leaked = _top_level_keys(run) & OPENAI_ONLY_KEYS
    assert not leaked, (
        "AgentRunRequest must not invent OpenAI-only top-level fields "
        f"{sorted(leaked)}; serialize to verified proto fields or reject"
    )
    unknown = _top_level_keys(run) - VERIFIED_AGENT_RUN_REQUEST_FIELDS
    # Nested OpenAI leftovers under conversation_state/action are allowed
    # only as serialized history, not as sibling proto fields.
    invented = unknown & OPENAI_ONLY_KEYS
    assert not invented


def _param_is_declared(param: str, value: Any) -> bool:
    config = _config()
    supported = config.get_supported_openai_params(MODEL)
    try:
        mapped = config.map_openai_params(
            non_default_params={param: value},
            optional_params={},
            model=MODEL,
            drop_params=False,
        )
    except _REJECT_TYPES:
        # Explicit map-time rejection is a handled declaration, not a silent drop.
        return True
    return param in supported or param in mapped


@pytest.mark.parametrize(
    "param,value",
    [
        ("temperature", 0.2),
        ("max_tokens", 128),
        ("tool_choice", "auto"),
        ("tool_choice", {"type": "function", "function": {"name": "lookup_ticket"}}),
    ],
)
def test_declared_openai_sampling_and_tool_choice_reject_or_verified_field(
    param: str, value: Any
):
    """temperature / max_tokens / tool_choice are not AgentRunRequest fields.

    Declaring them via get_supported_openai_params or map_openai_params
    requires an explicit pre-egress rejection. Silent drop is the RR-095 bug.
    Copying them onto the Connect body as OpenAI keys invents proto fields.
    """
    if not _param_is_declared(param, value):
        pytest.skip(f"{param} is no longer advertised or mapped")

    messages = [{"role": "user", "content": LAST_USER}]
    optional_params = {param: value}
    if param == "tool_choice":
        optional_params["tools"] = [LOOKUP_TOOL]

    status, result = _transform_or_reject(
        messages=messages, optional_params=optional_params
    )
    if status == "reject":
        assert isinstance(result, _REJECT_TYPES)
        return

    assert isinstance(result, dict)
    _assert_no_invented_openai_fields(result)
    pytest.fail(
        f"{param} is advertised/mapped but transform_request silently omitted it "
        "from AgentRunRequest (no verified proto field exists; reject instead)"
    )


def test_declared_tools_reach_mcp_tools_or_are_rejected():
    """Ingress-advertised tools serialize to mcp_tools, or the call is rejected."""
    messages = [{"role": "user", "content": LAST_USER}]
    optional_params = {"tools": [LOOKUP_TOOL], "tool_choice": "auto"}
    status, result = _transform_or_reject(
        messages=messages, optional_params=optional_params
    )
    if status == "reject":
        assert isinstance(result, _REJECT_TYPES)
        message = str(result)
        assert "tool" in message.lower() or "mcp" in message.lower() or message
        return

    assert isinstance(result, dict)
    _assert_no_invented_openai_fields(result)
    run = _run_request(result)
    assert "mcp_tools" in run, (
        "tools was advertised and the request was not rejected, but mcp_tools "
        "is missing from AgentRunRequest"
    )
    assert "lookup_ticket" in _dump(run["mcp_tools"]), (
        "mcp_tools must include the ingress-advertised tool name lookup_ticket"
    )
    assert "tools" not in run


def test_system_prompt_reaches_custom_system_prompt_or_is_rejected():
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": LAST_USER},
    ]
    status, result = _transform_or_reject(messages=messages, optional_params={})
    if status == "reject":
        assert isinstance(result, _REJECT_TYPES)
        return

    assert isinstance(result, dict)
    run = _run_request(result)
    dumped = _dump(run)
    assert SYSTEM_PROMPT in dumped, (
        "system content was dropped; serialize to custom_system_prompt "
        "(or conversation_state) or reject the request"
    )
    assert run.get("custom_system_prompt") == SYSTEM_PROMPT or SYSTEM_PROMPT in _dump(
        run.get("conversation_state")
    )
    action_text = (
        run.get("action", {})
        .get("user_message_action", {})
        .get("user_message", {})
        .get("text")
    )
    assert action_text != SYSTEM_PROMPT or LAST_USER in dumped


def test_multi_turn_history_is_not_silently_reduced_to_last_user_message():
    """extract_user_text currently keeps only the last user message.

    That is not an allowed silent reduction. Either conversation_state (or
    equivalent verified history) retains earlier turns, or the provider
    rejects multi-turn requests before egress.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": EARLIER_USER},
        {"role": "assistant", "content": ASSISTANT_PRIOR},
        {"role": "user", "content": LAST_USER},
    ]
    status, result = _transform_or_reject(messages=messages, optional_params={})
    if status == "reject":
        assert isinstance(result, _REJECT_TYPES)
        message = str(result).lower()
        assert any(
            token in message
            for token in ("multi", "histor", "turn", "conversation", "message")
        ) or message
        return

    assert isinstance(result, dict)
    run = _run_request(result)
    dumped = _dump(run)
    action_text = (
        run.get("action", {})
        .get("user_message_action", {})
        .get("user_message", {})
        .get("text")
    )
    assert action_text == LAST_USER or LAST_USER in dumped
    assert EARLIER_USER in dumped, (
        "earlier user turn was silently dropped; put it in conversation_state "
        "or reject multi-turn requests"
    )
    assert ASSISTANT_PRIOR in dumped, (
        "prior assistant turn was silently dropped; put it in conversation_state "
        "or reject multi-turn requests"
    )
    conversation_state = run.get("conversation_state")
    assert conversation_state not in ({}, None) or EARLIER_USER in dumped


def test_tool_result_continuation_is_not_reduced_to_last_user_message():
    messages = [
        {"role": "user", "content": "open ticket T-095"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_rr095",
                    "type": "function",
                    "function": {
                        "name": "lookup_ticket",
                        "arguments": '{"ticket_id":"T-095"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_rr095",
            "content": "TOOL-RESULT-RR095 status=open",
        },
        {"role": "user", "content": LAST_USER},
    ]
    optional_params = {"tools": [LOOKUP_TOOL]}
    status, result = _transform_or_reject(
        messages=messages, optional_params=optional_params
    )
    if status == "reject":
        assert isinstance(result, _REJECT_TYPES)
        return

    assert isinstance(result, dict)
    dumped = _dump(result)
    assert "TOOL-RESULT-RR095" in dumped, (
        "tool result continuation was silently dropped; serialize via "
        "conversation_state / interrupted_pending_tool_call_resolutions "
        "or reject tool-turn requests"
    )
    assert "lookup_ticket" in dumped
    run = _run_request(result)
    action_text = (
        run.get("action", {})
        .get("user_message_action", {})
        .get("user_message", {})
        .get("text")
    )
    assert action_text != "TOOL-RESULT-RR095 status=open" or LAST_USER in dumped


def test_map_openai_params_copies_declared_params_into_optional_params():
    """Regression: map_openai_params currently copies the advertised set.

    If mapping keeps copying a param, transform_request is responsible for
    serializing or rejecting it. This test fails if mapping advertises a
    param that transform then drops.
    """
    config = _config()
    try:
        mapped = config.map_openai_params(
            non_default_params={
                "tools": [LOOKUP_TOOL],
                "tool_choice": "auto",
                "max_tokens": 64,
                "temperature": 0.5,
            },
            optional_params={},
            model=MODEL,
            drop_params=False,
        )
    except _REJECT_TYPES:
        return
    declared = [
        name
        for name in ("tools", "tool_choice", "max_tokens", "temperature")
        if name in mapped
    ]
    if not declared:
        pytest.skip("map_openai_params no longer copies the RR-095 param set")

    status, result = _transform_or_reject(
        messages=[{"role": "user", "content": LAST_USER}],
        optional_params=mapped,
    )
    if status == "reject":
        return

    assert isinstance(result, dict)
    run = _run_request(result)
    dumped = _dump(run)
    missing = []
    if "tools" in declared and "lookup_ticket" not in dumped:
        missing.append("tools->mcp_tools")
    if "tool_choice" in declared and "tool_choice" not in dumped and "mcp_tools" not in run:
        missing.append("tool_choice")
    if "max_tokens" in declared and "max_tokens" not in dumped:
        missing.append("max_tokens")
    if "temperature" in declared and "temperature" not in dumped:
        missing.append("temperature")
    # Sampling params have no verified field: presence on the dump would be
    # invented proto. Either path is a failure unless transform rejected.
    invented = _top_level_keys(run) & OPENAI_ONLY_KEYS
    if invented:
        pytest.fail(f"invented OpenAI proto fields on AgentRunRequest: {sorted(invented)}")
    assert not missing, (
        "map_openai_params copied "
        f"{declared} but transform_request dropped {missing} instead of "
        "serializing verified AgentRunRequest fields or rejecting"
    )


def test_build_run_request_does_not_exec_cursor_cli():
    with patch("subprocess.Popen") as mock_popen, patch(
        "subprocess.run"
    ) as mock_run, patch("os.execv") as mock_execv, patch(
        "os.system"
    ) as mock_system:
        _transform_or_reject(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": EARLIER_USER},
                {"role": "assistant", "content": ASSISTANT_PRIOR},
                {"role": "user", "content": LAST_USER},
            ],
            optional_params={
                "tools": [LOOKUP_TOOL],
                "tool_choice": "auto",
                "max_tokens": 32,
                "temperature": 0.1,
            },
        )
        try:
            build_run_request(
                model=FULL_MODEL,
                messages=[{"role": "user", "content": LAST_USER}],
                optional_params={"tools": [LOOKUP_TOOL], "temperature": 0.1},
            )
        except _REJECT_TYPES:
            pass
        mock_popen.assert_not_called()
        mock_run.assert_not_called()
        mock_execv.assert_not_called()
        mock_system.assert_not_called()
