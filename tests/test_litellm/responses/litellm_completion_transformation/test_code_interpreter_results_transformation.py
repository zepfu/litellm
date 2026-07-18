"""
Focused tests for RR-063: convert provider_specific_fields["code_interpreter_results"]
into OpenAI Responses code_interpreter_call output items.
"""

from typing import Any, Dict, List, Optional

from openai.types.responses.response_code_interpreter_tool_call import (
    ResponseCodeInterpreterToolCall,
)

from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)
from litellm.types.responses.main import GenericResponseOutputItem
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Choices,
    Function,
    Message,
    ModelResponse,
    Usage,
)


def _code_interpreter_result(
    item_id: str = "srvtoolu_01ABC",
    code: str = "echo hello",
    container_id: str = "container_123",
    status: str = "completed",
    logs: str = "hello\n",
) -> Dict[str, Any]:
    return {
        "type": "code_interpreter_call",
        "id": item_id,
        "code": code,
        "container_id": container_id,
        "status": status,
        "outputs": [{"type": "logs", "logs": logs}],
    }


def _build_completion_response(
    *,
    content: Optional[str] = "done",
    tool_call_ids: Optional[List[str]] = None,
    message_code_results: Any = None,
    hidden_code_results: Any = None,
    extra_message_psf: Optional[Dict[str, Any]] = None,
    extra_hidden_psf: Optional[Dict[str, Any]] = None,
    finish_reason: str = "tool_calls",
) -> ModelResponse:
    tool_call_ids = tool_call_ids or []
    tool_calls = [
        ChatCompletionMessageToolCall(
            id=tool_id,
            type="function",
            function=Function(
                name="bash_code_execution",
                arguments='{"command":"echo hello"}',
            ),
        )
        for tool_id in tool_call_ids
    ]

    message_psf: Dict[str, Any] = {}
    if message_code_results is not None:
        message_psf["code_interpreter_results"] = message_code_results
    if extra_message_psf:
        message_psf.update(extra_message_psf)

    message_kwargs: Dict[str, Any] = {
        "role": "assistant",
        "content": content,
        "tool_calls": tool_calls or None,
    }
    if message_psf:
        message_kwargs["provider_specific_fields"] = message_psf

    response = ModelResponse(
        id="chatcmpl-rr063",
        created=1,
        model="claude-test",
        object="chat.completion",
        choices=[
            Choices(
                finish_reason=finish_reason,
                index=0,
                message=Message(**message_kwargs),
            )
        ],
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )

    hidden_psf: Dict[str, Any] = {}
    if hidden_code_results is not None:
        hidden_psf["code_interpreter_results"] = hidden_code_results
    if extra_hidden_psf:
        hidden_psf.update(extra_hidden_psf)
    if hidden_psf:
        response._hidden_params = {"provider_specific_fields": hidden_psf}
    else:
        response._hidden_params = {}

    return response


def _transform(response: ModelResponse):
    return LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
        request_input="run code",
        responses_api_request={},
        chat_completion_response=response,
    )


def _code_items(output) -> List[Any]:
    return [item for item in output if getattr(item, "type", None) == "code_interpreter_call"]


class TestCodeInterpreterResultsTransformation:
    def test_single_valid_result_replaces_function_call(self):
        result = _code_interpreter_result()
        response = _build_completion_response(
            tool_call_ids=["srvtoolu_01ABC"],
            message_code_results=[result],
            extra_message_psf={"other": "keep-me"},
            extra_hidden_psf={"other": "keep-me"},
            hidden_code_results=[result],
        )

        transformed = _transform(response)
        code_items = _code_items(transformed.output)

        assert len(code_items) == 1
        item = code_items[0]
        assert isinstance(item, ResponseCodeInterpreterToolCall)
        assert item.id == "srvtoolu_01ABC"
        assert item.code == "echo hello"
        assert item.container_id == "container_123"
        assert item.status == "completed"
        assert item.outputs is not None
        assert len(item.outputs) == 1
        assert item.outputs[0].type == "logs"
        assert item.outputs[0].logs == "hello\n"

        # Regression: normal message remains LiteLLM GenericResponseOutputItem
        # while code interpreter uses OpenAI ResponseCodeInterpreterToolCall.
        message_items = [
            out_item
            for out_item in transformed.output
            if getattr(out_item, "type", None) == "message"
        ]
        assert len(message_items) == 1
        assert isinstance(message_items[0], GenericResponseOutputItem)
        assert not isinstance(message_items[0], ResponseCodeInterpreterToolCall)
        assert type(message_items[0]).__name__ == "GenericResponseOutputItem"
        assert type(item).__name__ == "ResponseCodeInterpreterToolCall"

        # Matching function_call placeholder is replaced, not duplicated.
        function_calls = [
            item
            for item in transformed.output
            if getattr(item, "type", None) == "function_call"
        ]
        assert function_calls == []

        # Generic provider_specific_fields passthrough remains intact.
        psf = getattr(transformed, "provider_specific_fields", None)
        assert isinstance(psf, dict)
        assert psf.get("other") == "keep-me"
        assert "code_interpreter_results" in psf

    def test_multiple_results_preserve_order(self):
        results = [
            _code_interpreter_result(
                item_id="srv_A",
                code="echo first",
                logs="first\n",
            ),
            _code_interpreter_result(
                item_id="srv_B",
                code="echo second",
                logs="second\n",
            ),
        ]
        response = _build_completion_response(
            content="after tools",
            tool_call_ids=["srv_A", "srv_B"],
            message_code_results=results,
        )

        transformed = _transform(response)
        types = [getattr(item, "type", None) for item in transformed.output]
        ids = [
            getattr(item, "id", None)
            for item in transformed.output
            if getattr(item, "type", None) == "code_interpreter_call"
        ]

        # Message first, then code items in the original tool-call order.
        assert types[0] == "message"
        assert ids == ["srv_A", "srv_B"]

        code_items = _code_items(transformed.output)
        assert code_items[0].outputs[0].logs == "first\n"
        assert code_items[1].outputs[0].logs == "second\n"

    def test_hidden_params_only_streaming_aggregate_path(self):
        """RR-062 final aggregate may only put fields on _hidden_params."""
        result = _code_interpreter_result(item_id="srv_hidden")
        response = _build_completion_response(
            tool_call_ids=["srv_hidden"],
            message_code_results=None,
            hidden_code_results=[result],
            extra_hidden_psf={"streamed": True},
        )

        transformed = _transform(response)
        code_items = _code_items(transformed.output)

        assert len(code_items) == 1
        assert code_items[0].id == "srv_hidden"
        assert code_items[0].code == "echo hello"
        psf = getattr(transformed, "provider_specific_fields", None)
        assert isinstance(psf, dict)
        assert psf.get("streamed") is True

    def test_malformed_and_absent_results_are_ignored_safely(self):
        # Absent: no conversion, function_call remains.
        response_absent = _build_completion_response(
            tool_call_ids=["srv_keep"],
            message_code_results=None,
            hidden_code_results=None,
        )
        transformed_absent = _transform(response_absent)
        assert _code_items(transformed_absent.output) == []
        assert any(
            getattr(item, "type", None) == "function_call"
            for item in transformed_absent.output
        )

        # Malformed entries are skipped; valid siblings still convert.
        mixed = [
            None,
            "not-a-result",
            {"type": "code_interpreter_call", "status": "completed"},  # missing id
            {
                "type": "code_interpreter_call",
                "id": "srv_good",
                "code": "echo ok",
                "container_id": "ctr",
                "status": "weird-status",  # normalized to completed
                "outputs": [{"type": "logs", "logs": "ok\n"}],
            },
            {"id": "srv_bad_outputs", "status": "completed", "outputs": "nope"},
        ]
        response_mixed = _build_completion_response(
            tool_call_ids=["srv_good", "srv_other"],
            message_code_results=mixed,
        )
        transformed_mixed = _transform(response_mixed)
        code_items = _code_items(transformed_mixed.output)

        assert len(code_items) == 1
        assert code_items[0].id == "srv_good"
        assert code_items[0].status == "completed"
        assert code_items[0].outputs[0].logs == "ok\n"

        # Unmatched function_call for srv_other remains.
        remaining_fn_ids = {
            getattr(item, "call_id", None) or getattr(item, "id", None)
            for item in transformed_mixed.output
            if getattr(item, "type", None) == "function_call"
        }
        assert "srv_other" in remaining_fn_ids
        assert "srv_good" not in remaining_fn_ids

    def test_no_duplicates_when_conversion_runs_twice_or_item_already_present(self):
        result = _code_interpreter_result(item_id="srv_dup")
        response = _build_completion_response(
            tool_call_ids=["srv_dup"],
            message_code_results=[result],
            hidden_code_results=[result],  # same id in both sources
        )

        # First conversion via public transform.
        transformed = _transform(response)
        first_code = _code_items(transformed.output)
        assert len(first_code) == 1

        # Re-merge the same extracted items into already-converted output.
        extracted = LiteLLMCompletionResponsesConfig._extract_tool_result_output_items(
            response
        )
        assert len(extracted) == 1  # de-duped across message + hidden sources

        rematched = LiteLLMCompletionResponsesConfig._merge_code_interpreter_output_items(
            responses_output=list(transformed.output),
            tool_result_items=extracted,
        )
        rematched_again = (
            LiteLLMCompletionResponsesConfig._merge_code_interpreter_output_items(
                responses_output=rematched,
                tool_result_items=extracted,
            )
        )
        code_ids = [
            getattr(item, "id", None)
            for item in rematched_again
            if getattr(item, "type", None) == "code_interpreter_call"
        ]
        assert code_ids == ["srv_dup"]

    def test_generic_provider_specific_fields_passthrough_preserved(self):
        result = _code_interpreter_result()
        response = _build_completion_response(
            tool_call_ids=["srvtoolu_01ABC"],
            message_code_results=[result],
            hidden_code_results=[result],
            extra_hidden_psf={
                "tool_results": [{"tool_use_id": "srvtoolu_01ABC"}],
                "container": {"id": "container_123"},
                "custom_marker": 42,
            },
        )

        transformed = _transform(response)
        psf = getattr(transformed, "provider_specific_fields", None)
        assert isinstance(psf, dict)
        assert psf["custom_marker"] == 42
        assert psf["container"] == {"id": "container_123"}
        assert psf["tool_results"] == [{"tool_use_id": "srvtoolu_01ABC"}]
        assert "code_interpreter_results" in psf

        # Conversion still happened independently of passthrough.
        assert len(_code_items(transformed.output)) == 1

    def test_append_when_no_function_call_placeholder_exists(self):
        result = _code_interpreter_result(item_id="srv_appended")
        response = _build_completion_response(
            content="text only",
            tool_call_ids=[],  # no function_call placeholder
            message_code_results=[result],
            finish_reason="stop",
        )

        transformed = _transform(response)
        types = [getattr(item, "type", None) for item in transformed.output]
        assert types[0] == "message"
        assert types[-1] == "code_interpreter_call"
        assert _code_items(transformed.output)[0].id == "srv_appended"


    def test_mixed_output_preserves_generic_message_class(self):
        """Regression: do not coerce normal messages into OpenAI-native classes."""
        result = _code_interpreter_result()
        response = _build_completion_response(
            content="hello world",
            tool_call_ids=["srvtoolu_01ABC"],
            message_code_results=[result],
        )
        transformed = _transform(response)

        message_items = [
            item for item in transformed.output if getattr(item, "type", None) == "message"
        ]
        code_items = _code_items(transformed.output)

        assert len(message_items) == 1
        assert isinstance(message_items[0], GenericResponseOutputItem)
        assert type(message_items[0]).__name__ == "GenericResponseOutputItem"
        assert message_items[0].content[0].text == "hello world"

        assert len(code_items) == 1
        assert isinstance(code_items[0], ResponseCodeInterpreterToolCall)
        assert type(code_items[0]).__name__ == "ResponseCodeInterpreterToolCall"

    def test_normalize_accepts_object_and_dict_shapes(self):
        as_dict = _code_interpreter_result(item_id="from_dict")
        as_obj = ResponseCodeInterpreterToolCall(**as_dict)

        assert (
            LiteLLMCompletionResponsesConfig._normalize_code_interpreter_output_item(
                as_dict
            ).id
            == "from_dict"
        )
        assert (
            LiteLLMCompletionResponsesConfig._normalize_code_interpreter_output_item(
                as_obj
            )
            is as_obj
        )
        assert (
            LiteLLMCompletionResponsesConfig._normalize_code_interpreter_output_item(
                None
            )
            is None
        )
        assert (
            LiteLLMCompletionResponsesConfig._normalize_code_interpreter_output_item(
                {"status": "completed"}
            )
            is None
        )
