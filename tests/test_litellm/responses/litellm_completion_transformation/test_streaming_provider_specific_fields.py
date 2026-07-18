"""
Focused tests for RR-062: LiteLLMCompletionStreamingIterator accumulation of
provider_specific_fields (especially ordered code_interpreter_results) into the
terminal aggregated ModelResponse used by response.completed / RR-063.
"""

from unittest.mock import AsyncMock

from litellm.responses.litellm_completion_transformation.streaming_iterator import (
    LiteLLMCompletionStreamingIterator,
)
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices


def _make_iterator() -> LiteLLMCompletionStreamingIterator:
    return LiteLLMCompletionStreamingIterator(
        model="test-model",
        litellm_custom_stream_wrapper=AsyncMock(),
        request_input="run code",
        responses_api_request={},
    )


def _chunk_with_delta_psf(psf: dict, content: str = "", chunk_id: str = "c1") -> ModelResponseStream:
    return ModelResponseStream(
        id=chunk_id,
        created=123,
        model="test-model",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(
                    role="assistant",
                    content=content,
                    provider_specific_fields=psf,
                ),
            )
        ],
    )


def _code_result(item_id: str, code: str = "print(1)", container_id: str = "") -> dict:
    return {
        "type": "code_interpreter_call",
        "id": item_id,
        "code": code,
        "container_id": container_id,
        "status": "completed",
        "outputs": [{"type": "logs", "logs": "1\n"}],
    }


class TestMergeListProviderFieldValues:
    def test_cumulative_suffix_is_preferred(self):
        existing = [_code_result("a")]
        incoming = [_code_result("a"), _code_result("b")]
        merged = LiteLLMCompletionStreamingIterator._merge_list_provider_field_values(
            existing, incoming
        )
        assert [item["id"] for item in merged] == ["a", "b"]

    def test_incremental_append_preserves_order(self):
        existing = [_code_result("a")]
        incoming = [_code_result("b")]
        merged = LiteLLMCompletionStreamingIterator._merge_list_provider_field_values(
            existing, incoming
        )
        assert [item["id"] for item in merged] == ["a", "b"]

    def test_same_id_later_emission_enriches_without_duplicate(self):
        existing = [_code_result("srv_1", container_id="")]
        incoming = [_code_result("srv_1", container_id="ctr_final")]
        merged = LiteLLMCompletionStreamingIterator._merge_list_provider_field_values(
            existing, incoming
        )
        assert len(merged) == 1
        assert merged[0]["id"] == "srv_1"
        assert merged[0]["container_id"] == "ctr_final"

    def test_multi_result_order_preserved_with_id_enrichment(self):
        existing = [_code_result("a"), _code_result("b")]
        # Later chunk re-emits both with container ids and a new third result.
        incoming = [
            _code_result("a", container_id="c1"),
            _code_result("b", container_id="c1"),
            _code_result("c", container_id="c1"),
        ]
        merged = LiteLLMCompletionStreamingIterator._merge_list_provider_field_values(
            existing, incoming
        )
        assert [item["id"] for item in merged] == ["a", "b", "c"]
        assert all(item["container_id"] == "c1" for item in merged)


class TestAccumulateAndStampProviderSpecificFields:
    def test_accumulates_from_delta_and_stamps_hidden_params_and_message(self):
        iterator = _make_iterator()
        first = _code_result("srv_a", code="echo a")

        # Incremental emission of first result.
        chunk1 = _chunk_with_delta_psf(
            {"code_interpreter_results": [first], "tool_results": [{"type": "x"}]},
            content="hello",
            chunk_id="chunk-1",
        )
        # Later cumulative emission with both results + container enrichment.
        enriched_first = _code_result("srv_a", code="echo a", container_id="ctr_1")
        enriched_second = _code_result("srv_b", code="echo b", container_id="ctr_1")
        chunk2 = _chunk_with_delta_psf(
            {
                "code_interpreter_results": [enriched_first, enriched_second],
                "tool_results": [{"type": "x"}, {"type": "y"}],
            },
            content=" world",
            chunk_id="chunk-2",
        )

        iterator._accumulate_provider_specific_fields_from_chunk(chunk1)
        iterator._accumulate_provider_specific_fields_from_chunk(chunk2)
        iterator.collected_chat_completion_chunks = [
            iterator._snapshot_chunk_for_stream_chunk_builder(chunk1),
            iterator._snapshot_chunk_for_stream_chunk_builder(chunk2),
        ]

        response = iterator.create_litellm_model_response()
        assert response is not None

        hidden_psf = response._hidden_params.get("provider_specific_fields")
        assert isinstance(hidden_psf, dict)
        results = hidden_psf["code_interpreter_results"]
        assert [item["id"] for item in results] == ["srv_a", "srv_b"]
        assert results[0]["container_id"] == "ctr_1"
        assert results[1]["container_id"] == "ctr_1"

        message = response.choices[0].message
        msg_psf = getattr(message, "provider_specific_fields", None)
        assert isinstance(msg_psf, dict)
        assert [item["id"] for item in msg_psf["code_interpreter_results"]] == [
            "srv_a",
            "srv_b",
        ]

        # Copies are independent of the iterator accumulator.
        results[0]["code"] = "mutated"
        assert (
            iterator._accumulated_provider_specific_fields["code_interpreter_results"][
                0
            ]["code"]
            == "echo a"
        )

    def test_duplicate_chunk_surface_does_not_double_count(self):
        """Fields present on both chunk and delta surfaces should merge once."""
        iterator = _make_iterator()
        result = _code_result("srv_once")
        psf = {"code_interpreter_results": [result]}
        chunk = _chunk_with_delta_psf(psf)
        # Also set chunk-level provider_specific_fields with the same payload.
        chunk.provider_specific_fields = {
            "code_interpreter_results": [result],
        }
        iterator._accumulate_provider_specific_fields_from_chunk(chunk)
        acc = iterator._accumulated_provider_specific_fields["code_interpreter_results"]
        assert len(acc) == 1
        assert acc[0]["id"] == "srv_once"

    def test_terminal_completed_response_receives_complete_ordered_results(self):
        """
        End-to-end through create_litellm_model_response: RR-063 reconstruction
        must be able to read ordered code_interpreter_results from either the
        message or _hidden_params surface produced by RR-062.
        """
        iterator = _make_iterator()
        # Simulate Anthropic-style cumulative re-emission across chunks.
        r1 = _code_result("call_1", code="ls")
        r1_with_ctr = _code_result("call_1", code="ls", container_id="ctr_abc")
        r2 = _code_result("call_2", code="pwd", container_id="ctr_abc")

        chunks = [
            _chunk_with_delta_psf(
                {"code_interpreter_results": [r1]}, content="step1 ", chunk_id="1"
            ),
            _chunk_with_delta_psf(
                {"code_interpreter_results": [r1_with_ctr, r2]},
                content="step2",
                chunk_id="2",
            ),
        ]
        for chunk in chunks:
            iterator._accumulate_provider_specific_fields_from_chunk(chunk)
            iterator.collected_chat_completion_chunks.append(
                iterator._snapshot_chunk_for_stream_chunk_builder(chunk)
            )

        response = iterator.create_litellm_model_response()
        assert response is not None

        # Prefer message surface, fall back to hidden params (RR-063 order).
        msg_psf = getattr(response.choices[0].message, "provider_specific_fields", None)
        hidden_psf = response._hidden_params.get("provider_specific_fields")
        results = None
        if isinstance(msg_psf, dict):
            results = msg_psf.get("code_interpreter_results")
        if not results and isinstance(hidden_psf, dict):
            results = hidden_psf.get("code_interpreter_results")

        assert results is not None
        assert [item["id"] for item in results] == ["call_1", "call_2"]
        assert results[0]["container_id"] == "ctr_abc"
        assert results[1]["code"] == "pwd"
