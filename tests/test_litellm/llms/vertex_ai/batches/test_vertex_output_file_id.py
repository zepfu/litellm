"""RR-039: Vertex batch output path join must not double-slash."""

from __future__ import annotations

from litellm.llms.vertex_ai.batches.transformation import VertexAIBatchTransformation


def test_gcs_output_directory_trailing_slash_normalized() -> None:
    response = {
        "outputInfo": {
            "gcsOutputDirectory": "gs://bucket/path/",
        }
    }
    out = VertexAIBatchTransformation._get_output_file_id_from_vertex_ai_batch_response(
        response
    )
    assert out == "gs://bucket/path/predictions.jsonl"
    assert "//predictions" not in out


def test_gcs_output_directory_without_trailing_slash() -> None:
    response = {
        "outputInfo": {
            "gcsOutputDirectory": "gs://bucket/path",
        }
    }
    out = VertexAIBatchTransformation._get_output_file_id_from_vertex_ai_batch_response(
        response
    )
    assert out == "gs://bucket/path/predictions.jsonl"
