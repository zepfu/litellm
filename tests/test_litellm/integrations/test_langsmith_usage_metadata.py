"""
RR-014: Langsmith outputs must include usage_metadata for Cost column.
"""

from unittest.mock import MagicMock

from litellm.integrations.langsmith import LangsmithLogger


def test_prepare_log_data_includes_usage_metadata():
    logger = LangsmithLogger.__new__(LangsmithLogger)
    logger.langsmith_default_run_name = "LLMRun"

    payload = {
        "response": {
            "id": "chatcmpl-test",
            "choices": [{"message": {"role": "assistant", "content": "hi"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
        "metadata": {},
        "startTime": "2024-01-01T00:00:00Z",
        "endTime": "2024-01-01T00:00:01Z",
        "request_tags": [],
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "response_cost": 0.0012,
        "error_str": None,
        "status": "success",
    }
    credentials = {
        "LANGSMITH_PROJECT": "test-project",
        "LANGSMITH_API_KEY": "fake",
        "LANGSMITH_BASE_URL": "https://api.smith.langchain.com",
        "LANGSMITH_TENANT_ID": None,
    }
    data = logger._prepare_log_data(
        kwargs={"standard_logging_object": payload, "litellm_params": {}},
        response_obj=None,
        start_time=None,
        end_time=None,
        credentials=credentials,
    )
    usage_metadata = data["outputs"]["usage_metadata"]
    assert usage_metadata["input_tokens"] == 10
    assert usage_metadata["output_tokens"] == 5
    assert usage_metadata["total_tokens"] == 15
    assert usage_metadata["total_cost"] == 0.0012
    # Original response fields preserved
    assert data["outputs"]["id"] == "chatcmpl-test"
