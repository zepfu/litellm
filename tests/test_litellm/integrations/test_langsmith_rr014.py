"""
RR-014: Langsmith helper restoration, usage_metadata, nested id guards.
"""

from unittest.mock import patch

from litellm.integrations.langsmith import LangsmithLogger


def _make_logger() -> LangsmithLogger:
    logger = LangsmithLogger.__new__(LangsmithLogger)
    logger.langsmith_default_run_name = "LLMRun"
    return logger


def _credentials():
    return {
        "LANGSMITH_PROJECT": "test-project",
        "LANGSMITH_API_KEY": "fake",
        "LANGSMITH_BASE_URL": "https://api.smith.langchain.com",
        "LANGSMITH_TENANT_ID": None,
    }


def _base_payload(**overrides):
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
    payload.update(overrides)
    return payload


def test_helpers_exist_on_langsmith_logger():
    logger = _make_logger()
    assert callable(logger._extract_metadata_fields)
    assert callable(logger._build_extra_metadata)
    assert callable(logger._build_outputs_with_usage)
    assert callable(logger._ensure_required_ids)


def test_prepare_log_data_includes_usage_metadata():
    logger = _make_logger()
    data = logger._prepare_log_data(
        kwargs={"standard_logging_object": _base_payload(), "litellm_params": {}},
        response_obj=None,
        start_time=None,
        end_time=None,
        credentials=_credentials(),
    )
    usage_metadata = data["outputs"]["usage_metadata"]
    assert usage_metadata["input_tokens"] == 10
    assert usage_metadata["output_tokens"] == 5
    assert usage_metadata["total_tokens"] == 15
    assert usage_metadata["total_cost"] == 0.0012
    assert data["outputs"]["id"] == "chatcmpl-test"


def test_build_outputs_with_usage_non_dict_response():
    logger = _make_logger()
    payload = _base_payload(response="plain-text-response")
    outputs = logger._build_outputs_with_usage(payload)
    assert outputs["output"] == "plain-text-response"
    assert outputs["usage_metadata"]["input_tokens"] == 10
    assert outputs["usage_metadata"]["total_cost"] == 0.0012


def test_build_extra_metadata_hoists_requester_thread_keys():
    logger = _make_logger()
    metadata = {
        "model": "gpt-4o",
        "requester_metadata": {
            "session_id": "sess-1",
            "thread_id": "thr-1",
            "conversation_id": "conv-1",
            "ignored": "x",
        },
    }
    extra = logger._build_extra_metadata(metadata)
    assert extra["session_id"] == "sess-1"
    assert extra["thread_id"] == "thr-1"
    assert extra["conversation_id"] == "conv-1"
    # Does not overwrite existing top-level keys.
    metadata2 = {
        "session_id": "keep-me",
        "requester_metadata": {"session_id": "overwrite-me"},
    }
    extra2 = logger._build_extra_metadata(metadata2)
    assert extra2["session_id"] == "keep-me"


def test_ensure_required_ids_nested_guard_skips_non_str_run_id():
    logger = _make_logger()
    data = {}
    # run_id is non-str; nested guard must not assign trace_id / dotted_order.
    logger._ensure_required_ids(data, run_id=12345)  # type: ignore[arg-type]
    # id was missing so helper generated a uuid str and used that for fallthrough
    # When id is missing, run_id is reassigned to a uuid str first.
    assert isinstance(data["id"], str)
    assert data["trace_id"] == data["id"]
    assert isinstance(data["dotted_order"], str)

    data2 = {"id": "existing-id"}
    logger._ensure_required_ids(data2, run_id=None)
    assert data2["id"] == "existing-id"
    # nested guard: None run_id must not set trace/dotted from invalid value
    assert "trace_id" not in data2
    assert "dotted_order" not in data2

    data3 = {"id": "existing-id"}
    logger._ensure_required_ids(data3, run_id=999)  # type: ignore[arg-type]
    assert data3["id"] == "existing-id"
    assert "trace_id" not in data3
    assert "dotted_order" not in data3


def test_ensure_required_ids_sets_trace_and_dotted_from_valid_run_id():
    logger = _make_logger()
    data = {"id": "run-abc"}
    with patch.object(logger, "make_dot_order", return_value="dot-run-abc") as mock_dot:
        logger._ensure_required_ids(data, run_id="run-abc")
    assert data["trace_id"] == "run-abc"
    assert data["dotted_order"] == "dot-run-abc"
    mock_dot.assert_called_once_with(run_id="run-abc")


def test_ensure_required_ids_preserves_existing_trace_and_dotted():
    logger = _make_logger()
    data = {
        "id": "run-abc",
        "trace_id": "trace-keep",
        "dotted_order": "dot-keep",
    }
    with patch.object(logger, "make_dot_order") as mock_dot:
        logger._ensure_required_ids(data, run_id="run-abc")
    assert data["trace_id"] == "trace-keep"
    assert data["dotted_order"] == "dot-keep"
    mock_dot.assert_not_called()


def test_prepare_log_data_generates_required_ids_when_missing():
    logger = _make_logger()
    with patch.object(logger, "make_dot_order", return_value="generated-dot") as mock_dot:
        data = logger._prepare_log_data(
            kwargs={
                "standard_logging_object": _base_payload(),
                "litellm_params": {"metadata": {}},
            },
            response_obj=None,
            start_time=None,
            end_time=None,
            credentials=_credentials(),
        )
    assert isinstance(data["id"], str) and data["id"]
    assert data["trace_id"] == data["id"]
    assert data["dotted_order"] == "generated-dot"
    mock_dot.assert_called_once_with(run_id=data["id"])


def test_prepare_log_data_preserves_explicit_trace_and_dotted():
    logger = _make_logger()
    data = logger._prepare_log_data(
        kwargs={
            "standard_logging_object": _base_payload(),
            "litellm_params": {
                "metadata": {
                    "id": "run-explicit",
                    "trace_id": "trace-explicit",
                    "dotted_order": "dot-explicit",
                    "parent_run_id": "parent-1",
                    "session_id": "sess-1",
                    "project_name": "proj-explicit",
                    "run_name": "named-run",
                }
            },
        },
        response_obj=None,
        start_time=None,
        end_time=None,
        credentials=_credentials(),
    )
    assert data["id"] == "run-explicit"
    assert data["trace_id"] == "trace-explicit"
    assert data["dotted_order"] == "dot-explicit"
    assert data["parent_run_id"] == "parent-1"
    assert data["session_id"] == "sess-1"
    assert data["session_name"] == "proj-explicit"
    assert data["name"] == "named-run"
