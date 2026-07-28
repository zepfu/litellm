"""D1-542 NVIDIA NIM embedding integration and outbound acceptance tests."""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm.exceptions import UnsupportedParamsError
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.param_adaptation import (
    MAX_PARAM_NAME_LENGTH,
    MAX_RECORDS,
    PROVIDER_PARAMETER_ADAPTATIONS_METADATA_KEY,
    PROVIDER_PARAMETER_ADAPTATIONS_TRUNCATED_COUNT_METADATA_KEY,
    AdaptationCollector,
)
from litellm.llms.openai.openai import OpenAIChatCompletion
from litellm.utils import function_setup, get_optional_params_embeddings

MODEL = "nvidia/nv-embedqa-e5-v5"
ROUTED_MODEL = f"nvidia_nim/{MODEL}"


@pytest.fixture(autouse=True)
def restore_drop_params():
    original = litellm.drop_params
    litellm.drop_params = False
    yield
    litellm.drop_params = original


def _get_optional(**kwargs):
    return get_optional_params_embeddings(
        model=MODEL,
        custom_llm_provider="nvidia_nim",
        **kwargs,
    )


def _provider_response() -> MagicMock:
    response = MagicMock()
    response.model_dump.return_value = {
        "object": "list",
        "data": [{"object": "embedding", "index": 0, "embedding": [0.1]}],
        "model": MODEL,
        "usage": {"prompt_tokens": 1, "total_tokens": 1},
    }
    return response


def _logging_obj(metadata: dict, call_type: str) -> Logging:
    logging_obj = Logging(
        model=ROUTED_MODEL,
        messages=["hello"],
        stream=False,
        call_type=call_type,
        start_time=datetime.now(),
        litellm_call_id=f"d1-542-{call_type}",
        function_id="",
        kwargs={"metadata": metadata},
    )
    logging_obj.success_handler = MagicMock()  # type: ignore[method-assign]
    logging_obj.failure_handler = MagicMock()  # type: ignore[method-assign]
    logging_obj.async_failure_handler = AsyncMock()  # type: ignore[method-assign]
    logging_obj.handle_sync_success_callbacks_for_async_calls = MagicMock()  # type: ignore[method-assign]
    return logging_obj


def _capture_default_logging(captured: dict):
    def capture_function_setup(*args, **kwargs):
        logging_obj, updated_kwargs = function_setup(*args, **kwargs)
        metadata = updated_kwargs.get("metadata")
        request_id = (
            metadata.get("request_id", "default")
            if isinstance(metadata, dict)
            else "default"
        )
        captured[request_id] = logging_obj
        logging_obj.success_handler = MagicMock()  # type: ignore[method-assign]
        logging_obj.failure_handler = MagicMock()  # type: ignore[method-assign]
        logging_obj.async_failure_handler = AsyncMock()  # type: ignore[method-assign]
        logging_obj.handle_sync_success_callbacks_for_async_calls = MagicMock()  # type: ignore[method-assign]
        return logging_obj, updated_kwargs

    return capture_function_setup


def _dedicated_thread_executor_loop():
    running_loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="d1-542")
    loop = MagicMock()

    def run_in_executor(_executor, func):
        return running_loop.run_in_executor(executor, func)

    loop.run_in_executor.side_effect = run_in_executor
    return loop, executor


async def _gather_with_heartbeat(*awaitables):
    result = asyncio.gather(*awaitables)
    while not result.done():
        await asyncio.sleep(0.01)
    return await result


def _sync_outbound(metadata: dict, **kwargs):
    logging_obj = _logging_obj(metadata=metadata, call_type="embedding")
    request = MagicMock(return_value=({}, _provider_response()))
    with patch.object(
        OpenAIChatCompletion,
        "make_sync_openai_embedding_request",
        new=request,
    ):
        response = litellm.embedding(
            model=ROUTED_MODEL,
            input=["hello"],
            metadata=metadata,
            litellm_logging_obj=logging_obj,
            client=MagicMock(),
            **kwargs,
        )
    call_kwargs = request.call_args.kwargs
    sdk_kwargs = {**call_kwargs["data"], "timeout": call_kwargs["timeout"]}
    return response, sdk_kwargs, logging_obj


async def _async_outbound(metadata: dict, **kwargs):
    logging_obj = _logging_obj(metadata=metadata, call_type="aembedding")
    request = AsyncMock(return_value=({}, _provider_response()))
    executor_loop, executor = _dedicated_thread_executor_loop()
    try:
        with (
            patch.object(
                OpenAIChatCompletion,
                "make_openai_embedding_request",
                new=request,
            ),
            patch(
                "litellm.utils._client_async_logging_helper",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "litellm.main.asyncio.get_event_loop",
                return_value=executor_loop,
            ),
        ):
            response = (
                await _gather_with_heartbeat(
                    litellm.aembedding(
                        model=ROUTED_MODEL,
                        input=["hello"],
                        metadata=metadata,
                        litellm_logging_obj=logging_obj,
                        client=MagicMock(),
                        **kwargs,
                    )
                )
            )[0]
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    call_kwargs = request.call_args.kwargs
    sdk_kwargs = {**call_kwargs["data"], "timeout": call_kwargs["timeout"]}
    return response, sdk_kwargs, logging_obj


def test_should_serialize_plain_deterministic_value_free_metadata():
    collector = AdaptationCollector()
    collector.add("z_param", "dropped", "unsupported_param")
    collector.add("a_param", "rejected", "extra_body_policy")

    metadata = collector.to_metadata()

    assert metadata == {
        PROVIDER_PARAMETER_ADAPTATIONS_METADATA_KEY: [
            {
                "name": "a_param",
                "action": "rejected",
                "reason": "extra_body_policy",
            },
            {
                "name": "z_param",
                "action": "dropped",
                "reason": "unsupported_param",
            },
        ],
        PROVIDER_PARAMETER_ADAPTATIONS_TRUNCATED_COUNT_METADATA_KEY: 0,
    }
    assert json.loads(json.dumps(metadata)) == metadata


def test_should_serialize_value_free_provider_rename_metadata():
    collector = AdaptationCollector()
    collector.add("top_n", "renamed", "provider_rename")

    assert collector.to_metadata() == {
        PROVIDER_PARAMETER_ADAPTATIONS_METADATA_KEY: [
            {
                "name": "top_n",
                "action": "renamed",
                "reason": "provider_rename",
            }
        ],
        PROVIDER_PARAMETER_ADAPTATIONS_TRUNCATED_COUNT_METADATA_KEY: 0,
    }


def test_should_enforce_strict_default_with_names_only():
    top_level_value = "top-level-secret-value"
    nested_value = "nested-secret-value"

    with pytest.raises(UnsupportedParamsError) as exc_info:
        _get_optional(
            input_type="query",
            arbitrary=top_level_value,
            extra_body={"api_key": nested_value},
        )

    message = str(exc_info.value)
    assert "api_key" in message
    assert "arbitrary" in message
    assert top_level_value not in message
    assert nested_value not in message


def test_should_reject_strict_outbound_before_provider_invocation():
    secret = "strict-outbound-secret"
    spoof_value = "strict-spoof-value"
    metadata = {
        "source": "strict-direct",
        PROVIDER_PARAMETER_ADAPTATIONS_METADATA_KEY: [
            {"name": "spoof", "value": spoof_value}
        ],
        PROVIDER_PARAMETER_ADAPTATIONS_TRUNCATED_COUNT_METADATA_KEY: 999,
    }
    proxy_metadata = {
        "model_info": {"id": "strict-deployment"},
        PROVIDER_PARAMETER_ADAPTATIONS_METADATA_KEY: [
            {"name": "proxy-spoof", "value": spoof_value}
        ],
        PROVIDER_PARAMETER_ADAPTATIONS_TRUNCATED_COUNT_METADATA_KEY: 888,
    }
    captured = {}
    request = MagicMock()

    with (
        patch(
            "litellm.utils.function_setup",
            side_effect=_capture_default_logging(captured),
        ),
        patch.object(
            OpenAIChatCompletion,
            "make_sync_openai_embedding_request",
            new=request,
        ),
    ):
        with pytest.raises(UnsupportedParamsError) as exc_info:
            litellm.embedding(
                model=ROUTED_MODEL,
                input=["hello"],
                metadata=metadata,
                litellm_metadata=proxy_metadata,
                client=MagicMock(),
                arbitrary=secret,
            )

    adaptations = [
        {
            "name": "arbitrary",
            "action": "rejected",
            "reason": "unsupported_param",
        }
    ]
    assert "arbitrary" in str(exc_info.value)
    assert secret not in str(exc_info.value)
    assert metadata == {
        "source": "strict-direct",
        PROVIDER_PARAMETER_ADAPTATIONS_METADATA_KEY: adaptations,
        PROVIDER_PARAMETER_ADAPTATIONS_TRUNCATED_COUNT_METADATA_KEY: 0,
    }
    assert proxy_metadata == {"model_info": {"id": "strict-deployment"}}
    logging_metadata = captured["default"].model_call_details["litellm_params"][
        "metadata"
    ]
    assert logging_metadata == {
        "model_info": {"id": "strict-deployment"},
        "source": "strict-direct",
        PROVIDER_PARAMETER_ADAPTATIONS_METADATA_KEY: adaptations,
        PROVIDER_PARAMETER_ADAPTATIONS_TRUNCATED_COUNT_METADATA_KEY: 0,
    }
    serialized_metadata = json.dumps(logging_metadata)
    assert secret not in serialized_metadata
    assert spoof_value not in serialized_metadata
    request.assert_not_called()


def test_should_drop_request_local_params_and_preserve_allowed_fields():
    result = _get_optional(
        drop_params=True,
        user="user-1",
        dimensions=768,
        encoding_format="base64",
        input_type="query",
        truncate="END",
        modality="text",
        embedding_type="float",
        arbitrary="secret-value",
        none_value=None,
        max_tokens=1,
    )

    assert result == {
        "user": "user-1",
        "dimensions": 768,
        "encoding_format": "base64",
        "extra_body": {
            "input_type": "query",
            "truncate": "END",
            "modality": "text",
            "embedding_type": "float",
        },
    }


def test_should_apply_global_drop_params_without_mutating_global_state():
    litellm.drop_params = True

    assert _get_optional(
        drop_params=False,
        input_type="query",
        arbitrary="secret-value",
    ) == {"extra_body": {"input_type": "query"}}
    assert litellm.drop_params is True

    litellm.drop_params = False
    with pytest.raises(UnsupportedParamsError):
        _get_optional(input_type="query", arbitrary="secret-value")


def test_should_enforce_explicit_extra_body_rules_and_collision_policy():
    with pytest.raises(UnsupportedParamsError) as exc_info:
        _get_optional(
            input_type="query",
            extra_body={
                "input_type": "passage",
                "api_key": "nested-secret",
                "modality": "text",
            },
        )
    assert "api_key" in str(exc_info.value)
    assert "input_type" in str(exc_info.value)
    assert "nested-secret" not in str(exc_info.value)

    result = _get_optional(
        drop_params=True,
        input_type="query",
        extra_body={
            "input_type": "passage",
            "api_key": "nested-secret",
            "modality": "text",
        },
    )
    assert result == {
        "extra_body": {
            "input_type": "query",
            "modality": "text",
        }
    }


def test_should_omit_none_empty_extra_body_and_health_max_tokens():
    assert (
        _get_optional(
            drop_params=True,
            input_type=None,
            truncate=None,
            max_tokens=1,
        )
        == {}
    )
    assert _get_optional(
        drop_params=True,
        input_type="query",
        max_tokens=1,
    ) == {"extra_body": {"input_type": "query"}}


def test_should_honor_additional_drop_params_for_provider_fields():
    result = _get_optional(
        drop_params=True,
        additional_drop_params=["truncate"],
        input_type="query",
        truncate="END",
    )

    assert result == {"extra_body": {"input_type": "query"}}


def test_should_scrub_direct_metadata_spoofs_and_persist_genuine_records():
    top_secret = "top-level-secret"
    nested_secret = "nested-api-secret"
    metadata = {
        "source": "direct",
        PROVIDER_PARAMETER_ADAPTATIONS_METADATA_KEY: [
            {"name": "spoof", "value": "must-not-survive"}
        ],
        PROVIDER_PARAMETER_ADAPTATIONS_TRUNCATED_COUNT_METADATA_KEY: 999,
    }

    response, sdk_kwargs, logging_obj = _sync_outbound(
        metadata,
        drop_params=True,
        user="user-1",
        dimensions=512,
        encoding_format="float",
        input_type="query",
        truncate="END",
        embedding_type="float",
        extra_body={"modality": "text", "api_key": nested_secret},
        arbitrary=top_secret,
        internal_trace={"secret": top_secret},
        secret_token=top_secret,
    )

    assert isinstance(response, litellm.EmbeddingResponse)
    assert sdk_kwargs == {
        "model": MODEL,
        "input": ["hello"],
        "dimensions": 512,
        "encoding_format": "float",
        "user": "user-1",
        "extra_body": {
            "input_type": "query",
            "truncate": "END",
            "embedding_type": "float",
            "modality": "text",
        },
        "timeout": 600,
    }
    adaptations = [
        {
            "name": "api_key",
            "action": "rejected",
            "reason": "extra_body_policy",
        },
        {
            "name": "arbitrary",
            "action": "dropped",
            "reason": "unsupported_param",
        },
        {
            "name": "internal_trace",
            "action": "dropped",
            "reason": "unsupported_param",
        },
        {
            "name": "secret_token",
            "action": "dropped",
            "reason": "unsupported_param",
        },
    ]
    assert metadata[PROVIDER_PARAMETER_ADAPTATIONS_METADATA_KEY] == adaptations
    assert metadata[PROVIDER_PARAMETER_ADAPTATIONS_TRUNCATED_COUNT_METADATA_KEY] == 0
    assert metadata["source"] == "direct"
    logging_metadata = logging_obj.model_call_details["litellm_params"]["metadata"]
    assert logging_metadata[PROVIDER_PARAMETER_ADAPTATIONS_METADATA_KEY] == adaptations
    assert logging_obj.model_call_details["optional_params"] == {
        key: value
        for key, value in sdk_kwargs.items()
        if key not in {"model", "input", "timeout"}
    }
    serialized_metadata = json.dumps(logging_metadata)
    assert top_secret not in serialized_metadata
    assert nested_secret not in serialized_metadata
    assert "must-not-survive" not in serialized_metadata
    assert "must-not-survive" not in json.dumps(sdk_kwargs)
    assert PROVIDER_PARAMETER_ADAPTATIONS_METADATA_KEY not in sdk_kwargs
    assert PROVIDER_PARAMETER_ADAPTATIONS_TRUNCATED_COUNT_METADATA_KEY not in sdk_kwargs


def test_should_persist_metadata_through_default_direct_logging_setup():
    captured = {}

    request = MagicMock(return_value=({}, _provider_response()))
    with (
        patch(
            "litellm.utils.function_setup",
            side_effect=_capture_default_logging(captured),
        ),
        patch.object(
            OpenAIChatCompletion,
            "make_sync_openai_embedding_request",
            new=request,
        ),
    ):
        response = litellm.embedding(
            model=ROUTED_MODEL,
            input=["hello"],
            client=MagicMock(),
            drop_params=True,
            arbitrary="direct-secret-value",
        )

    assert isinstance(response, litellm.EmbeddingResponse)
    adaptations = [
        {
            "name": "arbitrary",
            "action": "dropped",
            "reason": "unsupported_param",
        }
    ]
    logging_obj = captured["default"]
    logging_metadata = logging_obj.model_call_details["litellm_params"]["metadata"]
    assert logging_metadata[PROVIDER_PARAMETER_ADAPTATIONS_METADATA_KEY] == adaptations
    assert "direct-secret-value" not in json.dumps(logging_metadata)
    sdk_kwargs = request.call_args.kwargs["data"]
    assert PROVIDER_PARAMETER_ADAPTATIONS_METADATA_KEY not in sdk_kwargs


@pytest.mark.asyncio
async def test_should_scrub_proxy_metadata_spoofs_and_persist_genuine_records():
    metadata = {"model_group": "proxy-nvidia"}
    spoof_value = "proxy-spoof-value"
    proxy_metadata = {
        "model_info": {"id": "deployment-1"},
        PROVIDER_PARAMETER_ADAPTATIONS_METADATA_KEY: [
            {"name": "proxy-spoof", "value": spoof_value}
        ],
        PROVIDER_PARAMETER_ADAPTATIONS_TRUNCATED_COUNT_METADATA_KEY: 777,
    }

    response, sdk_kwargs, logging_obj = await _async_outbound(
        metadata,
        drop_params=True,
        litellm_metadata=proxy_metadata,
        input_type="passage",
        truncate="NONE",
        arbitrary="async-secret-value",
    )

    assert isinstance(response, litellm.EmbeddingResponse)
    assert sdk_kwargs == {
        "model": MODEL,
        "input": ["hello"],
        "extra_body": {"input_type": "passage", "truncate": "NONE"},
        "timeout": 600,
    }
    adaptations = [
        {
            "name": "arbitrary",
            "action": "dropped",
            "reason": "unsupported_param",
        }
    ]
    logging_metadata = logging_obj.model_call_details["litellm_params"]["metadata"]
    assert logging_metadata["model_group"] == "proxy-nvidia"
    assert logging_metadata["model_info"] == {"id": "deployment-1"}
    assert logging_metadata[PROVIDER_PARAMETER_ADAPTATIONS_METADATA_KEY] == adaptations
    assert (
        logging_metadata[PROVIDER_PARAMETER_ADAPTATIONS_TRUNCATED_COUNT_METADATA_KEY]
        == 0
    )
    assert proxy_metadata == {"model_info": {"id": "deployment-1"}}
    assert "async-secret-value" not in json.dumps(logging_metadata)
    assert spoof_value not in json.dumps(logging_metadata)
    assert spoof_value not in json.dumps(sdk_kwargs)
    assert PROVIDER_PARAMETER_ADAPTATIONS_METADATA_KEY not in sdk_kwargs
    assert PROVIDER_PARAMETER_ADAPTATIONS_TRUNCATED_COUNT_METADATA_KEY not in sdk_kwargs


@pytest.mark.asyncio
async def test_should_isolate_concurrent_async_strict_adaptation_metadata():
    request = AsyncMock()
    executor_loop, executor = _dedicated_thread_executor_loop()

    async def strict_call(request_id: str, **kwargs):
        spoof_value = f"{request_id}-spoof-value"
        metadata = {
            "request_id": request_id,
            PROVIDER_PARAMETER_ADAPTATIONS_METADATA_KEY: [
                {"name": "spoof", "value": spoof_value}
            ],
            PROVIDER_PARAMETER_ADAPTATIONS_TRUNCATED_COUNT_METADATA_KEY: 999,
        }
        proxy_metadata = {
            "proxy_request_id": request_id,
            PROVIDER_PARAMETER_ADAPTATIONS_METADATA_KEY: [
                {"name": "proxy-spoof", "value": spoof_value}
            ],
            PROVIDER_PARAMETER_ADAPTATIONS_TRUNCATED_COUNT_METADATA_KEY: 888,
        }
        logging_obj = _logging_obj(metadata=metadata, call_type="aembedding")
        with pytest.raises(UnsupportedParamsError) as exc_info:
            await litellm.aembedding(
                model=ROUTED_MODEL,
                input=["hello"],
                metadata=metadata,
                litellm_metadata=proxy_metadata,
                litellm_logging_obj=logging_obj,
                client=MagicMock(),
                **kwargs,
            )
        return (
            metadata,
            proxy_metadata,
            logging_obj,
            str(exc_info.value),
            spoof_value,
        )

    try:
        with (
            patch.object(
                OpenAIChatCompletion,
                "make_openai_embedding_request",
                new=request,
            ),
            patch(
                "litellm.main.asyncio.get_event_loop",
                return_value=executor_loop,
            ),
        ):
            alpha, beta = await _gather_with_heartbeat(
                strict_call("alpha", alpha_param="alpha-secret-value"),
                strict_call(
                    "beta",
                    extra_body={"api_key": "beta-secret-value"},
                ),
            )
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    expected = {
        "alpha": [
            {
                "name": "alpha_param",
                "action": "rejected",
                "reason": "unsupported_param",
            }
        ],
        "beta": [
            {
                "name": "api_key",
                "action": "rejected",
                "reason": "extra_body_policy",
            }
        ],
    }
    for request_id, result in zip(("alpha", "beta"), (alpha, beta)):
        metadata, proxy_metadata, logging_obj, error_message, spoof_value = result
        assert (
            metadata[PROVIDER_PARAMETER_ADAPTATIONS_METADATA_KEY]
            == expected[request_id]
        )
        assert (
            metadata[PROVIDER_PARAMETER_ADAPTATIONS_TRUNCATED_COUNT_METADATA_KEY] == 0
        )
        assert proxy_metadata == {"proxy_request_id": request_id}
        logging_metadata = logging_obj.model_call_details["litellm_params"]["metadata"]
        assert logging_metadata == {
            "request_id": request_id,
            "proxy_request_id": request_id,
            PROVIDER_PARAMETER_ADAPTATIONS_METADATA_KEY: expected[request_id],
            PROVIDER_PARAMETER_ADAPTATIONS_TRUNCATED_COUNT_METADATA_KEY: 0,
        }
        other_request_id = "beta" if request_id == "alpha" else "alpha"
        assert other_request_id not in json.dumps(
            logging_metadata[PROVIDER_PARAMETER_ADAPTATIONS_METADATA_KEY]
        )
        assert spoof_value not in json.dumps(logging_metadata)
        assert "secret-value" not in error_message

    request.assert_not_called()


def test_should_bound_metadata_truncate_names_and_never_store_values():
    long_name = "a" * 200
    unsupported = {long_name: "long-secret-value"}
    unsupported.update(
        {f"param_{index:02d}": f"secret-value-{index:02d}" for index in range(39)}
    )
    metadata = {}

    _, sdk_kwargs, logging_obj = _sync_outbound(
        metadata,
        drop_params=True,
        **unsupported,
    )

    adaptations = metadata[PROVIDER_PARAMETER_ADAPTATIONS_METADATA_KEY]
    assert len(adaptations) == MAX_RECORDS
    assert metadata[PROVIDER_PARAMETER_ADAPTATIONS_TRUNCATED_COUNT_METADATA_KEY] == 8
    assert adaptations[0]["name"] == "a" * MAX_PARAM_NAME_LENGTH
    assert all(set(record) == {"name", "action", "reason"} for record in adaptations)
    serialized_metadata = json.dumps(
        logging_obj.model_call_details["litellm_params"]["metadata"]
    )
    assert "secret-value" not in serialized_metadata
    assert sdk_kwargs == {
        "model": MODEL,
        "input": ["hello"],
        "timeout": 600,
    }


def test_should_remove_spoofed_reserved_metadata_when_no_adaptation_occurs():
    metadata = {
        "source": "direct",
        PROVIDER_PARAMETER_ADAPTATIONS_METADATA_KEY: [
            {"name": "spoof", "value": "secret"}
        ],
        PROVIDER_PARAMETER_ADAPTATIONS_TRUNCATED_COUNT_METADATA_KEY: 999,
    }
    proxy_metadata = {
        "proxy_source": "no-adaptation",
        PROVIDER_PARAMETER_ADAPTATIONS_METADATA_KEY: [
            {"name": "proxy-spoof", "value": "proxy-secret"}
        ],
        PROVIDER_PARAMETER_ADAPTATIONS_TRUNCATED_COUNT_METADATA_KEY: 888,
    }

    _, sdk_kwargs, logging_obj = _sync_outbound(
        metadata,
        litellm_metadata=proxy_metadata,
        input_type="query",
    )

    assert PROVIDER_PARAMETER_ADAPTATIONS_METADATA_KEY not in metadata
    assert PROVIDER_PARAMETER_ADAPTATIONS_TRUNCATED_COUNT_METADATA_KEY not in metadata
    assert proxy_metadata == {"proxy_source": "no-adaptation"}
    logging_metadata = logging_obj.model_call_details["litellm_params"]["metadata"]
    assert PROVIDER_PARAMETER_ADAPTATIONS_METADATA_KEY not in logging_metadata
    assert (
        PROVIDER_PARAMETER_ADAPTATIONS_TRUNCATED_COUNT_METADATA_KEY
        not in logging_metadata
    )
    assert logging_metadata["source"] == "direct"
    assert logging_metadata["proxy_source"] == "no-adaptation"
    assert PROVIDER_PARAMETER_ADAPTATIONS_METADATA_KEY not in sdk_kwargs
    assert PROVIDER_PARAMETER_ADAPTATIONS_TRUNCATED_COUNT_METADATA_KEY not in sdk_kwargs
    assert sdk_kwargs["extra_body"] == {"input_type": "query"}
