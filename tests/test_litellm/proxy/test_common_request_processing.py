import asyncio
import copy
import datetime
import json
import logging
import re
from typing import AsyncGenerator, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request, status
from fastapi.responses import JSONResponse, StreamingResponse

import litellm
import litellm.proxy.aawm_route_logging as aawm_route_logging
from litellm._logging import (
    AawmHealthAccessLogFilter,
    AawmRouteAccessLogReplacementFilter,
    clear_aawm_route_access_log_replacements,
)
from litellm._uuid import uuid
from litellm.integrations.opentelemetry import UserAPIKeyAuth
from litellm.proxy.common_request_processing import (
    ProxyBaseLLMRequestProcessing,
    ProxyConfig,
    _extract_error_from_sse_chunk,
    _get_cost_breakdown_from_logging_obj,
    _has_attribute_error_in_chain,
    _is_azure_model_router_request,
    _override_openai_response_model,
    _parse_event_data_for_error,
    create_response,
)
from litellm.proxy.dd_span_tagger import DDSpanTagger
from litellm.proxy.aawm_route_logging import (
    clear_aawm_route_rollups,
    clear_aawm_route_log_dedup_state,
    emit_aawm_route_access_log,
    flush_aawm_route_rollups,
    record_aawm_route_rollup_turn,
)
from litellm.proxy.utils import ProxyLogging
from litellm.types.rerank import RerankResponse


def _build_aawm_route_log_request(
    *,
    method: str = "POST",
    url: str = "http://127.0.0.1:4001/v1/embeddings",
    client: tuple[str, int] = ("172.19.0.1", 52834),
    http_version: str = "1.1",
    headers: Optional[dict[str, str]] = None,
) -> MagicMock:
    request = MagicMock(spec=Request)
    request.method = method
    request.url = url
    request.headers = headers or {}
    query = ""
    if "?" in url:
        query = url.split("?", 1)[1]
    request.scope = {
        "type": "http",
        "method": method,
        "path": "/" + url.split("://", 1)[-1].split("/", 1)[1].split("?", 1)[0],
        "query_string": query.encode("utf-8"),
        "client": client,
        "http_version": http_version,
    }
    return request


def _build_uvicorn_access_record(
    *,
    client_addr: str = "172.19.0.1:52834",
    method: str = "POST",
    full_path: str = "/v1/embeddings",
    http_version: str = "1.1",
    status_code: int = 200,
) -> logging.LogRecord:
    return logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg='%s - "%s %s HTTP/%s" %d',
        args=(client_addr, method, full_path, http_version, status_code),
        exc_info=None,
    )


def _assert_rerank_diagnostic_artifact(
    artifact: dict,
    *,
    rendered: str,
) -> None:
    manifest = artifact["manifest"]
    assert artifact["capture_kind"] == "aawm_diagnostic_payload_capture"
    assert manifest["environment"] == "aawm-dev"
    assert manifest["route_family"] == "rerank"
    assert manifest["endpoint_template"] == "/rerank"
    assert manifest["litellm_call_id"] == "test-call-id"
    assert manifest["mode"] == "nonstream"
    assert manifest["redaction_mode"] == "shape_hash_manifest"
    assert manifest["byte_counts"]["request_body_bytes"] > 0
    assert manifest["byte_counts"]["response_body_bytes"] > 0
    assert manifest["hashes"]["request_body_sha256"]
    assert manifest["hashes"]["response_body_sha256"]
    assert "request.body.raw" in manifest["omitted_fields"]
    assert artifact["metadata"]["custom_llm_provider"] == "cohere"
    assert artifact["request"]["body_shape"]["query"].startswith("<str len=")
    assert artifact["request"]["body_shape"]["documents"][0].startswith("<str len=")
    assert (
        artifact["response"]["body"]["shape"]["results"][0]["document"]["text"]
        == "<str len=33>"
    )
    assert "customer secret query" not in rendered
    assert "customer secret document" not in rendered
    assert "customer secret returned document" not in rendered


def _assert_rerank_route_record(caplog) -> None:
    route_records = [
        record.getMessage()
        for record in caplog.records
        if " [RERANK] " in record.getMessage()
    ]
    assert len(route_records) == 1
    assert re.fullmatch(
        r"\d{8} \d{2}:\d{2}:\d{2} \[RERANK\] Codex/0\.119\.0-alpha\.29 - "
        r"rerank worker@litellm\.tei-reranker "
        r"POST 172\.19\.0\.1:52834 /rerank -> "
        r"rerank\.example\.com/rerank",
        route_records[0],
    )
    assert "api_key" not in route_records[0]
    assert "customer secret query" not in route_records[0]
    assert "customer secret document" not in route_records[0]


def _build_rerank_user_api_key_auth() -> MagicMock:
    mock_user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)
    mock_user_api_key_dict.tpm_limit = None
    mock_user_api_key_dict.rpm_limit = None
    mock_user_api_key_dict.max_budget = None
    mock_user_api_key_dict.spend = 0
    mock_user_api_key_dict.allowed_model_region = None
    return mock_user_api_key_dict


def _enable_rerank_diagnostic_capture(monkeypatch, diagnostic_dir) -> None:
    monkeypatch.delenv("AAWM_CAPTURE_PASSTHROUGH_SHAPES", raising=False)
    monkeypatch.delenv("AAWM_CAPTURE_PASSTHROUGH_FULL_PAYLOADS", raising=False)
    monkeypatch.setenv("AAWM_DIAGNOSTIC_PAYLOAD_CAPTURE", "1")
    monkeypatch.setenv("AAWM_DIAGNOSTIC_PAYLOAD_CAPTURE_ROUTE_FAMILIES", "rerank")
    monkeypatch.setenv("AAWM_DIAGNOSTIC_PAYLOAD_CAPTURE_DIR", str(diagnostic_dir))
    monkeypatch.setenv("AAWM_DIAGNOSTIC_PAYLOAD_CAPTURE_ENVIRONMENT", "aawm-dev")


class TestProxyBaseLLMRequestProcessing:
    @pytest.mark.asyncio
    async def test_common_processing_pre_call_logic_pre_call_hook_receives_litellm_call_id(
        self, monkeypatch
    ):
        processing_obj = ProxyBaseLLMRequestProcessing(data={})
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}

        async def mock_add_litellm_data_to_request(*args, **kwargs):
            return {}

        async def mock_common_processing_pre_call_logic(
            user_api_key_dict, data, call_type
        ):
            data_copy = copy.deepcopy(data)
            return data_copy

        mock_proxy_logging_obj = MagicMock(spec=ProxyLogging)
        mock_proxy_logging_obj.pre_call_hook = AsyncMock(
            side_effect=mock_common_processing_pre_call_logic
        )
        monkeypatch.setattr(
            litellm.proxy.common_request_processing,
            "add_litellm_data_to_request",
            mock_add_litellm_data_to_request,
        )
        mock_general_settings = {}
        mock_user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)
        mock_proxy_config = MagicMock(spec=ProxyConfig)
        route_type = "acompletion"

        # Call the actual method.
        (
            returned_data,
            logging_obj,
        ) = await processing_obj.common_processing_pre_call_logic(
            request=mock_request,
            general_settings=mock_general_settings,
            user_api_key_dict=mock_user_api_key_dict,
            proxy_logging_obj=mock_proxy_logging_obj,
            proxy_config=mock_proxy_config,
            route_type=route_type,
        )

        mock_proxy_logging_obj.pre_call_hook.assert_called_once()

        _, call_kwargs = mock_proxy_logging_obj.pre_call_hook.call_args
        data_passed = call_kwargs.get("data", {})

        assert "litellm_call_id" in data_passed
        try:
            uuid.UUID(data_passed["litellm_call_id"])
        except ValueError:
            pytest.fail("litellm_call_id is not a valid UUID")
        assert data_passed["litellm_call_id"] == returned_data["litellm_call_id"]

    def test_add_dd_apm_tags_for_litellm_call_id_uses_dd_tracing_helper(self, monkeypatch):
        mock_set_active_span_tag = MagicMock(return_value=True)
        import litellm.proxy.dd_span_tagger

        monkeypatch.setattr(
            litellm.proxy.dd_span_tagger,
            "set_active_span_tag",
            mock_set_active_span_tag,
        )

        DDSpanTagger.tag_call_id("test-call-id")

        mock_set_active_span_tag.assert_called_once_with(
            "litellm.call_id", "test-call-id"
        )

    @pytest.mark.asyncio
    async def test_should_apply_hierarchical_router_settings_as_override(
        self, monkeypatch
    ):
        """
        Test that hierarchical router settings are stored as router_settings_override
        instead of creating a full user_config with model_list.

        This approach avoids expensive per-request Router instantiation by passing
        settings as kwargs overrides to the main router.
        """
        processing_obj = ProxyBaseLLMRequestProcessing(data={})
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}

        async def mock_add_litellm_data_to_request(*args, **kwargs):
            return {}

        async def mock_common_processing_pre_call_logic(
            user_api_key_dict, data, call_type
        ):
            data_copy = copy.deepcopy(data)
            return data_copy

        mock_proxy_logging_obj = MagicMock(spec=ProxyLogging)
        mock_proxy_logging_obj.pre_call_hook = AsyncMock(
            side_effect=mock_common_processing_pre_call_logic
        )
        monkeypatch.setattr(
            litellm.proxy.common_request_processing,
            "add_litellm_data_to_request",
            mock_add_litellm_data_to_request,
        )

        mock_general_settings = {}
        mock_user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)
        mock_proxy_config = MagicMock(spec=ProxyConfig)

        mock_router_settings = {
            "routing_strategy": "least-busy",
            "timeout": 30.0,
            "num_retries": 3,
        }
        mock_proxy_config._get_hierarchical_router_settings = AsyncMock(
            return_value=mock_router_settings
        )

        mock_llm_router = MagicMock()

        mock_prisma_client = MagicMock()
        monkeypatch.setattr(
            "litellm.proxy.proxy_server.prisma_client",
            mock_prisma_client,
        )

        route_type = "acompletion"

        (
            returned_data,
            logging_obj,
        ) = await processing_obj.common_processing_pre_call_logic(
            request=mock_request,
            general_settings=mock_general_settings,
            user_api_key_dict=mock_user_api_key_dict,
            proxy_logging_obj=mock_proxy_logging_obj,
            proxy_config=mock_proxy_config,
            route_type=route_type,
            llm_router=mock_llm_router,
        )

        mock_proxy_config._get_hierarchical_router_settings.assert_called_once_with(
            user_api_key_dict=mock_user_api_key_dict,
            prisma_client=mock_prisma_client,
            proxy_logging_obj=mock_proxy_logging_obj,
        )
        # get_model_list should NOT be called - we no longer copy model list for per-request routers
        mock_llm_router.get_model_list.assert_not_called()

        # Settings should be stored as router_settings_override (not user_config)
        # This allows passing them as kwargs to the main router instead of creating a new one
        assert "router_settings_override" in returned_data
        assert "user_config" not in returned_data

        router_settings_override = returned_data["router_settings_override"]
        assert router_settings_override["routing_strategy"] == "least-busy"
        assert router_settings_override["timeout"] == 30.0
        assert router_settings_override["num_retries"] == 3
        # model_list should NOT be in the override settings
        assert "model_list" not in router_settings_override

    @pytest.mark.asyncio
    async def test_base_process_llm_request_emits_aawm_route_log_for_embeddings(
        self,
        caplog,
        monkeypatch,
    ):
        clear_aawm_route_access_log_replacements()
        clear_aawm_route_rollups()
        monkeypatch.setenv("AAWM_ROUTE_ROLLUP_INTERVAL_SECONDS", "0")
        data = {
            "model": "text-embedding-3-small",
            "input": ["hello"],
            "api_base": "https://api.openai.com/v1/embeddings?api_key=secret",
            "metadata": {
                "agent_name": "embed worker",
                "repository": "litellm",
                "requested_model_alias": "aawm-mini",
            },
        }
        processing_obj = ProxyBaseLLMRequestProcessing(data=data)
        mock_request = _build_aawm_route_log_request(
            headers={"user-agent": "codex-cli/0.119.0-alpha.29"},
        )
        mock_fastapi_response = MagicMock()
        mock_fastapi_response.headers = {}
        mock_user_api_key_dict = _build_rerank_user_api_key_auth()
        mock_proxy_config = MagicMock(spec=ProxyConfig)
        logging_obj = MagicMock()
        logging_obj.litellm_call_id = "test-call-id"

        processing_obj.common_processing_pre_call_logic = AsyncMock(
            return_value=(data, logging_obj)
        )

        async def mock_provider_response():
            return {
                "object": "list",
                "model": "text-embedding-3-small",
                "data": [],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            }

        async def mock_route_request(*args, **kwargs):
            return asyncio.create_task(mock_provider_response())

        monkeypatch.setattr(
            litellm.proxy.common_request_processing,
            "route_request",
            mock_route_request,
        )

        mock_proxy_logging_obj = MagicMock(spec=ProxyLogging)
        mock_proxy_logging_obj.during_call_hook = AsyncMock(return_value=None)
        mock_proxy_logging_obj.update_request_status = AsyncMock(return_value=None)
        mock_proxy_logging_obj.post_call_success_hook = AsyncMock(
            side_effect=lambda *, response, **kwargs: response
        )
        mock_proxy_logging_obj.post_call_response_headers_hook = AsyncMock(
            return_value={}
        )

        route_logger = logging.getLogger("LiteLLM AAWM Route")
        route_logger.addHandler(caplog.handler)
        try:
            with caplog.at_level(logging.INFO, logger=route_logger.name):
                response = await processing_obj.base_process_llm_request(
                    request=mock_request,
                    fastapi_response=mock_fastapi_response,
                    user_api_key_dict=mock_user_api_key_dict,
                    route_type="aembedding",
                    proxy_logging_obj=mock_proxy_logging_obj,
                    general_settings={},
                    proxy_config=mock_proxy_config,
                )
        finally:
            route_logger.removeHandler(caplog.handler)
            clear_aawm_route_rollups()

        assert response["model"] == "text-embedding-3-small"
        route_records = [
            record.getMessage()
            for record in caplog.records
            if " [EMBED] " in record.getMessage()
        ]
        assert len(route_records) == 1
        assert re.fullmatch(
            r"\d{8} \d{2}:\d{2}:\d{2} \[EMBED\] Codex/0\.119\.0-alpha\.29 - "
            r"embed worker@litellm\.text-embedding-3-small\(aawm-mini\) "
            r"POST 172\.19\.0\.1:52834 /v1/embeddings -> "
            r"api\.openai\.com/v1/embeddings",
            route_records[0],
        )
        assert "api_key" not in route_records[0]

        access_filter = AawmRouteAccessLogReplacementFilter()
        assert access_filter.filter(_build_uvicorn_access_record()) is False
        assert access_filter.filter(_build_uvicorn_access_record()) is True

    @pytest.mark.asyncio
    async def test_base_process_llm_request_emits_aawm_route_log_for_rerank(
        self,
        caplog,
        monkeypatch,
        tmp_path,
    ):
        clear_aawm_route_access_log_replacements()
        clear_aawm_route_rollups()
        monkeypatch.setenv("AAWM_ROUTE_ROLLUP_INTERVAL_SECONDS", "0")
        diagnostic_dir = tmp_path / "diagnostic"
        _enable_rerank_diagnostic_capture(monkeypatch, diagnostic_dir)
        data = {
            "model": "tei-reranker",
            "query": "customer secret query",
            "documents": ["customer secret document"],
            "api_base": "https://rerank.example.com/rerank?api_key=secret",
            "metadata": {
                "agent_name": "rerank worker",
                "repository": "litellm",
                "requested_model_alias": "tei-reranker",
            },
        }
        processing_obj = ProxyBaseLLMRequestProcessing(data=data)
        mock_request = _build_aawm_route_log_request(
            url="http://127.0.0.1:4001/rerank",
            headers={"user-agent": "codex-cli/0.119.0-alpha.29"},
        )
        mock_fastapi_response = MagicMock()
        mock_fastapi_response.headers = {}
        mock_user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)
        mock_user_api_key_dict.tpm_limit = None
        mock_user_api_key_dict.rpm_limit = None
        mock_user_api_key_dict.max_budget = None
        mock_user_api_key_dict.spend = 0
        mock_user_api_key_dict.allowed_model_region = None
        mock_proxy_config = MagicMock(spec=ProxyConfig)
        logging_obj = MagicMock()
        logging_obj.litellm_call_id = "test-call-id"

        processing_obj.common_processing_pre_call_logic = AsyncMock(
            return_value=(data, logging_obj)
        )

        async def mock_provider_response():
            response = RerankResponse(
                results=[
                    {
                        "index": 0,
                        "relevance_score": 0.91,
                        "document": {"text": "customer secret returned document"},
                    }
                ],
                meta={"billed_units": {"search_units": 1}},
            )
            response._hidden_params = {
                "api_base": "https://rerank.example.com/rerank?api_key=secret",
                "custom_llm_provider": "cohere",
            }
            return response

        async def mock_route_request(*args, **kwargs):
            return asyncio.create_task(mock_provider_response())

        monkeypatch.setattr(
            litellm.proxy.common_request_processing,
            "route_request",
            mock_route_request,
        )

        mock_proxy_logging_obj = MagicMock(spec=ProxyLogging)
        mock_proxy_logging_obj.during_call_hook = AsyncMock(return_value=None)
        mock_proxy_logging_obj.update_request_status = AsyncMock(return_value=None)
        mock_proxy_logging_obj.post_call_success_hook = AsyncMock(
            side_effect=lambda *, response, **kwargs: response
        )
        mock_proxy_logging_obj.post_call_response_headers_hook = AsyncMock(
            return_value={}
        )

        route_logger = logging.getLogger("LiteLLM AAWM Route")
        route_logger.addHandler(caplog.handler)
        try:
            with caplog.at_level(logging.INFO, logger=route_logger.name):
                response = await processing_obj.base_process_llm_request(
                    request=mock_request,
                    fastapi_response=mock_fastapi_response,
                    user_api_key_dict=mock_user_api_key_dict,
                    route_type="arerank",
                    proxy_logging_obj=mock_proxy_logging_obj,
                    general_settings={},
                    proxy_config=mock_proxy_config,
                )
        finally:
            route_logger.removeHandler(caplog.handler)
            clear_aawm_route_rollups()

        assert response._hidden_params["api_base"].startswith(
            "https://rerank.example.com/rerank"
        )
        _assert_rerank_route_record(caplog)

        artifacts = sorted(diagnostic_dir.glob("*.json"))
        assert len(artifacts) == 1
        artifact = json.loads(artifacts[0].read_text(encoding="utf-8"))
        rendered = json.dumps(artifact)
        _assert_rerank_diagnostic_artifact(artifact, rendered=rendered)

        access_filter = AawmRouteAccessLogReplacementFilter()
        assert access_filter.filter(
            _build_uvicorn_access_record(full_path="/rerank")
        ) is False
        assert access_filter.filter(
            _build_uvicorn_access_record(full_path="/rerank")
        ) is True

    def test_health_access_log_filter_suppresses_successful_health_checks(self):
        access_filter = AawmHealthAccessLogFilter()

        assert (
            access_filter.filter(
                _build_uvicorn_access_record(
                    method="GET",
                    full_path="/health/readiness",
                    status_code=200,
                )
            )
            is False
        )
        assert (
            access_filter.filter(
                _build_uvicorn_access_record(
                    method="GET",
                    full_path="/health/liveliness",
                    status_code=204,
                )
            )
            is False
        )
        assert (
            access_filter.filter(
                _build_uvicorn_access_record(
                    method="GET",
                    full_path="/health/readiness",
                    status_code=503,
                )
            )
            is True
        )
        assert access_filter.filter(_build_uvicorn_access_record()) is True

    def test_aawm_route_log_deduplicates_repeated_route_context(
        self,
        caplog,
        monkeypatch,
    ):
        clear_aawm_route_access_log_replacements()
        clear_aawm_route_rollups()
        clear_aawm_route_log_dedup_state()
        monkeypatch.setenv("AAWM_ROUTE_ROLLUP_INTERVAL_SECONDS", "0")
        monkeypatch.setenv("AAWM_ROUTE_LOG_DEDUP_WINDOW_SECONDS", "60")
        request_body = {
            "model": "text-embedding-3-small",
            "metadata": {
                "agent_name": "embed worker",
                "repository": "litellm",
                "requested_model_alias": "aawm-mini",
            },
        }
        first_request = _build_aawm_route_log_request(
            headers={"user-agent": "codex-cli/0.119.0-alpha.29"},
        )
        second_request = _build_aawm_route_log_request(
            client=("172.19.0.1", 52835),
            headers={"user-agent": "codex-cli/0.119.0-alpha.29"},
        )

        route_logger = logging.getLogger("LiteLLM AAWM Route")
        route_logger.addHandler(caplog.handler)
        try:
            with caplog.at_level(logging.INFO, logger=route_logger.name):
                emit_aawm_route_access_log(
                    request=first_request,
                    target="https://api.openai.com/v1/embeddings",
                    request_body=request_body,
                    route_type="aembedding",
                )
                emit_aawm_route_access_log(
                    request=second_request,
                    target="https://api.openai.com/v1/embeddings",
                    request_body=request_body,
                    route_type="aembedding",
                )
        finally:
            route_logger.removeHandler(caplog.handler)
            clear_aawm_route_rollups()
            clear_aawm_route_log_dedup_state()

        route_records = [
            record.getMessage()
            for record in caplog.records
            if " [EMBED] " in record.getMessage()
        ]
        assert len(route_records) == 1

        access_filter = AawmRouteAccessLogReplacementFilter()
        assert access_filter.filter(_build_uvicorn_access_record()) is False
        assert (
            access_filter.filter(
                _build_uvicorn_access_record(client_addr="172.19.0.1:52835")
            )
            is False
        )

    def test_aawm_route_log_rollup_records_completed_turns(
        self,
        caplog,
        monkeypatch,
    ):
        clear_aawm_route_access_log_replacements()
        clear_aawm_route_rollups()
        monkeypatch.setenv("AAWM_ROUTE_ROLLUP_INTERVAL_SECONDS", "60")
        request_body = {
            "model": "gpt-5.5",
            "metadata": {
                "agent_name": "litellm",
                "repository": "litellm",
            },
        }
        kwargs = {"litellm_params": {"metadata": {}}}
        request = _build_aawm_route_log_request(
            url="http://127.0.0.1:4001/openai_passthrough/responses",
            headers={"user-agent": "codex-cli/0.141.0"},
        )

        route_logger = logging.getLogger("LiteLLM AAWM Route")
        route_logger.addHandler(caplog.handler)
        try:
            with caplog.at_level(logging.INFO, logger=route_logger.name):
                emit_aawm_route_access_log(
                    request=request,
                    target="https://chatgpt.com/backend-api/codex/responses",
                    request_body=request_body,
                    kwargs=kwargs,
                )
                record_aawm_route_rollup_turn(kwargs)
                flushed = flush_aawm_route_rollups(force=True)
        finally:
            route_logger.removeHandler(caplog.handler)
            clear_aawm_route_rollups()

        rendered = "\n".join(flushed)
        assert len(flushed) == 2
        assert (
            "litellm@Codex[0.141.0] /openai_passthrough/responses"
        ) in rendered
        assert (
            " - gpt-5.5 - Turns: 1 -> "
            "chatgpt.com/backend-api/codex/responses"
        ) in rendered
        assert " [ROUTE] " not in rendered

    def test_aawm_route_rollup_groups_by_model_alias_without_agent_breakouts(
        self,
        caplog,
        monkeypatch,
    ):
        clear_aawm_route_access_log_replacements()
        clear_aawm_route_rollups()
        monkeypatch.setenv("AAWM_ROUTE_ROLLUP_INTERVAL_SECONDS", "60")
        base_metadata = {
            "repository": "aegis",
            "client_name": "claude-code",
            "client_version": "2.1.178",
        }
        request = _build_aawm_route_log_request(
            url="http://127.0.0.1:4001/anthropic/v1/messages?beta=true",
            headers={"user-agent": "claude-code/2.1.178"},
        )

        route_logger = logging.getLogger("LiteLLM AAWM Route")
        route_logger.addHandler(caplog.handler)
        try:
            with caplog.at_level(logging.INFO, logger=route_logger.name):
                first_kwargs = {"litellm_params": {"metadata": {}}}
                emit_aawm_route_access_log(
                    request=request,
                    target="https://api.anthropic.com/v1/messages",
                    request_body={
                        "model": "claude-opus-4-8",
                        "metadata": base_metadata,
                    },
                    kwargs=first_kwargs,
                )
                record_aawm_route_rollup_turn(first_kwargs)

                second_kwargs = {"litellm_params": {"metadata": {}}}
                emit_aawm_route_access_log(
                    request=_build_aawm_route_log_request(
                        url="http://127.0.0.1:4001/anthropic/v1/messages?beta=true",
                        headers={"user-agent": "claude-code/2.1.178"},
                    ),
                    target="https://api.anthropic.com/v1/messages",
                    request_body={
                        "model": "claude-opus-4-8",
                        "metadata": {
                            **base_metadata,
                            "agent_name": "orchestrator",
                        },
                    },
                    kwargs=second_kwargs,
                )
                record_aawm_route_rollup_turn(second_kwargs)
                flushed = flush_aawm_route_rollups(force=True)
        finally:
            route_logger.removeHandler(caplog.handler)
            clear_aawm_route_rollups()

        rendered = "\n".join(flushed)
        assert len(flushed) == 2
        assert (
            "aegis@Claude[2.1.178] /anthropic/v1/messages?beta=true"
        ) in rendered
        assert (
            " - claude-opus-4-8 - Turns: 2 -> "
            "api.anthropic.com/v1/messages"
        ) in rendered
        assert "orchestrator.claude-opus-4-8" not in rendered

    def test_aawm_route_rollup_disabled_restores_immediate_route_log(
        self,
        caplog,
        monkeypatch,
    ):
        clear_aawm_route_access_log_replacements()
        clear_aawm_route_rollups()
        monkeypatch.setenv("AAWM_ROUTE_ROLLUP_INTERVAL_SECONDS", "0")
        request = _build_aawm_route_log_request(
            url="http://127.0.0.1:4001/openai_passthrough/responses",
            headers={"user-agent": "codex-cli/0.141.0"},
        )

        route_logger = logging.getLogger("LiteLLM AAWM Route")
        route_logger.addHandler(caplog.handler)
        try:
            with caplog.at_level(logging.INFO, logger=route_logger.name):
                emit_aawm_route_access_log(
                    request=request,
                    target="https://chatgpt.com/backend-api/codex/responses",
                    request_body={"model": "gpt-5.5"},
                    completed=True,
                )
        finally:
            route_logger.removeHandler(caplog.handler)
            clear_aawm_route_rollups()

        rendered = "\n".join(record.getMessage() for record in caplog.records)
        assert " [ROUTE] Codex/0.141.0" in rendered
        assert "gpt-5.5" in rendered

    @pytest.mark.asyncio
    async def test_stream_timeout_header_processing(self):
        """
        Test that x-litellm-stream-timeout header gets processed and added to request data as stream_timeout.
        """
        from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup

        # Test with stream timeout header
        headers_with_timeout = {"x-litellm-stream-timeout": "30.5"}
        result = LiteLLMProxyRequestSetup._get_stream_timeout_from_request(
            headers_with_timeout
        )
        assert result == 30.5

        # Test without stream timeout header
        headers_without_timeout = {}
        result = LiteLLMProxyRequestSetup._get_stream_timeout_from_request(
            headers_without_timeout
        )
        assert result is None

        # Test with invalid header value (should raise ValueError when converting to float)
        headers_with_invalid = {"x-litellm-stream-timeout": "invalid"}
        with pytest.raises(ValueError):
            LiteLLMProxyRequestSetup._get_stream_timeout_from_request(
                headers_with_invalid
            )

    @pytest.mark.asyncio
    async def test_add_litellm_data_to_request_with_stream_timeout_header(self):
        """
        Test that x-litellm-stream-timeout header gets processed and added to request data
        when calling add_litellm_data_to_request.
        """
        from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request

        # Create test data with a basic completion request
        test_data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        # Mock request with stream timeout header
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {"x-litellm-stream-timeout": "45.0"}
        mock_request.url.path = "/v1/chat/completions"
        mock_request.method = "POST"
        mock_request.query_params = {}
        mock_request.client = None

        # Create a minimal mock with just the required attributes
        mock_user_api_key_dict = MagicMock()
        mock_user_api_key_dict.api_key = "test_api_key_hash"
        mock_user_api_key_dict.tpm_limit = None
        mock_user_api_key_dict.rpm_limit = None
        mock_user_api_key_dict.max_budget = None
        mock_user_api_key_dict.spend = 0
        mock_user_api_key_dict.allowed_model_region = None
        mock_user_api_key_dict.key_alias = None
        mock_user_api_key_dict.user_id = None
        mock_user_api_key_dict.team_id = None
        mock_user_api_key_dict.metadata = {}  # Prevent enterprise feature check
        mock_user_api_key_dict.team_metadata = None
        mock_user_api_key_dict.org_id = None
        mock_user_api_key_dict.team_alias = None
        mock_user_api_key_dict.end_user_id = None
        mock_user_api_key_dict.user_email = None
        mock_user_api_key_dict.request_route = None
        mock_user_api_key_dict.team_max_budget = None
        mock_user_api_key_dict.team_spend = None
        mock_user_api_key_dict.model_max_budget = None
        mock_user_api_key_dict.parent_otel_span = None
        mock_user_api_key_dict.team_model_aliases = None

        general_settings = {}
        mock_proxy_config = MagicMock()

        # Call the actual function that processes headers and adds data
        result_data = await add_litellm_data_to_request(
            data=test_data,
            request=mock_request,
            general_settings=general_settings,
            user_api_key_dict=mock_user_api_key_dict,
            version=None,
            proxy_config=mock_proxy_config,
        )

        # Verify that stream_timeout was extracted from header and added to request data
        assert "stream_timeout" in result_data
        assert result_data["stream_timeout"] == 45.0

        # Verify that the original test data is preserved
        assert result_data["model"] == "gpt-3.5-turbo"
        assert result_data["messages"] == [{"role": "user", "content": "Hello"}]

    def test_get_custom_headers_with_discount_info(self):
        """
        Test that discount information is correctly extracted from logging object
        and included in response headers.
        """
        from litellm.litellm_core_utils.litellm_logging import (
            Logging as LiteLLMLoggingObj,
        )

        # Create mock user API key dict
        mock_user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)
        mock_user_api_key_dict.tpm_limit = None
        mock_user_api_key_dict.rpm_limit = None
        mock_user_api_key_dict.max_budget = None
        mock_user_api_key_dict.spend = 0

        # Create logging object with cost breakdown including discount
        logging_obj = LiteLLMLoggingObj(
            model="vertex_ai/gemini-pro",
            messages=[{"role": "user", "content": "test"}],
            stream=False,
            call_type="completion",
            start_time=None,
            litellm_call_id="test-call-id",
            function_id="test-function-id",
        )

        # Set cost breakdown with discount information
        logging_obj.set_cost_breakdown(
            input_cost=0.00005,
            output_cost=0.00005,
            total_cost=0.000095,  # After 5% discount
            cost_for_built_in_tools_cost_usd_dollar=0.0,
            original_cost=0.0001,
            discount_percent=0.05,
            discount_amount=0.000005,
        )

        # Call get_custom_headers with discount info
        headers = ProxyBaseLLMRequestProcessing.get_custom_headers(
            user_api_key_dict=mock_user_api_key_dict,
            call_id="test-call-id",
            response_cost=0.000095,
            litellm_logging_obj=logging_obj,
        )

        # Verify discount headers are present
        assert "x-litellm-response-cost" in headers
        assert float(headers["x-litellm-response-cost"]) == 0.000095

        assert "x-litellm-response-cost-original" in headers
        assert float(headers["x-litellm-response-cost-original"]) == 0.0001

        assert "x-litellm-response-cost-discount-amount" in headers
        assert float(headers["x-litellm-response-cost-discount-amount"]) == 0.000005

    def test_get_custom_headers_without_discount_info(self):
        """
        Test that when no discount is applied, discount headers are not included.
        """
        from litellm.litellm_core_utils.litellm_logging import (
            Logging as LiteLLMLoggingObj,
        )

        # Create mock user API key dict
        mock_user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)
        mock_user_api_key_dict.tpm_limit = None
        mock_user_api_key_dict.rpm_limit = None
        mock_user_api_key_dict.max_budget = None
        mock_user_api_key_dict.spend = 0

        # Create logging object without discount
        logging_obj = LiteLLMLoggingObj(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            stream=False,
            call_type="completion",
            start_time=None,
            litellm_call_id="test-call-id",
            function_id="test-function-id",
        )

        # Set cost breakdown without discount information
        logging_obj.set_cost_breakdown(
            input_cost=0.00005,
            output_cost=0.00005,
            total_cost=0.0001,
            cost_for_built_in_tools_cost_usd_dollar=0.0,
        )

        # Call get_custom_headers
        headers = ProxyBaseLLMRequestProcessing.get_custom_headers(
            user_api_key_dict=mock_user_api_key_dict,
            call_id="test-call-id",
            response_cost=0.0001,
            litellm_logging_obj=logging_obj,
        )

        # Verify discount headers are NOT present
        assert "x-litellm-response-cost" in headers
        assert float(headers["x-litellm-response-cost"]) == 0.0001

        # Discount headers should not be in the final dict
        assert "x-litellm-response-cost-original" not in headers
        assert "x-litellm-response-cost-discount-amount" not in headers

    def test_get_custom_headers_with_margin_info(self):
        """
        Test that margin headers are included when margin is applied.
        """
        from litellm.litellm_core_utils.litellm_logging import (
            Logging as LiteLLMLoggingObj,
        )

        # Create mock user API key dict
        mock_user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)
        mock_user_api_key_dict.tpm_limit = None
        mock_user_api_key_dict.rpm_limit = None
        mock_user_api_key_dict.max_budget = None
        mock_user_api_key_dict.spend = 0

        # Create logging object with margin
        logging_obj = LiteLLMLoggingObj(
            model="gpt-4",
            messages=[],
            stream=False,
            call_type="completion",
            start_time=None,
            litellm_call_id="test-call-id-margin",
            function_id="test-function",
        )
        logging_obj.set_cost_breakdown(
            input_cost=0.00005,
            output_cost=0.00005,
            total_cost=0.00011,
            cost_for_built_in_tools_cost_usd_dollar=0.0,
            original_cost=0.0001,
            margin_percent=0.10,
            margin_total_amount=0.00001,
        )

        headers = ProxyBaseLLMRequestProcessing.get_custom_headers(
            user_api_key_dict=mock_user_api_key_dict,
            response_cost=0.00011,
            litellm_logging_obj=logging_obj,
        )

        # Verify margin headers are present
        assert "x-litellm-response-cost" in headers
        assert float(headers["x-litellm-response-cost"]) == 0.00011

        assert "x-litellm-response-cost-margin-amount" in headers
        assert float(headers["x-litellm-response-cost-margin-amount"]) == 0.00001

        assert "x-litellm-response-cost-margin-percent" in headers
        assert float(headers["x-litellm-response-cost-margin-percent"]) == 0.10

    def test_get_custom_headers_without_margin_info(self):
        """
        Test that when no margin is applied, margin headers are not included.
        """
        from litellm.litellm_core_utils.litellm_logging import (
            Logging as LiteLLMLoggingObj,
        )

        # Create mock user API key dict
        mock_user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)
        mock_user_api_key_dict.tpm_limit = None
        mock_user_api_key_dict.rpm_limit = None
        mock_user_api_key_dict.max_budget = None
        mock_user_api_key_dict.spend = 0

        # Create logging object without margin
        logging_obj = LiteLLMLoggingObj(
            model="gpt-4",
            messages=[],
            stream=False,
            call_type="completion",
            start_time=None,
            litellm_call_id="test-call-id-no-margin",
            function_id="test-function",
        )
        logging_obj.set_cost_breakdown(
            input_cost=0.00005,
            output_cost=0.00005,
            total_cost=0.0001,
            cost_for_built_in_tools_cost_usd_dollar=0.0,
        )

        headers = ProxyBaseLLMRequestProcessing.get_custom_headers(
            user_api_key_dict=mock_user_api_key_dict,
            response_cost=0.0001,
            litellm_logging_obj=logging_obj,
        )

        # Verify margin headers are not present
        assert "x-litellm-response-cost-margin-amount" not in headers
        assert "x-litellm-response-cost-margin-percent" not in headers

    def test_get_cost_breakdown_from_logging_obj_helper(self):
        """
        Test the helper function that extracts cost breakdown information.
        """
        from litellm.litellm_core_utils.litellm_logging import (
            Logging as LiteLLMLoggingObj,
        )

        # Test with discount info
        logging_obj = LiteLLMLoggingObj(
            model="vertex_ai/gemini-pro",
            messages=[{"role": "user", "content": "test"}],
            stream=False,
            call_type="completion",
            start_time=None,
            litellm_call_id="test-call-id",
            function_id="test-function-id",
        )
        logging_obj.set_cost_breakdown(
            input_cost=0.00005,
            output_cost=0.00005,
            total_cost=0.000095,
            cost_for_built_in_tools_cost_usd_dollar=0.0,
            original_cost=0.0001,
            discount_percent=0.05,
            discount_amount=0.000005,
        )

        (
            original_cost,
            discount_amount,
            margin_total_amount,
            margin_percent,
        ) = _get_cost_breakdown_from_logging_obj(logging_obj)
        assert original_cost == 0.0001
        assert discount_amount == 0.000005
        assert margin_total_amount is None
        assert margin_percent is None

        # Test with margin info
        logging_obj_with_margin = LiteLLMLoggingObj(
            model="gpt-4",
            messages=[{"role": "user", "content": "test"}],
            stream=False,
            call_type="completion",
            start_time=None,
            litellm_call_id="test-call-id-margin",
            function_id="test-function-id-margin",
        )
        logging_obj_with_margin.set_cost_breakdown(
            input_cost=0.00005,
            output_cost=0.00005,
            total_cost=0.00011,
            cost_for_built_in_tools_cost_usd_dollar=0.0,
            original_cost=0.0001,
            margin_percent=0.10,
            margin_total_amount=0.00001,
        )

        (
            original_cost,
            discount_amount,
            margin_total_amount,
            margin_percent,
        ) = _get_cost_breakdown_from_logging_obj(logging_obj_with_margin)
        assert original_cost == 0.0001
        assert discount_amount is None
        assert margin_total_amount == 0.00001
        assert margin_percent == 0.10

        # Test with no discount or margin info
        logging_obj_no_discount = LiteLLMLoggingObj(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            stream=False,
            call_type="completion",
            start_time=None,
            litellm_call_id="test-call-id-2",
            function_id="test-function-id-2",
        )
        logging_obj_no_discount.set_cost_breakdown(
            input_cost=0.00005,
            output_cost=0.00005,
            total_cost=0.0001,
            cost_for_built_in_tools_cost_usd_dollar=0.0,
        )

        (
            original_cost,
            discount_amount,
            margin_total_amount,
            margin_percent,
        ) = _get_cost_breakdown_from_logging_obj(logging_obj_no_discount)
        assert original_cost is None
        assert discount_amount is None
        assert margin_total_amount is None
        assert margin_percent is None

        # Test with None logging object
        (
            original_cost,
            discount_amount,
            margin_total_amount,
            margin_percent,
        ) = _get_cost_breakdown_from_logging_obj(None)
        assert original_cost is None
        assert discount_amount is None
        assert margin_total_amount is None
        assert margin_percent is None

    def test_get_custom_headers_key_spend_includes_response_cost(self):
        """
        Test that x-litellm-key-spend header includes the current request's response_cost.

        This ensures that the spend header reflects the updated spend including the current
        request, even though spend tracking updates happen asynchronously after the response.
        """
        # Create mock user API key dict with initial spend
        mock_user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)
        mock_user_api_key_dict.tpm_limit = None
        mock_user_api_key_dict.rpm_limit = None
        mock_user_api_key_dict.max_budget = None
        mock_user_api_key_dict.spend = 0.001  # Initial spend: $0.001

        # Test case 1: response_cost is provided as float
        response_cost_1 = 0.0005  # Current request cost: $0.0005
        headers_1 = ProxyBaseLLMRequestProcessing.get_custom_headers(
            user_api_key_dict=mock_user_api_key_dict,
            call_id="test-call-id-1",
            response_cost=response_cost_1,
        )

        assert "x-litellm-key-spend" in headers_1
        expected_spend_1 = 0.001 + 0.0005  # Initial spend + current request cost
        assert float(headers_1["x-litellm-key-spend"]) == pytest.approx(
            expected_spend_1, abs=1e-10
        )
        assert float(headers_1["x-litellm-response-cost"]) == response_cost_1

        # Test case 2: response_cost is provided as string
        response_cost_2 = "0.0003"  # Current request cost as string
        headers_2 = ProxyBaseLLMRequestProcessing.get_custom_headers(
            user_api_key_dict=mock_user_api_key_dict,
            call_id="test-call-id-2",
            response_cost=response_cost_2,
        )

        assert "x-litellm-key-spend" in headers_2
        expected_spend_2 = 0.001 + 0.0003  # Initial spend + current request cost
        assert float(headers_2["x-litellm-key-spend"]) == pytest.approx(
            expected_spend_2, abs=1e-10
        )

        # Test case 3: response_cost is None (should use original spend)
        headers_3 = ProxyBaseLLMRequestProcessing.get_custom_headers(
            user_api_key_dict=mock_user_api_key_dict,
            call_id="test-call-id-3",
            response_cost=None,
        )

        assert "x-litellm-key-spend" in headers_3
        assert (
            float(headers_3["x-litellm-key-spend"]) == 0.001
        )  # Should use original spend

        # Test case 4: response_cost is 0 (should not change spend)
        headers_4 = ProxyBaseLLMRequestProcessing.get_custom_headers(
            user_api_key_dict=mock_user_api_key_dict,
            call_id="test-call-id-4",
            response_cost=0.0,
        )

        assert "x-litellm-key-spend" in headers_4
        assert (
            float(headers_4["x-litellm-key-spend"]) == 0.001
        )  # Should remain unchanged for 0 cost

        # Test case 5: user_api_key_dict.spend is None (should default to 0.0)
        mock_user_api_key_dict.spend = None
        headers_5 = ProxyBaseLLMRequestProcessing.get_custom_headers(
            user_api_key_dict=mock_user_api_key_dict,
            call_id="test-call-id-5",
            response_cost=0.0002,
        )

        assert "x-litellm-key-spend" in headers_5
        assert float(headers_5["x-litellm-key-spend"]) == 0.0002  # 0.0 + 0.0002

        # Test case 6: response_cost is negative (should not be added, use original spend)
        mock_user_api_key_dict.spend = 0.001
        headers_6 = ProxyBaseLLMRequestProcessing.get_custom_headers(
            user_api_key_dict=mock_user_api_key_dict,
            call_id="test-call-id-6",
            response_cost=-0.0001,  # Negative cost (should not be added)
        )

        assert "x-litellm-key-spend" in headers_6
        assert (
            float(headers_6["x-litellm-key-spend"]) == 0.001
        )  # Should use original spend

        # Test case 7: response_cost is invalid string (should fallback to original spend)
        headers_7 = ProxyBaseLLMRequestProcessing.get_custom_headers(
            user_api_key_dict=mock_user_api_key_dict,
            call_id="test-call-id-7",
            response_cost="invalid",  # Invalid string
        )

        assert "x-litellm-key-spend" in headers_7
        assert (
            float(headers_7["x-litellm-key-spend"]) == 0.001
        )  # Should use original spend on error

    @pytest.mark.asyncio
    async def test_queue_time_seconds_is_set_in_metadata(self, monkeypatch):
        """
        Test that queue_time_seconds is correctly calculated and stored in metadata
        after add_litellm_data_to_request populates arrival_time.

        This verifies the fix for the bug where queue_time_seconds was always None
        because arrival_time was read BEFORE add_litellm_data_to_request set it.
        """
        processing_obj = ProxyBaseLLMRequestProcessing(data={})
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        mock_request.url = MagicMock()
        mock_request.url.path = "/v1/chat/completions"

        async def mock_add_litellm_data_to_request(*args, **kwargs):
            data = kwargs.get("data", args[0] if args else {})
            # Simulate what add_litellm_data_to_request does: set arrival_time
            import time

            data["proxy_server_request"] = {
                "url": "/v1/chat/completions",
                "method": "POST",
                "headers": {},
                "body": {},
                "arrival_time": time.time() - 0.5,  # Simulate request arrived 0.5s ago
            }
            data["metadata"] = data.get("metadata", {})
            return data

        async def mock_pre_call_hook(user_api_key_dict, data, call_type):
            return copy.deepcopy(data)

        mock_proxy_logging_obj = MagicMock(spec=ProxyLogging)
        mock_proxy_logging_obj.pre_call_hook = AsyncMock(side_effect=mock_pre_call_hook)
        monkeypatch.setattr(
            litellm.proxy.common_request_processing,
            "add_litellm_data_to_request",
            mock_add_litellm_data_to_request,
        )
        mock_general_settings = {}
        mock_user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)
        mock_proxy_config = MagicMock(spec=ProxyConfig)
        route_type = "acompletion"

        (
            returned_data,
            logging_obj,
        ) = await processing_obj.common_processing_pre_call_logic(
            request=mock_request,
            general_settings=mock_general_settings,
            user_api_key_dict=mock_user_api_key_dict,
            proxy_logging_obj=mock_proxy_logging_obj,
            proxy_config=mock_proxy_config,
            route_type=route_type,
        )

        # Verify queue_time_seconds is set and non-negative
        metadata = returned_data.get("metadata", {})
        assert (
            "queue_time_seconds" in metadata
        ), "queue_time_seconds should be set in metadata"
        assert (
            metadata["queue_time_seconds"] >= 0.5
        ), f"queue_time_seconds should be at least 0.5, got {metadata['queue_time_seconds']}"


@pytest.mark.asyncio
class TestCommonRequestProcessingHelpers:
    async def consume_stream(self, streaming_response: StreamingResponse) -> list:
        content = []
        async for chunk_bytes in streaming_response.body_iterator:
            content.append(chunk_bytes)
        return content

    @pytest.mark.parametrize(
        "event_line, expected_code",
        [
            (
                'data: {"error": {"code": 400, "message": "bad request"}}',
                400,
            ),  # Valid integer code
            (
                'data: {"error": {"code": "401", "message": "unauthorized"}}',
                401,
            ),  # Valid string-integer code
            (
                'data: {"error": {"code": "invalid_code", "message": "error"}}',
                None,
            ),  # Invalid string code
            (
                'data: {"error": {"code": 99, "message": "too low"}}',
                None,
            ),  # Integer code too low
            (
                'data: {"error": {"code": 600, "message": "too high"}}',
                None,
            ),  # Integer code too high
            (
                'data: {"id": "123", "content": "hello"}',
                None,
            ),  # Non-error SSE event
            ("data: [DONE]", None),  # SSE [DONE] event
            ("data: ", None),  # SSE empty data event
            (
                'data: {"error": {"code": 400',
                None,
            ),  # Malformed JSON
            ("id: 123", None),  # Non-SSE event line
            (
                'data: {"error": {"message": "some error"}}',
                None,
            ),  # Error event without 'code' field
            (
                'data: {"error": {"code": null, "message": "code is null"}}',
                None,
            ),  # Error with null code
        ],
    )
    async def test_parse_event_data_for_error(self, event_line, expected_code):
        assert await _parse_event_data_for_error(event_line) == expected_code

    async def test_create_streaming_response_first_chunk_is_error(self):
        """
        Test that when the first chunk is an error, a JSON error response is returned
        instead of an SSE streaming response
        """

        async def mock_generator():
            yield 'data: {"error": {"code": 403, "message": "forbidden"}}\n\n'
            yield 'data: {"content": "more data"}\n\n'
            yield "data: [DONE]\n\n"

        response = await create_response(mock_generator(), "text/event-stream", {})
        # Should return JSONResponse instead of StreamingResponse
        assert isinstance(response, JSONResponse)
        assert response.status_code == status.HTTP_403_FORBIDDEN
        # Verify the response is in standard JSON error format
        import json

        body = json.loads(response.body.decode())
        assert "error" in body
        assert body["error"]["code"] == 403
        assert body["error"]["message"] == "forbidden"

    async def test_create_streaming_response_first_chunk_not_error(self):
        async def mock_generator():
            yield 'data: {"content": "first part"}\n\n'
            yield 'data: {"content": "second part"}\n\n'
            yield "data: [DONE]\n\n"

        response = await create_response(mock_generator(), "text/event-stream", {})
        assert response.status_code == status.HTTP_200_OK
        content = await self.consume_stream(response)
        assert content == [
            'data: {"content": "first part"}\n\n',
            'data: {"content": "second part"}\n\n',
            "data: [DONE]\n\n",
        ]

    async def test_create_streaming_response_empty_generator(self):
        async def mock_generator():
            if False:  # Never yields
                yield
            # Implicitly raises StopAsyncIteration

        response = await create_response(mock_generator(), "text/event-stream", {})
        assert response.status_code == status.HTTP_200_OK
        content = await self.consume_stream(response)
        assert content == []

    async def test_create_streaming_response_generator_raises_stop_async_iteration_immediately(
        self,
    ):
        mock_gen = AsyncMock()
        mock_gen.__anext__.side_effect = StopAsyncIteration

        response = await create_response(mock_gen, "text/event-stream", {})
        assert response.status_code == status.HTTP_200_OK
        content = await self.consume_stream(response)
        assert content == []

    async def test_create_streaming_response_generator_raises_unexpected_exception(
        self,
    ):
        mock_gen = AsyncMock()
        mock_gen.__anext__.side_effect = ValueError("Test error from generator")

        response = await create_response(mock_gen, "text/event-stream", {})
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        content = await self.consume_stream(response)
        expected_error_data = {
            "error": {
                "message": "Error processing stream start",
                "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
            }
        }
        assert len(content) == 2
        # Use json.dumps to match the formatting in create_streaming_response's exception handler
        import json

        assert content[0] == f"data: {json.dumps(expected_error_data)}\n\n"
        assert content[1] == "data: [DONE]\n\n"

    async def test_create_streaming_response_first_chunk_error_string_code(self):
        """
        Test that when the first chunk contains a string error code, a JSON error response is returned
        """

        async def mock_generator():
            yield 'data: {"error": {"code": "429", "message": "too many requests"}}\n\n'
            yield "data: [DONE]\n\n"

        response = await create_response(mock_generator(), "text/event-stream", {})
        assert isinstance(response, JSONResponse)
        assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS
        # Verify the response is in standard JSON error format
        import json

        body = json.loads(response.body.decode())
        assert "error" in body
        assert body["error"]["code"] == "429"
        assert body["error"]["message"] == "too many requests"

    async def test_create_streaming_response_custom_headers(self):
        async def mock_generator():
            yield 'data: {"content": "data"}\n\n'
            yield "data: [DONE]\n\n"

        custom_headers = {"X-Custom-Header": "TestValue"}
        response = await create_response(
            mock_generator(), "text/event-stream", custom_headers
        )
        assert response.headers["x-custom-header"] == "TestValue"

    async def test_create_streaming_response_non_default_status_code(self):
        async def mock_generator():
            yield 'data: {"content": "data"}\n\n'
            yield "data: [DONE]\n\n"

        response = await create_response(
            mock_generator(),
            "text/event-stream",
            {},
            default_status_code=status.HTTP_201_CREATED,
        )
        assert response.status_code == status.HTTP_201_CREATED
        content = await self.consume_stream(response)
        assert content == [
            'data: {"content": "data"}\n\n',
            "data: [DONE]\n\n",
        ]

    async def test_create_streaming_response_first_chunk_is_done(self):
        async def mock_generator():
            yield "data: [DONE]\n\n"

        response = await create_response(mock_generator(), "text/event-stream", {})
        assert response.status_code == status.HTTP_200_OK  # Default status
        content = await self.consume_stream(response)
        assert content == ["data: [DONE]\n\n"]

    async def test_create_streaming_response_first_chunk_is_empty_data(self):
        async def mock_generator():
            yield "data: \n\n"
            yield 'data: {"content": "actual data"}\n\n'
            yield "data: [DONE]\n\n"

        response = await create_response(mock_generator(), "text/event-stream", {})
        assert response.status_code == status.HTTP_200_OK  # Default status
        content = await self.consume_stream(response)
        assert content == [
            "data: \n\n",
            'data: {"content": "actual data"}\n\n',
            "data: [DONE]\n\n",
        ]

    async def test_create_streaming_response_all_chunks_have_dd_trace(self):
        """Test that all stream chunks are wrapped with dd trace at the streaming generator level"""
        from unittest.mock import patch

        # Create a mock tracer
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.trace.return_value.__enter__.return_value = mock_span
        mock_tracer.trace.return_value.__exit__.return_value = None

        # Mock generator with multiple chunks
        async def mock_generator():
            yield 'data: {"content": "chunk 1"}\n\n'
            yield 'data: {"content": "chunk 2"}\n\n'
            yield 'data: {"content": "chunk 3"}\n\n'
            yield "data: [DONE]\n\n"

        # Patch the tracer in the common_request_processing module
        with patch("litellm.proxy.common_request_processing.tracer", mock_tracer):
            response = await create_response(mock_generator(), "text/event-stream", {})

            assert response.status_code == 200

            # Consume the stream to trigger the tracer calls
            content = await self.consume_stream(response)

            # Verify all chunks are present
            assert len(content) == 4
            assert content[0] == 'data: {"content": "chunk 1"}\n\n'
            assert content[1] == 'data: {"content": "chunk 2"}\n\n'
            assert content[2] == 'data: {"content": "chunk 3"}\n\n'
            assert content[3] == "data: [DONE]\n\n"

            # Verify that tracer.trace was called for each chunk (4 chunks total)
            assert mock_tracer.trace.call_count == 4

            actual_calls = mock_tracer.trace.call_args_list
            assert len(actual_calls) == 4

            for i, call in enumerate(actual_calls):
                args, kwargs = call
                assert (
                    args[0] == "streaming.chunk.yield"
                ), f"Call {i} should have operation name 'streaming.chunk.yield', got {args[0]}"

    async def test_create_streaming_response_dd_trace_with_error_chunk(self):
        """
        Test that when the first chunk contains an error, JSONResponse is returned
        and tracing is not triggered (since it's not a streaming response)
        """
        from unittest.mock import patch

        # Create a mock tracer
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.trace.return_value.__enter__.return_value = mock_span
        mock_tracer.trace.return_value.__exit__.return_value = None

        # Mock generator with error in first chunk
        async def mock_generator():
            yield 'data: {"error": {"code": 400, "message": "bad request"}}\n\n'
            yield 'data: {"content": "chunk after error"}\n\n'
            yield "data: [DONE]\n\n"

        # Patch the tracer in the common_request_processing module
        with patch("litellm.proxy.common_request_processing.tracer", mock_tracer):
            response = await create_response(mock_generator(), "text/event-stream", {})

            # Should return JSONResponse instead of StreamingResponse
            assert isinstance(response, JSONResponse)
            assert response.status_code == 400

            # Verify the response is in standard JSON error format
            import json

            body = json.loads(response.body.decode())
            assert "error" in body
            assert body["error"]["code"] == 400
            assert body["error"]["message"] == "bad request"

            # Since JSONResponse is returned instead of StreamingResponse, streaming tracing should not be triggered
            # tracer.trace should not be called
            assert mock_tracer.trace.call_count == 0


class TestExtractErrorFromSSEChunk:
    """Tests for _extract_error_from_sse_chunk function"""

    def test_extract_error_from_sse_chunk_with_valid_error(self):
        """Test extracting error information from a standard SSE chunk"""
        chunk = 'data: {"error": {"code": 403, "message": "forbidden", "type": "auth_error", "param": "api_key"}}\n\n'
        error = _extract_error_from_sse_chunk(chunk)

        assert error["code"] == 403
        assert error["message"] == "forbidden"
        assert error["type"] == "auth_error"
        assert error["param"] == "api_key"

    def test_extract_error_from_sse_chunk_with_string_code(self):
        """Test error code as string type"""
        chunk = 'data: {"error": {"code": "429", "message": "too many requests"}}\n\n'
        error = _extract_error_from_sse_chunk(chunk)

        assert error["code"] == "429"
        assert error["message"] == "too many requests"

    def test_extract_error_from_sse_chunk_with_bytes(self):
        """Test input as bytes type"""
        chunk = b'data: {"error": {"code": 500, "message": "internal error"}}\n\n'
        error = _extract_error_from_sse_chunk(chunk)

        assert error["code"] == 500
        assert error["message"] == "internal error"

    def test_extract_error_from_sse_chunk_with_done(self):
        """Test [DONE] marker should return default error"""
        chunk = "data: [DONE]\n\n"
        error = _extract_error_from_sse_chunk(chunk)

        assert error["message"] == "Unknown error"
        assert error["type"] == "internal_server_error"
        assert error["code"] == "500"
        assert error["param"] is None

    def test_extract_error_from_sse_chunk_without_error_field(self):
        """Test missing error field should return default error"""
        chunk = 'data: {"content": "some content"}\n\n'
        error = _extract_error_from_sse_chunk(chunk)

        assert error["message"] == "Unknown error"
        assert error["type"] == "internal_server_error"
        assert error["code"] == "500"

    def test_extract_error_from_sse_chunk_with_invalid_json(self):
        """Test invalid JSON should return default error"""
        chunk = "data: {invalid json}\n\n"
        error = _extract_error_from_sse_chunk(chunk)

        assert error["message"] == "Unknown error"
        assert error["type"] == "internal_server_error"
        assert error["code"] == "500"

    def test_extract_error_from_sse_chunk_without_data_prefix(self):
        """Test missing 'data:' prefix should return default error"""
        chunk = '{"error": {"code": 400, "message": "bad request"}}\n\n'
        error = _extract_error_from_sse_chunk(chunk)

        assert error["message"] == "Unknown error"
        assert error["type"] == "internal_server_error"
        assert error["code"] == "500"

    def test_extract_error_from_sse_chunk_with_empty_string(self):
        """Test empty string should return default error"""
        chunk = ""
        error = _extract_error_from_sse_chunk(chunk)

        assert error["message"] == "Unknown error"
        assert error["type"] == "internal_server_error"
        assert error["code"] == "500"

    def test_extract_error_from_sse_chunk_with_minimal_error(self):
        """Test minimal error object"""
        chunk = 'data: {"error": {"message": "error occurred"}}\n\n'
        error = _extract_error_from_sse_chunk(chunk)

        assert error["message"] == "error occurred"
        # Other fields should be obtained from the original error object (if exists)


class TestOverrideOpenAIResponseModel:
    """Tests for _override_openai_response_model function"""

    def test_override_model_preserves_fallback_model_when_fallback_occurred_object(
        self,
    ):
        """
        Test that when a fallback occurred (x-litellm-attempted-fallbacks > 0),
        the actual model used (fallback model) is preserved instead of being
        overridden with the requested model.

        This is the regression test to ensure the model being called is properly
        displayed when a fallback happens.
        """
        requested_model = "gpt-4"
        fallback_model = "gpt-3.5-turbo"

        # Create a mock object response with fallback model
        # _hidden_params is an attribute (not a dict key) accessed via getattr
        response_obj = MagicMock()
        response_obj.model = fallback_model
        response_obj._hidden_params = {
            "additional_headers": {"x-litellm-attempted-fallbacks": 1}
        }

        # Call the function - should preserve fallback model
        _override_openai_response_model(
            response_obj=response_obj,
            requested_model=requested_model,
            log_context="test_context",
        )

        # Verify the model was NOT overridden - should still be the fallback model
        assert response_obj.model == fallback_model
        assert response_obj.model != requested_model

    def test_override_model_preserves_fallback_model_multiple_fallbacks(self):
        """
        Test that when multiple fallbacks occurred, the actual model used
        (fallback model) is preserved.
        """
        requested_model = "gpt-4"
        fallback_model = "claude-haiku-4-5-20251001"

        # Create a mock object response with fallback model
        response_obj = MagicMock()
        response_obj.model = fallback_model
        response_obj._hidden_params = {
            "additional_headers": {
                "x-litellm-attempted-fallbacks": 2  # Multiple fallbacks
            }
        }

        # Call the function - should preserve fallback model
        _override_openai_response_model(
            response_obj=response_obj,
            requested_model=requested_model,
            log_context="test_context",
        )

        # Verify the model was NOT overridden - should still be the fallback model
        assert response_obj.model == fallback_model
        assert response_obj.model != requested_model

    def test_override_model_overrides_when_no_fallback_dict(self):
        """
        Test that when no fallback occurred, the model is overridden
        to match the requested model (dict response).
        """
        requested_model = "gpt-4"
        downstream_model = "gpt-3.5-turbo"

        # Create a dict response without fallback
        # For dict responses, _hidden_params won't be found via getattr,
        # so the fallback check won't trigger and model will be overridden
        response_obj = {"model": downstream_model}

        # Call the function - should override to requested model
        _override_openai_response_model(
            response_obj=response_obj,
            requested_model=requested_model,
            log_context="test_context",
        )

        # Verify the model WAS overridden to requested model
        assert response_obj["model"] == requested_model

    def test_override_model_overrides_when_no_fallback_object(self):
        """
        Test that when no fallback occurred (object response), the model is overridden
        to match the requested model.
        """
        requested_model = "gpt-4"
        downstream_model = "gpt-3.5-turbo"

        # Create a mock object response without fallback
        response_obj = MagicMock()
        response_obj.model = downstream_model
        response_obj._hidden_params = {
            "additional_headers": {}  # No attempted_fallbacks header
        }

        # Call the function - should override to requested model
        _override_openai_response_model(
            response_obj=response_obj,
            requested_model=requested_model,
            log_context="test_context",
        )

        # Verify the model WAS overridden to requested model
        assert response_obj.model == requested_model

    def test_override_model_overrides_when_attempted_fallbacks_is_zero(self):
        """
        Test that when attempted_fallbacks is 0 (no fallback occurred),
        the model is overridden to match the requested model.
        """
        requested_model = "gpt-4"
        downstream_model = "gpt-3.5-turbo"

        # Create a mock object response
        response_obj = MagicMock()
        response_obj.model = downstream_model
        response_obj._hidden_params = {
            "additional_headers": {
                "x-litellm-attempted-fallbacks": 0  # Zero means no fallback occurred
            }
        }

        # Call the function - should override to requested model
        _override_openai_response_model(
            response_obj=response_obj,
            requested_model=requested_model,
            log_context="test_context",
        )

        # Verify the model WAS overridden to requested model
        assert response_obj.model == requested_model

    def test_override_model_overrides_when_attempted_fallbacks_is_none(self):
        """
        Test that when attempted_fallbacks is None (not set),
        the model is overridden to match the requested model.
        """
        requested_model = "gpt-4"
        downstream_model = "gpt-3.5-turbo"

        # Create a mock object response
        response_obj = MagicMock()
        response_obj.model = downstream_model
        response_obj._hidden_params = {
            "additional_headers": {"x-litellm-attempted-fallbacks": None}
        }

        # Call the function - should override to requested model
        _override_openai_response_model(
            response_obj=response_obj,
            requested_model=requested_model,
            log_context="test_context",
        )

        # Verify the model WAS overridden to requested model
        assert response_obj.model == requested_model

    def test_override_model_no_hidden_params(self):
        """
        Test that when _hidden_params is not present, the model is overridden
        to match the requested model.
        """
        requested_model = "gpt-4"
        downstream_model = "gpt-3.5-turbo"

        # Create a mock object response without _hidden_params
        response_obj = MagicMock()
        response_obj.model = downstream_model
        # Don't set _hidden_params - getattr will return {}

        # Call the function - should override to requested model
        _override_openai_response_model(
            response_obj=response_obj,
            requested_model=requested_model,
            log_context="test_context",
        )

        # Verify the model WAS overridden to requested model
        assert response_obj.model == requested_model

    def test_override_model_skips_response_without_model_field(self, caplog):
        """
        Test that response types without OpenAI-compatible model fields are skipped
        without emitting an error.
        """
        response_obj = object()

        with caplog.at_level(logging.ERROR):
            _override_openai_response_model(
                response_obj=response_obj,
                requested_model="tei-reranker",
                log_context="test_context",
            )

        assert not caplog.records

    def test_override_model_no_requested_model(self):
        """
        Test that when requested_model is None or empty, the function returns early
        without modifying the response.
        """
        fallback_model = "gpt-3.5-turbo"

        # Create a mock object response
        response_obj = MagicMock()
        response_obj.model = fallback_model
        response_obj._hidden_params = {
            "additional_headers": {"x-litellm-attempted-fallbacks": 1}
        }

        # Call the function with None requested_model
        _override_openai_response_model(
            response_obj=response_obj,
            requested_model=None,
            log_context="test_context",
        )

        # Verify the model was not changed
        assert response_obj.model == fallback_model

        # Call with empty string
        _override_openai_response_model(
            response_obj=response_obj,
            requested_model="",
            log_context="test_context",
        )

        # Verify the model was not changed
        assert response_obj.model == fallback_model

    def test_override_model_preserves_azure_model_router_actual_model(self):
        """
        Test that when the requested model is an Azure Model Router, the actual
        model used (returned in the response) is preserved instead of being
        overridden.
        """
        requested_model = "azure_ai/model_router"
        actual_model_used = "azure_ai/gpt-5-nano-2025-08-07"

        response_obj = MagicMock()
        response_obj.model = actual_model_used
        response_obj._hidden_params = {"additional_headers": {}}

        _override_openai_response_model(
            response_obj=response_obj,
            requested_model=requested_model,
            log_context="test_context",
        )
        assert response_obj.model == actual_model_used
        assert response_obj.model != requested_model

    def test_override_model_preserves_azure_model_router_with_deployment_name(self):
        """
        Test that Azure Model Router with deployment name pattern also preserves
        the actual model used.
        """
        requested_model = "azure_ai/model_router/my-deployment"
        actual_model_used = "azure_ai/gpt-4.1-nano-2025-04-14"

        response_obj = MagicMock()
        response_obj.model = actual_model_used
        response_obj._hidden_params = {"additional_headers": {}}

        _override_openai_response_model(
            response_obj=response_obj,
            requested_model=requested_model,
            log_context="test_context",
        )
        assert response_obj.model == actual_model_used
        assert response_obj.model != requested_model

    def test_override_model_preserves_azure_model_router_with_hyphen(self):
        """
        Test that Azure Model Router with hyphen pattern (model-router) also preserves
        the actual model used.
        """
        requested_model = "azure_ai/model-router"
        actual_model_used = "azure_ai/gpt-5-nano-2025-08-07"

        response_obj = MagicMock()
        response_obj.model = actual_model_used
        response_obj._hidden_params = {"additional_headers": {}}

        _override_openai_response_model(
            response_obj=response_obj,
            requested_model=requested_model,
            log_context="test_context",
        )
        assert response_obj.model == actual_model_used
        assert response_obj.model != requested_model


class TestIsAzureModelRouterRequest:
    """Tests for _is_azure_model_router_request helper"""

    def test_detects_model_router_with_underscore(self):
        assert _is_azure_model_router_request("azure_ai/model_router") is True
        assert _is_azure_model_router_request("azure_ai/model_router/my-deployment") is True

    def test_detects_model_router_with_hyphen(self):
        assert _is_azure_model_router_request("azure_ai/model-router") is True
        assert _is_azure_model_router_request("model-router") is True

    def test_rejects_regular_models(self):
        assert _is_azure_model_router_request("azure_ai/gpt-4") is False
        assert _is_azure_model_router_request("gpt-4") is False
        assert _is_azure_model_router_request("openai/gpt-3.5-turbo") is False


class TestStreamingOverheadHeader:
    """
    Tests that x-litellm-overhead-duration-ms is emitted in streaming responses.

    Regression tests for: streaming requests not including overhead header.
    """

    def test_get_custom_headers_includes_overhead_when_set(self):
        """
        get_custom_headers() returns x-litellm-overhead-duration-ms
        when litellm_overhead_time_ms is in hidden_params.
        """
        mock_user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)
        mock_user_api_key_dict.tpm_limit = None
        mock_user_api_key_dict.rpm_limit = None
        mock_user_api_key_dict.max_budget = None
        mock_user_api_key_dict.spend = 0.0
        mock_user_api_key_dict.allowed_model_region = None

        hidden_params = {
            "litellm_overhead_time_ms": 42.5,
            "_response_ms": 500.0,
            "model_id": "test-model-id",
            "api_base": "https://api.openai.com",
        }

        headers = ProxyBaseLLMRequestProcessing.get_custom_headers(
            user_api_key_dict=mock_user_api_key_dict,
            call_id="test-call-id",
            model_id="test-model-id",
            cache_key="",
            api_base="https://api.openai.com",
            version="1.0.0",
            response_cost=0.001,
            model_region="",
            hidden_params=hidden_params,
        )

        assert "x-litellm-overhead-duration-ms" in headers
        assert headers["x-litellm-overhead-duration-ms"] == "42.5"

    def test_get_custom_headers_omits_overhead_when_none(self):
        """
        get_custom_headers() omits x-litellm-overhead-duration-ms
        when litellm_overhead_time_ms is not in hidden_params.
        """
        mock_user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)
        mock_user_api_key_dict.tpm_limit = None
        mock_user_api_key_dict.rpm_limit = None
        mock_user_api_key_dict.max_budget = None
        mock_user_api_key_dict.spend = 0.0
        mock_user_api_key_dict.allowed_model_region = None

        hidden_params = {
            "_response_ms": 500.0,
            "model_id": "test-model-id",
        }

        headers = ProxyBaseLLMRequestProcessing.get_custom_headers(
            user_api_key_dict=mock_user_api_key_dict,
            call_id="test-call-id",
            model_id="test-model-id",
            cache_key="",
            api_base="https://api.openai.com",
            version="1.0.0",
            response_cost=0.001,
            model_region="",
            hidden_params=hidden_params,
        )

        # Should be absent (None gets filtered by exclude_values)
        assert "x-litellm-overhead-duration-ms" not in headers

    def test_update_response_metadata_sets_overhead_on_stream_wrapper(self):
        """
        update_response_metadata() sets litellm_overhead_time_ms on
        a streaming response's _hidden_params when llm_api_duration_ms is available.
        """
        from litellm.litellm_core_utils.llm_response_utils.response_metadata import (
            update_response_metadata,
        )

        # Mock the logging object with llm_api_duration_ms set
        mock_logging_obj = MagicMock()
        mock_logging_obj.model_call_details = {
            "llm_api_duration_ms": 200.0,
            "litellm_params": {},
        }
        mock_logging_obj.caching_details = None
        mock_logging_obj.callback_duration_ms = None
        mock_logging_obj.litellm_call_id = "test-call-id"
        mock_logging_obj._response_cost_calculator = MagicMock(return_value=0.001)

        # Simulate a streaming result object with _hidden_params (like CustomStreamWrapper)
        stream_result = MagicMock()
        stream_result._hidden_params = {
            "model_id": "test-model-id",
            "api_base": "https://api.openai.com",
            "additional_headers": {},
        }

        start_time = datetime.datetime.now() - datetime.timedelta(milliseconds=300)
        end_time = datetime.datetime.now()

        update_response_metadata(
            result=stream_result,
            logging_obj=mock_logging_obj,
            model="gpt-4o",
            kwargs={},
            start_time=start_time,
            end_time=end_time,
        )

        assert "litellm_overhead_time_ms" in stream_result._hidden_params
        overhead = stream_result._hidden_params["litellm_overhead_time_ms"]
        assert overhead is not None
        assert isinstance(overhead, float)
        # overhead = total_response_ms (~300ms) - llm_api_duration_ms (200ms) = ~100ms
        assert overhead > 0

    @pytest.mark.asyncio
    async def test_streaming_response_includes_overhead_header(self):
        """
        StreamingResponse returned by create_response() includes
        x-litellm-overhead-duration-ms in its headers.
        """

        async def mock_generator() -> AsyncGenerator[str, None]:
            yield 'data: {"id":"chatcmpl-test","choices":[{"delta":{"content":"hi"}}]}\n\n'
            yield "data: [DONE]\n\n"

        headers = {
            "x-litellm-overhead-duration-ms": "42.5",
            "x-litellm-call-id": "test-call-id",
            "x-litellm-model-id": "test-model-id",
        }

        response = await create_response(
            generator=mock_generator(),
            media_type="text/event-stream",
            headers=headers,
        )

        assert isinstance(response, StreamingResponse)
        assert response.headers.get("x-litellm-overhead-duration-ms") == "42.5"

    def test_streaming_overhead_header_in_custom_headers_from_stream_hidden_params(
        self,
    ):
        """
        Verifies that when get_custom_headers() is called with a streaming
        response's hidden_params (containing litellm_overhead_time_ms),
        the x-litellm-overhead-duration-ms header is correctly populated.

        This tests the critical path: update_response_metadata sets the value
        → get_custom_headers reads it → StreamingResponse header is set.
        """
        mock_user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)
        mock_user_api_key_dict.tpm_limit = None
        mock_user_api_key_dict.rpm_limit = None
        mock_user_api_key_dict.max_budget = None
        mock_user_api_key_dict.spend = 0.0
        mock_user_api_key_dict.allowed_model_region = None

        # This is what CustomStreamWrapper._hidden_params looks like after
        # update_response_metadata() has been called on it
        hidden_params = {
            "model_id": "openai-gpt4o-deployment",
            "api_base": "https://api.openai.com",
            "additional_headers": {},
            "litellm_overhead_time_ms": 55.3,  # set by update_response_metadata
            "_response_ms": 280.0,
            "litellm_call_id": "test-call-id",
            "response_cost": 0.002,
            "cache_key": None,
            "fastest_response_batch_completion": None,
            "callback_duration_ms": None,
        }

        custom_headers = ProxyBaseLLMRequestProcessing.get_custom_headers(
            user_api_key_dict=mock_user_api_key_dict,
            call_id="test-call-id",
            model_id=hidden_params.get("model_id"),
            cache_key=hidden_params.get("cache_key") or "",
            api_base=hidden_params.get("api_base") or "",
            version="1.0.0",
            response_cost=hidden_params.get("response_cost"),
            model_region="",
            hidden_params=hidden_params,
        )

        # The overhead header must be present and correct
        assert "x-litellm-overhead-duration-ms" in custom_headers, (
            "x-litellm-overhead-duration-ms header must be emitted during streaming. "
            "It was missing — this is the streaming overhead header regression."
        )
        assert custom_headers["x-litellm-overhead-duration-ms"] == "55.3"


class TestDDSpanTaggerTagRequest:
    """Tests for DDSpanTagger.tag_request - key/model DD span tagging."""

    def _make_user_api_key_dict(self, key_alias=None, token=None):
        from litellm.proxy._types import UserAPIKeyAuth

        d = UserAPIKeyAuth()
        d.key_alias = key_alias
        d.token = token
        return d

    def test_tags_key_alias_and_model(self):
        """key_alias and requested_model are set on the span when present."""
        user_key = self._make_user_api_key_dict(key_alias="my-prod-key", token="hashed123")

        with patch(
            "litellm.proxy.dd_span_tagger.set_active_span_tag"
        ) as mock_set_tag:
            DDSpanTagger.tag_request(
                user_api_key_dict=user_key,
                requested_model="gpt-4o",
            )

        mock_set_tag.assert_any_call("litellm.key_alias", "my-prod-key")
        mock_set_tag.assert_any_call("litellm.key_hash", "hashed123")
        mock_set_tag.assert_any_call("litellm.requested_model", "gpt-4o")

    def test_no_tags_when_key_absent(self):
        """No key tags are set when key_alias and token are None (e.g. 401 path)."""
        user_key = self._make_user_api_key_dict(key_alias=None, token=None)

        with patch(
            "litellm.proxy.dd_span_tagger.set_active_span_tag"
        ) as mock_set_tag:
            DDSpanTagger.tag_request(
                user_api_key_dict=user_key,
                requested_model=None,
            )

        mock_set_tag.assert_not_called()

    def test_only_model_tagged_when_no_key_info(self):
        """requested_model is tagged even when there's no key info."""
        user_key = self._make_user_api_key_dict(key_alias=None, token=None)

        with patch(
            "litellm.proxy.dd_span_tagger.set_active_span_tag"
        ) as mock_set_tag:
            DDSpanTagger.tag_request(
                user_api_key_dict=user_key,
                requested_model="claude-3-5-sonnet",
            )

        mock_set_tag.assert_called_once_with("litellm.requested_model", "claude-3-5-sonnet")


class TestHasAttributeErrorInChain:
    """Tests for _has_attribute_error_in_chain helper."""

    def test_direct_attribute_error(self):
        exc = AttributeError("'str' object has no attribute 'get'")
        assert _has_attribute_error_in_chain(exc) is True

    def test_no_attribute_error(self):
        exc = ValueError("some other error")
        assert _has_attribute_error_in_chain(exc) is False

    def test_attribute_error_in_cause(self):
        inner = AttributeError("bad attribute")
        outer = RuntimeError("wrapper")
        outer.__cause__ = inner
        assert _has_attribute_error_in_chain(outer) is True

    def test_attribute_error_in_context(self):
        inner = AttributeError("bad attribute")
        outer = RuntimeError("wrapper")
        outer.__context__ = inner
        assert _has_attribute_error_in_chain(outer) is True

    def test_attribute_error_in_original_exception(self):
        inner = AttributeError("bad attribute")
        outer = RuntimeError("wrapper")
        outer.original_exception = inner  # type: ignore
        assert _has_attribute_error_in_chain(outer) is True

    def test_attribute_error_nested_two_levels(self):
        """Simulates the real failure: AttributeError -> OpenAIException -> APIConnectionError."""
        attr_err = AttributeError("'str' object has no attribute 'get'")
        mid = Exception("OpenAIException wrapper")
        mid.__context__ = attr_err
        outer = Exception("APIConnectionError wrapper")
        outer.__context__ = mid
        assert _has_attribute_error_in_chain(outer) is True

    def test_depth_limit_prevents_infinite_loop(self):
        """Ensure circular references don't cause infinite recursion."""
        exc_a = RuntimeError("a")
        exc_b = RuntimeError("b")
        exc_a.__context__ = exc_b
        exc_b.__context__ = exc_a  # circular
        assert _has_attribute_error_in_chain(exc_a) is False


class TestAawmRouteRollup:
    def test_route_rollup_header_and_subline_formatting(self):
        from datetime import datetime

        from litellm.proxy.aawm_route_logging import (
            AawmRouteRollupAccumulator,
            build_aawm_route_rollup_group_header_label,
        )

        now = datetime(2026, 6, 23, 1, 6, 52)
        accumulator = AawmRouteRollupAccumulator(interval_seconds=60)
        header = build_aawm_route_rollup_group_header_label(
            repository="litellm",
            client_product_label="codex-cli/0.141.0",
        )
        lines = accumulator.record(
            group_header_label=header,
            incoming_endpoint="/openai_passthrough/responses",
            outgoing_target="chatgpt.com/backend-api/codex/responses",
            model_label="gemini-3.5-flash-low(aawm-low)",
            turns=3,
            now=now,
        )
        flushed = accumulator.flush(force=True, now=now)
        assert lines == []
        assert len(flushed) == 2
        assert (
            flushed[0]
            == "20260623 01:06:52 litellm@Codex[0.141.0] /openai_passthrough/responses"
        )
        assert (
            flushed[1]
            == " - gemini-3.5-flash-low(aawm-low) - Turns: 3 -> chatgpt.com/backend-api/codex/responses"
        )

    def test_route_rollup_groups_destinations_under_local_endpoint(self):
        from datetime import datetime

        from litellm.proxy.aawm_route_logging import AawmRouteRollupAccumulator

        now = datetime(2026, 6, 23, 2, 51, 23)
        accumulator = AawmRouteRollupAccumulator(interval_seconds=60)
        header = "aawm@Codex[0.141.0]"
        endpoint = "/openai_passthrough/responses"

        accumulator.record(
            group_header_label=header,
            incoming_endpoint=endpoint,
            outgoing_target="daily-cloudcode-pa.googleapis.com/v1internal:streamGenerateContent",
            model_label="gemini-3.5-flash-low(aawm-low)",
            turns=7,
            now=now,
        )
        accumulator.record(
            group_header_label=header,
            incoming_endpoint=endpoint,
            outgoing_target="chatgpt.com/backend-api/codex/responses",
            model_label="gpt-5.5",
            turns=4,
            now=now,
        )

        flushed = accumulator.flush(force=True, now=now)
        assert flushed == [
            "20260623 02:51:23 aawm@Codex[0.141.0] /openai_passthrough/responses",
            (
                " - gemini-3.5-flash-low(aawm-low) - Turns: 7 -> "
                "daily-cloudcode-pa.googleapis.com/v1internal:streamGenerateContent"
            ),
            (
                " - gpt-5.5 - Turns: 4 -> "
                "chatgpt.com/backend-api/codex/responses"
            ),
        ]

    def test_route_rollup_status_latest_material_state_wins(self):
        from datetime import datetime

        from litellm.proxy.aawm_route_logging import AawmRouteRollupAccumulator

        now = datetime(2026, 6, 23, 1, 7, 0)
        accumulator = AawmRouteRollupAccumulator(interval_seconds=60)
        header = "litellm@Codex[0.141.0]"
        endpoint = "/openai_passthrough/responses"
        target = "chatgpt.com/backend-api/codex/responses"
        model = "gpt-5.5(aawm-low)"

        accumulator.record(
            group_header_label=header,
            incoming_endpoint=endpoint,
            outgoing_target=target,
            model_label=model,
            turns=1,
            status="Cooling Down",
            now=now,
        )
        accumulator.record(
            group_header_label=header,
            incoming_endpoint=endpoint,
            outgoing_target=target,
            model_label=model,
            turns=2,
            status="Exhausted",
            now=now,
        )
        flushed = accumulator.flush(force=True, now=now)
        assert (
            " - gpt-5.5(aawm-low) - Turns: 3 [Exhausted] -> "
            "chatgpt.com/backend-api/codex/responses"
        ) in flushed

    def test_route_rollup_interval_zero_disables_accumulation(self, monkeypatch):
        from litellm.proxy.aawm_route_logging import (
            AawmRouteRollupAccumulator,
            aawm_route_rollups_enabled,
            get_aawm_route_rollup_interval_seconds,
        )

        monkeypatch.setenv("AAWM_ROUTE_ROLLUP_INTERVAL_SECONDS", "0")
        assert get_aawm_route_rollup_interval_seconds() == 0
        assert aawm_route_rollups_enabled() is False

        accumulator = AawmRouteRollupAccumulator(interval_seconds=0)
        assert accumulator.record(
            group_header_label="litellm@Codex[0.141.0]",
            incoming_endpoint="/openai_passthrough/responses",
            outgoing_target="chatgpt.com/backend-api/codex/responses",
            model_label="gpt-5.5",
            turns=1,
        ) == []

    def test_route_rollup_early_flush_prefixes_header(self):
        from datetime import datetime

        from litellm.proxy.aawm_route_logging import AawmRouteRollupAccumulator

        now = datetime(2026, 6, 23, 1, 8, 12)
        accumulator = AawmRouteRollupAccumulator(
            interval_seconds=60,
            max_sublines=2,
        )
        header = "litellm@Codex[0.141.0]"
        endpoint = "/openai_passthrough/responses"
        target = "chatgpt.com/backend-api/codex/responses"

        accumulator.record(
            group_header_label=header,
            incoming_endpoint=endpoint,
            outgoing_target=target,
            model_label="gemini-3.5-flash-low(aawm-low)",
            turns=1,
            now=now,
        )
        accumulator.record(
            group_header_label=header,
            incoming_endpoint=endpoint,
            outgoing_target=target,
            model_label="gpt-5.5(aawm-low)",
            turns=1,
            now=now,
        )
        early_lines = accumulator.record(
            group_header_label=header,
            incoming_endpoint=endpoint,
            outgoing_target=target,
            model_label="deepseek-v4-flash(aawm-low)",
            turns=1,
            now=now,
        )
        assert any(line.startswith("20260623 01:08:12 [EARLY]") for line in early_lines)
        assert "\x1b" not in "\n".join(early_lines)

    def test_route_rollup_output_has_no_ansi_by_default(self):
        from datetime import datetime

        from litellm.proxy.aawm_route_logging import AawmRouteRollupAccumulator

        now = datetime(2026, 6, 23, 1, 6, 43)
        accumulator = AawmRouteRollupAccumulator(interval_seconds=60)
        accumulator.record(
            group_header_label="litellm@Codex[0.141.0]",
            incoming_endpoint="/openai_passthrough/responses",
            outgoing_target="chatgpt.com/backend-api/codex/responses",
            model_label="gemini-3.5-flash-low(aawm-low)",
            turns=0,
            status="Degraded",
            now=now,
        )
        flushed = accumulator.flush(force=True, now=now)
        rendered = "\n".join(flushed)
        assert "[Degraded]" in rendered
        assert "\x1b" not in rendered

    def test_clear_and_flush_aawm_route_rollups_helpers(self):
        from datetime import datetime

        from litellm.proxy.aawm_route_logging import (
            clear_aawm_route_rollups,
            flush_aawm_route_rollups,
            get_aawm_route_rollup_accumulator,
        )

        now = datetime(2026, 6, 23, 2, 0, 0)
        accumulator = get_aawm_route_rollup_accumulator()
        clear_aawm_route_rollups()
        accumulator.record(
            group_header_label="aegis@Claude[2.1.178]",
            incoming_endpoint="/anthropic/v1/messages?beta=true",
            outgoing_target="api.anthropic.com/v1/messages",
            model_label="orchestrator.claude-opus-4-8",
            turns=2,
            now=now,
        )
        flushed = flush_aawm_route_rollups(force=True, now=now)
        assert any("aegis@Claude[2.1.178]" in line for line in flushed)
        clear_aawm_route_rollups()
        assert flush_aawm_route_rollups(force=True, now=now) == []

    def test_route_rollup_flush_due_without_subsequent_record(self):
        from datetime import datetime

        from litellm.proxy.aawm_route_logging import AawmRouteRollupAccumulator

        now = datetime(2026, 6, 23, 2, 30, 0)
        accumulator = AawmRouteRollupAccumulator(interval_seconds=60)
        accumulator.record(
            group_header_label="litellm@Codex[0.141.0]",
            incoming_endpoint="/openai_passthrough/responses",
            outgoing_target="chatgpt.com/backend-api/codex/responses",
            model_label="gpt-5.5",
            turns=2,
            now=now,
        )
        accumulator._last_flush_monotonic = 0.0
        flushed = accumulator.flush_due(now=now, monotonic_now=61.0)
        assert len(flushed) == 2
        assert "Turns: 2" in flushed[1]
        assert accumulator.flush(force=True, now=now) == []

    def test_route_rollup_interval_tick_flushes_idle_bucket(self, monkeypatch):
        from datetime import datetime

        from litellm.proxy.aawm_route_logging import (
            _set_aawm_route_rollup_monotonic_now_for_tests,
            _stop_aawm_route_rollup_flush_worker,
            _tick_aawm_route_rollup_interval_flush,
            clear_aawm_route_rollups,
            get_aawm_route_rollup_accumulator,
        )

        monkeypatch.setenv("AAWM_ROUTE_ROLLUP_INTERVAL_SECONDS", "60")
        clear_aawm_route_rollups()
        now = datetime(2026, 6, 23, 2, 31, 0)

        def fake_monotonic() -> float:
            return 61.0

        _set_aawm_route_rollup_monotonic_now_for_tests(fake_monotonic)
        try:
            accumulator = get_aawm_route_rollup_accumulator()
            accumulator.record(
                group_header_label="litellm@Codex[0.141.0]",
                incoming_endpoint="/openai_passthrough/responses",
                outgoing_target="chatgpt.com/backend-api/codex/responses",
                model_label="gpt-5.5",
                turns=1,
                now=now,
            )
            accumulator._last_flush_monotonic = 0.0
            _tick_aawm_route_rollup_interval_flush()
            assert flush_aawm_route_rollups(force=True, now=now) == []
        finally:
            _set_aawm_route_rollup_monotonic_now_for_tests(None)
            _stop_aawm_route_rollup_flush_worker()
            clear_aawm_route_rollups()

    def test_route_rollup_interval_zero_skips_background_flush_worker(
        self,
        monkeypatch,
    ):
        from litellm.proxy.aawm_route_logging import (
            _stop_aawm_route_rollup_flush_worker,
            clear_aawm_route_rollups,
            get_aawm_route_rollup_accumulator,
        )

        monkeypatch.setenv("AAWM_ROUTE_ROLLUP_INTERVAL_SECONDS", "0")
        _stop_aawm_route_rollup_flush_worker()
        clear_aawm_route_rollups()
        get_aawm_route_rollup_accumulator()
        assert aawm_route_logging._aawm_route_rollup_flush_thread is None
