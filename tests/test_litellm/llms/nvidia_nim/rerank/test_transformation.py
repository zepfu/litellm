"""
D1-556: Focused tests for NVIDIA NIM rerank transformation.

Requires the committed D1-542 helper contract:
- litellm.litellm_core_utils.param_adaptation with renamed/provider_rename
- litellm.types.utils.get_nvidia_nim_model_metadata
- litellm.exceptions.UnsupportedParamsError (canonical, status 400)

Covers:
- Outbound request snapshots for both endpoint families
- top_n -> top_k mapping with conflict detection (renamed/provider_rename)
- truncate validation (valid/invalid)
- Unsupported param handling via real caller path (strict/drop, names-only)
- return_documents: never sent to provider, local response shaping sync/async
- Catalog-driven endpoint/body_model with unknown-model derivation
- Bounded provider allowlist (unknown/infra kwargs rejected or dropped)
- Permissive invalid truncate (drop + record) vs strict reject
- return_documents: local response control exempt from provider-omission records
- Model alias/cleanup/shared endpoint behavior
- Shared-instance concurrency safety
- Metadata schema (flat keys, not nested)
- Canonical exception type and status 400
"""

import concurrent.futures
import datetime
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import litellm
from litellm.exceptions import UnsupportedParamsError
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.param_adaptation import (
    PROVIDER_PARAMETER_ADAPTATIONS_METADATA_KEY,
    PROVIDER_PARAMETER_ADAPTATIONS_TRUNCATED_COUNT_METADATA_KEY,
)
from litellm.llms.nvidia_nim.rerank.transformation import (
    _CTX_KEY,
    NvidiaNimRerankConfig,
)
from litellm.llms.nvidia_nim.rerank.ranking_transformation import (
    NvidiaNimRankingConfig,
)
from litellm.types.rerank import RerankResponse
from litellm.types.utils import NvidiaNimModelMetadata, NvidiaNimRerankMetadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_response(rankings=None, usage=None):
    """Create a mock httpx.Response with NVIDIA NIM rerank JSON."""
    if rankings is None:
        rankings = [{"index": 0, "logit": 0.95}, {"index": 1, "logit": 0.75}]
    body = {"rankings": rankings}
    if usage:
        body["usage"] = usage
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.json.return_value = body
    mock_resp.status_code = 200
    mock_resp.text = json.dumps(body)
    mock_resp.headers = {"content-type": "application/json"}
    return mock_resp


def _make_logging_obj():
    mock_logging = MagicMock()
    mock_logging.pre_call = MagicMock()
    mock_logging.post_call = MagicMock()
    return mock_logging


def _make_real_logging_obj(
    sync_callback=None,
    async_callback=None,
):
    return Logging(
        model="nvidia/llama-3_2-nv-rerankqa-1b-v2",
        messages="test query",
        stream=False,
        call_type="rerank",
        start_time=datetime.datetime.now(),
        litellm_call_id="test-rerank-call",
        function_id="test-rerank-function",
        dynamic_success_callbacks=(
            [sync_callback] if sync_callback is not None else None
        ),
        dynamic_async_success_callbacks=(
            [async_callback] if async_callback is not None else None
        ),
    )


class _CaptureCallback(CustomLogger):
    def __init__(self):
        super().__init__()
        self.sync_kwargs = None
        self.async_kwargs = None

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.sync_kwargs = kwargs

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.async_kwargs = kwargs


def _assert_recursive_absence(value, sentinels, forbidden_keys, seen=None):
    if seen is None:
        seen = set()
    value_id = id(value)
    if value_id in seen:
        return
    seen.add(value_id)

    if isinstance(value, dict):
        for key, nested_value in value.items():
            assert str(key) not in forbidden_keys
            _assert_recursive_absence(
                nested_value, sentinels, forbidden_keys, seen=seen
            )
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            _assert_recursive_absence(item, sentinels, forbidden_keys, seen=seen)
        return

    value_text = str(value)
    for sentinel in sentinels:
        assert sentinel not in value_text


def _sync_rerank(**kwargs):
    """Call litellm.rerank with mocked HTTP, return (response, sent_data)."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.json.return_value = {
        "rankings": [{"index": 0, "logit": 0.9}, {"index": 1, "logit": 0.5}]
    }
    mock_response.status_code = 200
    mock_response.text = "{}"
    mock_response.headers = {}

    with patch(
        "litellm.llms.custom_httpx.http_handler.HTTPHandler.post",
        return_value=mock_response,
    ) as mock_post:
        api_key = kwargs.pop("api_key", "fake-key")
        response = litellm.rerank(
            model="nvidia_nim/nvidia/llama-3_2-nv-rerankqa-1b-v2",
            query="test query",
            documents=["first doc", "second doc"],
            api_key=api_key,
            **kwargs,
        )
        sent_data = json.loads(mock_post.call_args.kwargs["data"])
        return response, sent_data


async def _async_rerank(**kwargs):
    """Call litellm.arerank with mocked HTTP, return (response, sent_data)."""
    mock_response = AsyncMock()
    mock_response.json = lambda: {
        "rankings": [{"index": 0, "logit": 0.9}, {"index": 1, "logit": 0.5}]
    }
    mock_response.headers = {"content-type": "application/json"}
    mock_response.status_code = 200

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        return_value=mock_response,
    ) as mock_post:
        api_key = kwargs.pop("api_key", "fake-key")
        response = await litellm.arerank(
            model="nvidia_nim/nvidia/llama-3_2-nv-rerankqa-1b-v2",
            query="test query",
            documents=["first doc", "second doc"],
            api_key=api_key,
            **kwargs,
        )
        sent_data = json.loads(mock_post.call_args.kwargs["data"])
        return response, sent_data


def _catalog_rerank_meta(endpoint_path: str, body_model: str):
    """Build a NvidiaNimModelMetadata with rerank entry for patching."""
    return NvidiaNimModelMetadata(
        rerank=NvidiaNimRerankMetadata(
            endpoint_path=endpoint_path,
            body_model=body_model,
        )
    )


# ---------------------------------------------------------------------------
# Endpoint family: /v1/retrieval/{model}/reranking (default derived)
# ---------------------------------------------------------------------------


class TestRerankingEndpointFamily:
    def setup_method(self):
        self.config = NvidiaNimRerankConfig()

    def test_default_url_construction(self):
        url = self.config.get_complete_url(
            api_base=None, model="nvidia_nim/nvidia/llama-3_2-nv-rerankqa-1b-v2"
        )
        assert (
            url
            == "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-3_2-nv-rerankqa-1b-v2/reranking"
        )

    def test_custom_api_base(self):
        url = self.config.get_complete_url(
            api_base="https://custom.api.example.com/v1",
            model="nvidia_nim/nvidia/llama-3_2-nv-rerankqa-1b-v2",
        )
        assert (
            url
            == "https://custom.api.example.com/v1/retrieval/nvidia/llama-3_2-nv-rerankqa-1b-v2/reranking"
        )

    def test_full_url_passthrough(self):
        full_url = "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-3_2-nv-rerankqa-1b-v2/reranking"
        url = self.config.get_complete_url(api_base=full_url, model="ignored")
        assert url == full_url

    def test_outbound_request_snapshot(self):
        """Full outbound body snapshot for /v1/retrieval endpoint."""
        optional_params = self.config.map_cohere_rerank_params(
            non_default_params={"truncate": "END"},
            model="nvidia_nim/nvidia/llama-3_2-nv-rerankqa-1b-v2",
            drop_params=False,
            query="What is GPU bandwidth?",
            documents=["H100 has 3TB/s", "A100 has 2TB/s"],
            top_n=2,
        )
        request_body = self.config.transform_rerank_request(
            model="nvidia_nim/nvidia/llama-3_2-nv-rerankqa-1b-v2",
            optional_rerank_params=optional_params,
            headers={},
        )
        assert request_body == {
            "model": "nvidia/llama-3.2-nv-rerankqa-1b-v2",
            "query": {"text": "What is GPU bandwidth?"},
            "passages": [{"text": "H100 has 3TB/s"}, {"text": "A100 has 2TB/s"}],
            "top_k": 2,
            "truncate": "END",
        }


# ---------------------------------------------------------------------------
# Endpoint family: /v1/ranking
# ---------------------------------------------------------------------------


class TestRankingEndpointFamily:
    def setup_method(self):
        self.config = NvidiaNimRankingConfig()

    def test_ranking_url_construction(self):
        url = self.config.get_complete_url(
            api_base=None, model="nvidia_nim/ranking/nvidia/llama-3.2-nv-rerankqa-1b-v2"
        )
        assert url == "https://ai.api.nvidia.com/v1/ranking"

    def test_ranking_url_already_ranking(self):
        url = self.config.get_complete_url(
            api_base="https://ai.api.nvidia.com/v1/ranking", model="whatever"
        )
        assert url == "https://ai.api.nvidia.com/v1/ranking"

    def test_outbound_request_snapshot_ranking(self):
        """Full outbound body snapshot for /v1/ranking endpoint."""
        optional_params = self.config.map_cohere_rerank_params(
            non_default_params={},
            model="nvidia_nim/ranking/nvidia/llama-3.2-nv-rerankqa-1b-v2",
            drop_params=False,
            query="Which GPU is faster?",
            documents=["H100 is fast", "A100 is slower"],
            top_n=1,
        )
        request_body = self.config.transform_rerank_request(
            model="nvidia_nim/ranking/nvidia/llama-3.2-nv-rerankqa-1b-v2",
            optional_rerank_params=optional_params,
            headers={},
        )
        assert request_body == {
            "model": "nvidia/llama-3.2-nv-rerankqa-1b-v2",
            "query": {"text": "Which GPU is faster?"},
            "passages": [{"text": "H100 is fast"}, {"text": "A100 is slower"}],
            "top_k": 1,
        }


# ---------------------------------------------------------------------------
# Catalog-driven endpoint/body_model with unknown-model fallback
# ---------------------------------------------------------------------------


class TestCatalogDrivenMetadata:
    def setup_method(self):
        self.config = NvidiaNimRerankConfig()

    def test_catalog_endpoint_path_used_for_url(self):
        """When catalog provides endpoint_path, it is used for URL construction."""
        meta = _catalog_rerank_meta(
            endpoint_path="/v1/retrieval/nvidia/reranking",
            body_model="nv-rerank-qa-mistral-4b:1",
        )
        with patch(
            "litellm.llms.nvidia_nim.rerank.transformation.get_nvidia_nim_model_metadata",
            return_value=meta,
        ):
            url = self.config.get_complete_url(
                api_base=None, model="nvidia_nim/nvidia/rerank-qa-mistral-4b"
            )
            assert url == "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"
        assert "//" not in url.split("://", 1)[1]

    def test_catalog_body_model_used_in_request(self):
        """When catalog provides body_model, it is used in the request body."""
        meta = _catalog_rerank_meta(
            endpoint_path="/v1/retrieval/nvidia/reranking",
            body_model="nv-rerank-qa-mistral-4b:1",
        )
        with patch(
            "litellm.llms.nvidia_nim.rerank.transformation.get_nvidia_nim_model_metadata",
            return_value=meta,
        ):
            body_name = self.config._get_model_name_for_body(
                "nvidia_nim/nvidia/rerank-qa-mistral-4b"
            )
            assert body_name == "nv-rerank-qa-mistral-4b:1"

    def test_unknown_model_falls_back_to_derived_url(self):
        """Unknown models (no catalog entry) use derived /v1/retrieval/{model}/reranking."""
        with patch(
            "litellm.llms.nvidia_nim.rerank.transformation.get_nvidia_nim_model_metadata",
            return_value=NvidiaNimModelMetadata(),
        ):
            url = self.config.get_complete_url(
                api_base=None, model="nvidia_nim/nvidia/some-unknown-model"
            )
            assert (
                url
                == "https://ai.api.nvidia.com/v1/retrieval/nvidia/some-unknown-model/reranking"
            )

    def test_unknown_model_falls_back_to_derived_body_name(self):
        """Unknown models use underscore-to-dot derived body name."""
        with patch(
            "litellm.llms.nvidia_nim.rerank.transformation.get_nvidia_nim_model_metadata",
            return_value=NvidiaNimModelMetadata(),
        ):
            body_name = self.config._get_model_name_for_body(
                "nvidia_nim/nvidia/llama-3_2-nv-rerankqa-1b-v2"
            )
            assert body_name == "nvidia/llama-3.2-nv-rerankqa-1b-v2"

    def test_catalog_exception_propagates(self):
        """Malformed or unavailable catalog metadata must not fail open."""
        with patch(
            "litellm.llms.nvidia_nim.rerank.transformation.get_nvidia_nim_model_metadata",
            side_effect=RuntimeError("catalog unavailable"),
        ):
            with pytest.raises(RuntimeError, match="catalog unavailable"):
                self.config.get_complete_url(
                    api_base=None, model="nvidia_nim/nvidia/some-model"
                )

    def test_catalog_metadata_used_for_url_and_body(self):
        """When catalog provides rerank metadata, it is used for URL and body model."""
        meta = _catalog_rerank_meta(
            endpoint_path="/v1/custom/endpoint",
            body_model="custom-body-model",
        )
        with patch(
            "litellm.llms.nvidia_nim.rerank.transformation.get_nvidia_nim_model_metadata",
            return_value=meta,
        ):
            url = self.config.get_complete_url(
                api_base=None, model="nvidia_nim/nvidia/rerank-qa-mistral-4b"
            )
            assert url == "https://ai.api.nvidia.com/v1/custom/endpoint"
            assert "//" not in url.split("://", 1)[1]
            body = self.config._get_model_name_for_body(
                "nvidia_nim/nvidia/rerank-qa-mistral-4b"
            )
            assert body == "custom-body-model"


# ---------------------------------------------------------------------------
# URL joining: no double slashes with catalog endpoint_path
# ---------------------------------------------------------------------------


class TestUrlJoiningNoDoubleSlash:
    """Catalog endpoint_path begins with /v1/...; joining must not produce //v1."""

    def setup_method(self):
        self.config = NvidiaNimRerankConfig()

    def test_catalog_leading_slash_no_double_slash(self):
        """endpoint_path='/v1/retrieval/nvidia/reranking' joins without //v1."""
        meta = _catalog_rerank_meta(
            endpoint_path="/v1/retrieval/nvidia/reranking",
            body_model="some-model",
        )
        with patch(
            "litellm.llms.nvidia_nim.rerank.transformation.get_nvidia_nim_model_metadata",
            return_value=meta,
        ):
            url = self.config.get_complete_url(
                api_base="https://ai.api.nvidia.com", model="nvidia_nim/x"
            )
            assert url == "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"
            # Exact: no double slash after scheme
            assert "//" not in url.split("://", 1)[1]

    def test_catalog_no_leading_slash_also_works(self):
        """endpoint_path='v1/retrieval/nvidia/reranking' (no leading /) also joins correctly."""
        meta = _catalog_rerank_meta(
            endpoint_path="v1/retrieval/nvidia/reranking",
            body_model="some-model",
        )
        with patch(
            "litellm.llms.nvidia_nim.rerank.transformation.get_nvidia_nim_model_metadata",
            return_value=meta,
        ):
            url = self.config.get_complete_url(
                api_base="https://ai.api.nvidia.com", model="nvidia_nim/x"
            )
            assert url == "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"
            assert "//" not in url.split("://", 1)[1]

    def test_catalog_with_trailing_slash_api_base(self):
        """api_base with trailing slash + leading-slash endpoint_path: no //."""
        meta = _catalog_rerank_meta(
            endpoint_path="/v1/retrieval/nvidia/reranking",
            body_model="some-model",
        )
        with patch(
            "litellm.llms.nvidia_nim.rerank.transformation.get_nvidia_nim_model_metadata",
            return_value=meta,
        ):
            url = self.config.get_complete_url(
                api_base="https://ai.api.nvidia.com/", model="nvidia_nim/x"
            )
            assert url == "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"
            assert "//" not in url.split("://", 1)[1]

    def test_ranking_endpoint_no_double_slash(self):
        """/v1/ranking endpoint must not produce double slashes."""
        config = NvidiaNimRankingConfig()
        url = config.get_complete_url(
            api_base="https://ai.api.nvidia.com",
            model="nvidia_nim/ranking/nvidia/llama-3.2-nv-rerankqa-1b-v2",
        )
        assert url == "https://ai.api.nvidia.com/v1/ranking"
        assert "//" not in url.split("://", 1)[1]


# ---------------------------------------------------------------------------
# top_n -> top_k mapping and conflict detection (renamed/provider_rename)
# ---------------------------------------------------------------------------


class TestTopNTopKMapping:
    def setup_method(self):
        self.config = NvidiaNimRerankConfig()

    def test_top_n_maps_to_top_k(self):
        params = self.config.map_cohere_rerank_params(
            non_default_params={},
            model="nvidia_nim/test-model",
            drop_params=False,
            query="q",
            documents=["d"],
            top_n=5,
        )
        assert params["top_k"] == 5
        assert "top_n" not in params

    def test_top_n_none_omits_top_k(self):
        params = self.config.map_cohere_rerank_params(
            non_default_params={},
            model="nvidia_nim/test-model",
            drop_params=False,
            query="q",
            documents=["d"],
            top_n=None,
        )
        assert "top_k" not in params

    def test_top_k_in_request_body(self):
        optional_params = self.config.map_cohere_rerank_params(
            non_default_params={},
            model="nvidia_nim/test-model",
            drop_params=False,
            query="q",
            documents=["d1", "d2"],
            top_n=3,
        )
        body = self.config.transform_rerank_request(
            model="nvidia_nim/test-model",
            optional_rerank_params=optional_params,
            headers={},
        )
        assert body["top_k"] == 3

    def test_top_n_rename_recorded_as_renamed_provider_rename(self):
        """top_n -> top_k is recorded as action=renamed reason=provider_rename."""
        params = self.config.map_cohere_rerank_params(
            non_default_params={},
            model="nvidia_nim/test-model",
            drop_params=False,
            query="q",
            documents=["d"],
            top_n=5,
        )
        ctx = params[_CTX_KEY]
        rename_records = [
            r
            for r in ctx["adaptation_records"]
            if r["name"] == "top_n"
            and r["action"] == "renamed"
            and r["reason"] == "provider_rename"
        ]
        assert len(rename_records) == 1

    def test_equal_top_n_top_k_dedup_records_rename(self):
        """Equal top_n and direct top_k: dedup, record as renamed."""
        params = self.config.map_cohere_rerank_params(
            non_default_params={"top_k": 3},
            model="nvidia_nim/test-model",
            drop_params=False,
            query="q",
            documents=["d"],
            top_n=3,
        )
        assert params["top_k"] == 3
        ctx = params[_CTX_KEY]
        rename_records = [
            r
            for r in ctx["adaptation_records"]
            if r["name"] == "top_n" and r["action"] == "renamed"
        ]
        assert len(rename_records) == 1

    def test_differing_top_n_top_k_strict_raises(self):
        """Differing top_n and top_k in strict mode raises UnsupportedParamsError."""
        with pytest.raises(UnsupportedParamsError) as exc_info:
            self.config.map_cohere_rerank_params(
                non_default_params={"top_k": 5},
                model="nvidia_nim/test-model",
                drop_params=False,
                query="q",
                documents=["d"],
                top_n=3,
            )
        msg = str(exc_info.value)
        assert "top_n" in msg
        assert "top_k" in msg

    def test_differing_top_n_top_k_drop_prefers_top_k_records_drop(self):
        """Differing top_n and top_k in drop mode: prefer top_k, record drop."""
        params = self.config.map_cohere_rerank_params(
            non_default_params={"top_k": 5},
            model="nvidia_nim/test-model",
            drop_params=True,
            query="q",
            documents=["d"],
            top_n=3,
        )
        assert params["top_k"] == 5
        ctx = params[_CTX_KEY]
        drop_records = [
            r
            for r in ctx["adaptation_records"]
            if r["name"] == "top_n"
            and r["action"] == "dropped"
            and r["reason"] == "unsupported_param"
        ]
        assert len(drop_records) == 1

    def test_direct_top_k_only(self):
        """Direct top_k without top_n passes through without adaptation record."""
        params = self.config.map_cohere_rerank_params(
            non_default_params={"top_k": 7},
            model="nvidia_nim/test-model",
            drop_params=False,
            query="q",
            documents=["d"],
            top_n=None,
        )
        assert params["top_k"] == 7
        ctx = params[_CTX_KEY]
        top_n_records = [r for r in ctx["adaptation_records"] if r["name"] == "top_n"]
        assert len(top_n_records) == 0


# ---------------------------------------------------------------------------
# truncate validation
# ---------------------------------------------------------------------------


class TestTruncateValidation:
    def setup_method(self):
        self.config = NvidiaNimRerankConfig()

    def test_valid_truncate_none(self):
        body = self.config.transform_rerank_request(
            model="nvidia_nim/test-model",
            optional_rerank_params={
                "query": "q",
                "documents": ["d"],
                "truncate": "NONE",
            },
            headers={},
        )
        assert body["truncate"] == "NONE"

    def test_valid_truncate_end(self):
        body = self.config.transform_rerank_request(
            model="nvidia_nim/test-model",
            optional_rerank_params={
                "query": "q",
                "documents": ["d"],
                "truncate": "END",
            },
            headers={},
        )
        assert body["truncate"] == "END"

    def test_invalid_truncate_raises_unsupported_params_error(self):
        with pytest.raises(UnsupportedParamsError) as exc_info:
            self.config.transform_rerank_request(
                model="nvidia_nim/test-model",
                optional_rerank_params={
                    "query": "q",
                    "documents": ["d"],
                    "truncate": "MIDDLE",
                },
                headers={},
            )
        assert "truncate" in str(exc_info.value)

    def test_truncate_not_in_body_when_absent(self):
        body = self.config.transform_rerank_request(
            model="nvidia_nim/test-model",
            optional_rerank_params={"query": "q", "documents": ["d"]},
            headers={},
        )
        assert "truncate" not in body


# ---------------------------------------------------------------------------
# Unsupported param handling via real caller path
# ---------------------------------------------------------------------------


class TestUnsupportedParamHandling:
    """Tests use the real litellm.rerank caller path with named params."""

    def test_strict_rejects_rank_fields_via_caller(self):
        """rank_fields passed as named param raises canonical UnsupportedParamsError."""
        with pytest.raises(UnsupportedParamsError) as exc_info:
            litellm.rerank(
                model="nvidia_nim/nvidia/llama-3_2-nv-rerankqa-1b-v2",
                query="q",
                documents=["d"],
                rank_fields=["text"],
                api_key="fake-key",
            )
        assert exc_info.value.status_code == 400
        msg = str(exc_info.value)
        assert "rank_fields" in msg
        # Names-only: no values in error
        assert "text" not in msg

    def test_strict_rejects_max_chunks_per_doc_via_caller(self):
        with pytest.raises(UnsupportedParamsError) as exc_info:
            litellm.rerank(
                model="nvidia_nim/nvidia/llama-3_2-nv-rerankqa-1b-v2",
                query="q",
                documents=["d"],
                max_chunks_per_doc=5,
                api_key="fake-key",
            )
        assert exc_info.value.status_code == 400
        assert "max_chunks_per_doc" in str(exc_info.value)

    def test_strict_error_names_only_no_values(self):
        """Error message names params but does not include their values."""
        try:
            litellm.rerank(
                model="nvidia_nim/nvidia/llama-3_2-nv-rerankqa-1b-v2",
                query="q",
                documents=["d"],
                rank_fields=["secret_field_name"],
                api_key="fake-key",
            )
            pytest.fail("Should have raised")
        except UnsupportedParamsError as e:
            msg = str(e)
            assert "rank_fields" in msg
            assert "secret_field_name" not in msg

    def test_request_level_drop_params(self):
        """drop_params=True at request level silently drops unsupported params."""
        response, sent_data = _sync_rerank(
            rank_fields=["text"],
            max_chunks_per_doc=5,
            drop_params=True,
        )
        assert "rank_fields" not in sent_data
        assert "max_chunks_per_doc" not in sent_data
        assert response.results is not None

    def test_global_drop_params(self):
        """litellm.drop_params=True globally drops unsupported params."""
        old = litellm.drop_params
        try:
            litellm.drop_params = True
            response, sent_data = _sync_rerank(rank_fields=["text"])
            assert "rank_fields" not in sent_data
            assert response.results is not None
        finally:
            litellm.drop_params = old

    def test_drop_records_names_only_in_metadata(self):
        """Dropped params produce names-only adaptation records in metadata."""
        response, _ = _sync_rerank(
            rank_fields=["text"],
            max_chunks_per_doc=5,
            drop_params=True,
        )
        hp = response._hidden_params
        assert PROVIDER_PARAMETER_ADAPTATIONS_METADATA_KEY in hp
        records = hp[PROVIDER_PARAMETER_ADAPTATIONS_METADATA_KEY]
        names = {r["name"] for r in records}
        assert "rank_fields" in names
        assert "max_chunks_per_doc" in names
        for r in records:
            assert set(r.keys()) == {"name", "action", "reason"}
            assert r["action"] == "dropped"
            assert r["reason"] == "unsupported_param"

    def test_no_metadata_when_no_unsupported_params(self):
        response, _ = _sync_rerank()
        hp = response._hidden_params
        assert PROVIDER_PARAMETER_ADAPTATIONS_METADATA_KEY not in hp


# ---------------------------------------------------------------------------
# Metadata schema: flat keys, not nested
# ---------------------------------------------------------------------------


class TestMetadataSchema:
    def test_flat_metadata_keys(self):
        """Adaptation metadata uses flat keys, not nested dict."""
        response, _ = _sync_rerank(
            rank_fields=["text"],
            drop_params=True,
        )
        hp = response._hidden_params
        records = hp[PROVIDER_PARAMETER_ADAPTATIONS_METADATA_KEY]
        assert isinstance(records, list)
        assert all(isinstance(r, dict) for r in records)
        truncated = hp[PROVIDER_PARAMETER_ADAPTATIONS_TRUNCATED_COUNT_METADATA_KEY]
        assert isinstance(truncated, int)
        assert not isinstance(records, dict)

    def test_no_nested_provider_parameter_adaptations_key(self):
        response, _ = _sync_rerank(
            rank_fields=["text"],
            drop_params=True,
        )
        hp = response._hidden_params
        records = hp[PROVIDER_PARAMETER_ADAPTATIONS_METADATA_KEY]
        for r in records:
            assert "provider_parameter_adaptations" not in r


# ---------------------------------------------------------------------------
# return_documents: provider-body omission and local shaping
# ---------------------------------------------------------------------------


class TestReturnDocuments:
    def test_return_documents_never_in_provider_body(self):
        _, sent_data = _sync_rerank(return_documents=True)
        assert "return_documents" not in sent_data

    def test_return_documents_false_never_in_provider_body(self):
        _, sent_data = _sync_rerank(return_documents=False)
        assert "return_documents" not in sent_data

    def test_return_documents_true_includes_documents_sync(self):
        response, _ = _sync_rerank(return_documents=True)
        assert response.results is not None
        assert response.results[0].get("document") == {"text": "first doc"}

    def test_return_documents_false_strips_documents_sync(self):
        response, _ = _sync_rerank(return_documents=False)
        assert response.results is not None
        for result in response.results:
            assert "document" not in result

    @pytest.mark.asyncio()
    async def test_return_documents_false_strips_documents_async(self):
        response, sent_data = await _async_rerank(return_documents=False)
        assert "return_documents" not in sent_data
        assert response.results is not None
        for result in response.results:
            assert "document" not in result

    @pytest.mark.asyncio()
    async def test_return_documents_true_includes_documents_async(self):
        response, _ = await _async_rerank(return_documents=True)
        assert response.results is not None
        assert response.results[0].get("document") == {"text": "first doc"}

    def test_return_documents_default_includes_documents(self):
        response, _ = _sync_rerank()
        assert response.results is not None
        assert response.results[0].get("document") == {"text": "first doc"}

    def test_ctx_key_not_in_provider_body(self):
        _, sent_data = _sync_rerank(return_documents=False, drop_params=True)
        assert _CTX_KEY not in sent_data


# ---------------------------------------------------------------------------
# Model alias / cleanup / shared endpoint behavior (catalog-driven)
# ---------------------------------------------------------------------------


class TestModelAliasAndCleanup:
    def setup_method(self):
        self.config = NvidiaNimRerankConfig()

    def test_strip_nvidia_nim_prefix(self):
        assert (
            self.config._get_clean_model_name("nvidia_nim/nvidia/model")
            == "nvidia/model"
        )
        assert self.config._get_clean_model_name("nvidia/model") == "nvidia/model"

    def test_shared_endpoint_url_via_catalog(self):
        """Shared endpoint models use catalog endpoint_path."""
        meta = _catalog_rerank_meta(
            endpoint_path="/v1/retrieval/nvidia/reranking",
            body_model="nv-rerank-qa-mistral-4b:1",
        )
        with patch(
            "litellm.llms.nvidia_nim.rerank.transformation.get_nvidia_nim_model_metadata",
            return_value=meta,
        ):
            url = self.config.get_complete_url(
                api_base=None, model="nvidia_nim/nvidia/rerank-qa-mistral-4b"
            )
            assert url == "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"

    def test_body_model_alias_via_catalog(self):
        """Body model alias resolved through catalog body_model."""
        meta = _catalog_rerank_meta(
            endpoint_path="/v1/retrieval/nvidia/reranking",
            body_model="nv-rerank-qa-mistral-4b:1",
        )
        with patch(
            "litellm.llms.nvidia_nim.rerank.transformation.get_nvidia_nim_model_metadata",
            return_value=meta,
        ):
            body_name = self.config._get_model_name_for_body(
                "nvidia_nim/nvidia/rerank-qa-mistral-4b"
            )
            assert body_name == "nv-rerank-qa-mistral-4b:1"

    def test_non_shared_endpoint_url_derived(self):
        """Non-catalog models use derived URL."""
        with patch(
            "litellm.llms.nvidia_nim.rerank.transformation.get_nvidia_nim_model_metadata",
            return_value=NvidiaNimModelMetadata(),
        ):
            url = self.config.get_complete_url(
                api_base=None, model="nvidia_nim/nvidia/llama-3_2-nv-rerankqa-1b-v2"
            )
            assert (
                url
                == "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-3_2-nv-rerankqa-1b-v2/reranking"
            )


# ---------------------------------------------------------------------------
# Policy metadata resolver
# ---------------------------------------------------------------------------


class TestPolicyMetadata:
    def test_policy_metadata_structure(self):
        meta = NvidiaNimRerankConfig.get_rerank_policy_metadata()
        assert meta["provider"] == "nvidia_nim"
        assert "top_n" in meta["supported_cohere_params"]
        assert "return_documents" in meta["supported_cohere_params"]
        assert "rank_fields" in meta["unsupported_cohere_params"]
        assert meta["param_renames"] == {"top_n": "top_k"}
        assert meta["local_only_params"] == ["return_documents"]
        assert "NONE" in meta["valid_truncate_values"]
        assert "END" in meta["valid_truncate_values"]

    def test_policy_metadata_excludes_model_specific_data(self):
        meta = NvidiaNimRerankConfig.get_rerank_policy_metadata()
        assert "SHARED_RERANKING_ENDPOINT_MODELS" not in meta
        assert "BODY_MODEL_ALIASES" not in meta


# ---------------------------------------------------------------------------
# Canonical exception type and status 400
# ---------------------------------------------------------------------------


class TestExceptionTypes:
    def test_canonical_unsupported_params_error_is_bad_request(self):
        """UnsupportedParamsError is a BadRequestError subclass with status 400."""
        from litellm.exceptions import BadRequestError

        assert issubclass(UnsupportedParamsError, BadRequestError)
        err = UnsupportedParamsError(message="test")
        assert err.status_code == 400

    def test_unsupported_params_error_names_only(self):
        err = UnsupportedParamsError(
            message="Unsupported parameter(s): rank_fields, max_chunks_per_doc"
        )
        assert "rank_fields" in str(err)
        assert "max_chunks_per_doc" in str(err)

    def test_invalid_truncate_raises_canonical_error(self):
        config = NvidiaNimRerankConfig()
        with pytest.raises(UnsupportedParamsError):
            config.transform_rerank_request(
                model="nvidia_nim/test-model",
                optional_rerank_params={
                    "query": "q",
                    "documents": ["d"],
                    "truncate": "INVALID",
                },
                headers={},
            )

    def test_conflicting_top_n_top_k_raises_canonical_error(self):
        config = NvidiaNimRerankConfig()
        with pytest.raises(UnsupportedParamsError):
            config.map_cohere_rerank_params(
                non_default_params={"top_k": 5},
                model="nvidia_nim/test-model",
                drop_params=False,
                query="q",
                documents=["d"],
                top_n=3,
            )

    def test_real_caller_strict_raises_canonical_400(self):
        """Real litellm.rerank caller raises canonical UnsupportedParamsError with 400."""
        with pytest.raises(UnsupportedParamsError) as exc_info:
            litellm.rerank(
                model="nvidia_nim/nvidia/llama-3_2-nv-rerankqa-1b-v2",
                query="q",
                documents=["d"],
                rank_fields=["text"],
                api_key="fake-key",
            )
        assert exc_info.value.status_code == 400
        assert "rank_fields" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Shared-instance concurrency safety
# ---------------------------------------------------------------------------


class TestConcurrencySafety:
    def test_shared_config_instance_concurrent_requests(self):
        """A single config instance handles concurrent requests without cross-contamination."""
        config = NvidiaNimRerankConfig()

        def make_request(idx: int):
            params = config.map_cohere_rerank_params(
                non_default_params={},
                model="nvidia_nim/test-model",
                drop_params=True,
                query=f"query-{idx}",
                documents=[f"doc-{idx}"],
                top_n=idx + 1,
                return_documents=(idx % 2 == 0),
                rank_fields=["text"] if idx % 3 == 0 else None,
            )
            return idx, params

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(make_request, i) for i in range(64)]
            results = [f.result() for f in futures]

        for idx, params in results:
            ctx = params[_CTX_KEY]
            assert ctx["return_documents"] == (idx % 2 == 0)
            assert params["top_k"] == idx + 1
            assert params["query"] == f"query-{idx}"

    def test_config_instance_is_stateless(self):
        """Config instance has no mutable state after map_cohere_rerank_params."""
        config = NvidiaNimRerankConfig()
        config.map_cohere_rerank_params(
            non_default_params={"rank_fields": ["text"]},
            model="nvidia_nim/test-model",
            drop_params=True,
            query="q",
            documents=["d"],
            return_documents=False,
        )
        assert vars(config) == {}

    def test_interleaved_map_and_transform_no_cross_contamination(self):
        """Interleaving map and transform on shared instance produces correct results."""
        config = NvidiaNimRerankConfig()

        params_a = config.map_cohere_rerank_params(
            non_default_params={},
            model="nvidia_nim/test-model",
            drop_params=False,
            query="qA",
            documents=["docA"],
            return_documents=False,
        )
        params_b = config.map_cohere_rerank_params(
            non_default_params={},
            model="nvidia_nim/test-model",
            drop_params=False,
            query="qB",
            documents=["docB"],
            return_documents=True,
        )

        mock_resp = _make_mock_response(rankings=[{"index": 0, "logit": 0.9}])

        resp_a = config.transform_rerank_response(
            model="nvidia_nim/test-model",
            raw_response=mock_resp,
            model_response=RerankResponse(),
            logging_obj=_make_logging_obj(),
            request_data={"passages": [{"text": "docA"}]},
            optional_params=params_a[_CTX_KEY],
        )
        assert "document" not in resp_a.results[0]

        resp_b = config.transform_rerank_response(
            model="nvidia_nim/test-model",
            raw_response=mock_resp,
            model_response=RerankResponse(),
            logging_obj=_make_logging_obj(),
            request_data={"passages": [{"text": "docB"}]},
            optional_params=params_b[_CTX_KEY],
        )
        assert resp_b.results[0]["document"] == {"text": "docB"}


# ---------------------------------------------------------------------------
# Bounded provider allowlist: unknown/infra kwargs
# ---------------------------------------------------------------------------


class TestBoundedProviderAllowlist:
    """Arbitrary unknown/internal/sensitive kwargs must not be silently copied."""

    def setup_method(self):
        self.config = NvidiaNimRerankConfig()

    def test_strict_rejects_unknown_params(self):
        """Unknown params in non_default_params raise canonical error in strict mode."""
        with pytest.raises(UnsupportedParamsError) as exc_info:
            self.config.map_cohere_rerank_params(
                non_default_params={"some_random_param": "value"},
                model="nvidia_nim/test-model",
                drop_params=False,
                query="q",
                documents=["d"],
            )
        assert exc_info.value.status_code == 400
        assert "some_random_param" in str(exc_info.value)

    def test_strict_rejects_multiple_unknown_params_sorted(self):
        """Multiple unknown params are reported sorted, names-only."""
        with pytest.raises(UnsupportedParamsError) as exc_info:
            self.config.map_cohere_rerank_params(
                non_default_params={"zeta_param": 1, "alpha_param": 2},
                model="nvidia_nim/test-model",
                drop_params=False,
                query="q",
                documents=["d"],
            )
        msg = str(exc_info.value)
        assert "alpha_param" in msg
        assert "zeta_param" in msg
        # Names-only: no values
        assert "1" not in msg.split("alpha_param")[0]

    def test_drop_params_drops_unknown_and_records(self):
        """drop_params=True drops unknown params and records names-only."""
        params = self.config.map_cohere_rerank_params(
            non_default_params={"secret_internal": "sensitive_value"},
            model="nvidia_nim/test-model",
            drop_params=True,
            query="q",
            documents=["d"],
        )
        assert "secret_internal" not in params
        ctx = params[_CTX_KEY]
        records = ctx["adaptation_records"]
        names = {r["name"] for r in records}
        assert "secret_internal" in names
        for r in records:
            if r["name"] == "secret_internal":
                assert r["action"] == "dropped"
                assert r["reason"] == "unsupported_param"

    def test_infra_keys_silently_skipped_no_records(self):
        """Infrastructure keys (api_key, metadata, etc.) are silently skipped."""
        params = self.config.map_cohere_rerank_params(
            non_default_params={
                "api_key": "sk-secret",
                "litellm_call_id": "abc-123",
                "proxy_server_request": {"url": "/rerank"},
                "metadata": {"user_id": "u1"},
                "timeout": 30,
                "max_retries": 2,
                "request_timeout": 15,
                "litellm_trace_id": "trace-123",
            },
            model="nvidia_nim/test-model",
            drop_params=False,
            query="q",
            documents=["d"],
        )
        # None of the infra keys appear in output
        for key in (
            "api_key",
            "litellm_call_id",
            "proxy_server_request",
            "metadata",
            "timeout",
            "max_retries",
            "request_timeout",
            "litellm_trace_id",
        ):
            assert key not in params
        # No adaptation records for infra keys
        ctx = params[_CTX_KEY]
        record_names = {r["name"] for r in ctx["adaptation_records"]}
        assert "api_key" not in record_names
        assert "metadata" not in record_names

    def test_no_secret_leakage_in_optional_params(self):
        """Sensitive values never appear in the returned optional params dict."""
        params = self.config.map_cohere_rerank_params(
            non_default_params={
                "api_key": "sk-super-secret-key",
                "truncate": "END",
            },
            model="nvidia_nim/test-model",
            drop_params=False,
            query="q",
            documents=["d"],
        )
        params_str = json.dumps(params, default=str)
        assert "sk-super-secret-key" not in params_str

    def test_provider_specific_truncate_allowed(self):
        """Known provider-specific param 'truncate' passes through."""
        params = self.config.map_cohere_rerank_params(
            non_default_params={"truncate": "END"},
            model="nvidia_nim/test-model",
            drop_params=False,
            query="q",
            documents=["d"],
        )
        assert params["truncate"] == "END"

    def test_real_caller_allows_max_retries_before_mocked_http(self):
        """Standard retry infrastructure reaches the mocked HTTP call."""
        response, sent_data = _sync_rerank(max_retries=2)
        assert response.results is not None
        assert "max_retries" not in sent_data


# ---------------------------------------------------------------------------
# Permissive invalid truncate: drop + record (not unconditional late raise)
# ---------------------------------------------------------------------------


class TestPermissiveTruncateHandling:
    """Invalid truncate with drop_params=True drops and records; strict raises."""

    def setup_method(self):
        self.config = NvidiaNimRerankConfig()

    def test_strict_invalid_truncate_raises_at_map_time(self):
        """Strict mode raises canonical error for invalid truncate at map time."""
        with pytest.raises(UnsupportedParamsError) as exc_info:
            self.config.map_cohere_rerank_params(
                non_default_params={"truncate": "MIDDLE"},
                model="nvidia_nim/test-model",
                drop_params=False,
                query="q",
                documents=["d"],
            )
        assert "truncate" in str(exc_info.value)
        assert exc_info.value.status_code == 400

    def test_permissive_invalid_truncate_drops_and_records(self):
        """drop_params=True drops invalid truncate and records names-only."""
        params = self.config.map_cohere_rerank_params(
            non_default_params={"truncate": "INVALID"},
            model="nvidia_nim/test-model",
            drop_params=True,
            query="q",
            documents=["d"],
        )
        assert "truncate" not in params
        ctx = params[_CTX_KEY]
        records = ctx["adaptation_records"]
        truncate_records = [r for r in records if r["name"] == "truncate"]
        assert len(truncate_records) == 1
        assert truncate_records[0]["action"] == "dropped"
        assert truncate_records[0]["reason"] == "unsupported_param"

    def test_valid_truncate_not_dropped_in_permissive_mode(self):
        """Valid truncate passes through even with drop_params=True."""
        params = self.config.map_cohere_rerank_params(
            non_default_params={"truncate": "NONE"},
            model="nvidia_nim/test-model",
            drop_params=True,
            query="q",
            documents=["d"],
        )
        assert params["truncate"] == "NONE"

    def test_transform_still_validates_for_direct_calls(self):
        """transform_rerank_request still raises for invalid truncate (safety net)."""
        with pytest.raises(UnsupportedParamsError):
            self.config.transform_rerank_request(
                model="nvidia_nim/test-model",
                optional_rerank_params={
                    "query": "q",
                    "documents": ["d"],
                    "truncate": "BADVALUE",
                },
                headers={},
            )


# ---------------------------------------------------------------------------
# return_documents: local response control exempt from provider-omission records
# ---------------------------------------------------------------------------


class TestReturnDocumentsExemption:
    """return_documents is a local response control, not a provider param.

    It must remain absent from provider payload and must NOT produce
    provider-omission/drop adaptation records.
    """

    def setup_method(self):
        self.config = NvidiaNimRerankConfig()

    def test_return_documents_no_adaptation_record(self):
        """return_documents does not produce any adaptation record."""
        params = self.config.map_cohere_rerank_params(
            non_default_params={},
            model="nvidia_nim/test-model",
            drop_params=False,
            query="q",
            documents=["d"],
            return_documents=False,
        )
        ctx = params[_CTX_KEY]
        record_names = {r["name"] for r in ctx["adaptation_records"]}
        assert "return_documents" not in record_names

    def test_return_documents_not_in_provider_body(self):
        """return_documents never appears in the transformed request body."""
        params = self.config.map_cohere_rerank_params(
            non_default_params={},
            model="nvidia_nim/test-model",
            drop_params=False,
            query="q",
            documents=["d"],
            return_documents=False,
        )
        body = self.config.transform_rerank_request(
            model="nvidia_nim/test-model",
            optional_rerank_params=params,
            headers={},
        )
        assert "return_documents" not in body

    def test_return_documents_in_ndp_not_treated_as_unknown(self):
        """return_documents in non_default_params is handled, not unknown."""
        params = self.config.map_cohere_rerank_params(
            non_default_params={"return_documents": True},
            model="nvidia_nim/test-model",
            drop_params=False,
            query="q",
            documents=["d"],
            return_documents=True,
        )
        # Should not raise, should not be in output params
        assert "return_documents" not in params
        ctx = params[_CTX_KEY]
        record_names = {r["name"] for r in ctx["adaptation_records"]}
        assert "return_documents" not in record_names

    def test_return_documents_controls_response_shaping(self):
        """return_documents=False in context strips documents from response."""
        params = self.config.map_cohere_rerank_params(
            non_default_params={},
            model="nvidia_nim/test-model",
            drop_params=False,
            query="q",
            documents=["d"],
            return_documents=False,
        )
        mock_resp = _make_mock_response(rankings=[{"index": 0, "logit": 0.9}])
        resp = self.config.transform_rerank_response(
            model="nvidia_nim/test-model",
            raw_response=mock_resp,
            model_response=RerankResponse(),
            logging_obj=_make_logging_obj(),
            request_data={"passages": [{"text": "d"}]},
            optional_params=params[_CTX_KEY],
        )
        assert "document" not in resp.results[0]


class TestLoggingContainment:
    def _assert_context_absent(self, logging_obj):
        assert _CTX_KEY not in logging_obj.optional_params
        assert _CTX_KEY not in logging_obj.model_call_details["optional_params"]
        assert _CTX_KEY not in logging_obj.model_call_details
        assert "return_documents" not in logging_obj.optional_params
        assert "adaptation_records" not in logging_obj.model_call_details

    def test_sync_logging_excludes_response_context(self):
        logging_obj = _make_real_logging_obj()
        response, sent_data = _sync_rerank(
            litellm_logging_obj=logging_obj,
            return_documents=False,
            rank_fields=["private-field"],
            drop_params=True,
        )
        self._assert_context_absent(logging_obj)
        assert _CTX_KEY not in sent_data
        assert "private-field" not in json.dumps(
            logging_obj.model_call_details, default=lambda _: "<non-serializable>"
        )
        assert "document" not in response.results[0]
        records = response._hidden_params[PROVIDER_PARAMETER_ADAPTATIONS_METADATA_KEY]
        assert {record["name"] for record in records} == {"rank_fields"}

    @pytest.mark.asyncio()
    async def test_async_logging_excludes_response_context(self):
        logging_obj = _make_real_logging_obj()
        response, sent_data = await _async_rerank(
            litellm_logging_obj=logging_obj,
            return_documents=False,
            rank_fields=["private-field"],
            drop_params=True,
        )
        self._assert_context_absent(logging_obj)
        assert _CTX_KEY not in sent_data
        assert "private-field" not in json.dumps(
            logging_obj.model_call_details, default=lambda _: "<non-serializable>"
        )
        assert "document" not in response.results[0]
        records = response._hidden_params[PROVIDER_PARAMETER_ADAPTATIONS_METADATA_KEY]
        assert {record["name"] for record in records} == {"rank_fields"}


class TestCustomCallbackSanitization:
    API_KEY_SENTINEL = "nv-api-key-sentinel-4f8d21"
    DROPPED_VALUE_SENTINEL = "nv-dropped-value-sentinel-91ac73"
    FORBIDDEN_CALLBACK_KEYS = {
        _CTX_KEY,
        "adaptation_records",
        "adaptation_truncated_count",
        "api_key",
        "authorization",
        "headers",
        "litellm_logging_obj",
        "secret_internal",
        "secret_fields",
    }

    def _assert_callback_clean(self, callback_kwargs):
        assert callback_kwargs is not None
        _assert_recursive_absence(
            callback_kwargs,
            {self.API_KEY_SENTINEL, self.DROPPED_VALUE_SENTINEL},
            self.FORBIDDEN_CALLBACK_KEYS,
        )

    def _assert_names_only_response_metadata(self, response):
        records = response._hidden_params[PROVIDER_PARAMETER_ADAPTATIONS_METADATA_KEY]
        secret_internal_records = [
            record for record in records if record["name"] == "secret_internal"
        ]
        assert secret_internal_records == [
            {
                "name": "secret_internal",
                "action": "dropped",
                "reason": "unsupported_param",
            }
        ]
        assert self.DROPPED_VALUE_SENTINEL not in json.dumps(records)
        assert self.API_KEY_SENTINEL not in json.dumps(records)

    def test_sync_custom_callback_excludes_credentials_and_dropped_values(self):
        callback = _CaptureCallback()
        logging_obj = _make_real_logging_obj(sync_callback=callback)

        with patch("litellm.utils.executor.submit"):
            response, sent_data = _sync_rerank(
                litellm_logging_obj=logging_obj,
                api_key=self.API_KEY_SENTINEL,
                secret_internal=self.DROPPED_VALUE_SENTINEL,
                drop_params=True,
            )

        logging_obj.success_handler(
            result=response,
            start_time=logging_obj.start_time,
            end_time=datetime.datetime.now(),
        )

        self._assert_callback_clean(callback.sync_kwargs)
        self._assert_names_only_response_metadata(response)
        assert self.DROPPED_VALUE_SENTINEL not in json.dumps(sent_data)
        assert self.API_KEY_SENTINEL not in json.dumps(sent_data)

    @pytest.mark.asyncio()
    async def test_async_custom_callback_excludes_credentials_and_dropped_values(
        self,
    ):
        callback = _CaptureCallback()
        logging_obj = _make_real_logging_obj(async_callback=callback)
        old_drop_params = litellm.drop_params
        litellm.drop_params = True
        try:
            with patch("litellm.utils.executor.submit"):
                response, sent_data = await _async_rerank(
                    litellm_logging_obj=logging_obj,
                    api_key=self.API_KEY_SENTINEL,
                    secret_internal=self.DROPPED_VALUE_SENTINEL,
                )

            await logging_obj.async_success_handler(
                result=response,
                start_time=logging_obj.start_time,
                end_time=datetime.datetime.now(),
            )
        finally:
            litellm.drop_params = old_drop_params

        self._assert_callback_clean(callback.async_kwargs)
        self._assert_names_only_response_metadata(response)
        assert self.DROPPED_VALUE_SENTINEL not in json.dumps(sent_data)
        assert self.API_KEY_SENTINEL not in json.dumps(sent_data)
