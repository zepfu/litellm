"""
C2: HuggingFace rerank must not put heuristic estimates into billed_units.total_tokens.
"""

import json
from unittest.mock import MagicMock

import httpx

from litellm.llms.huggingface.rerank.transformation import HuggingFaceRerankConfig
from litellm.types.rerank import RerankResponse


def test_huggingface_rerank_does_not_set_estimated_billed_total_tokens():
    config = HuggingFaceRerankConfig()
    response_data = [{"index": 0, "score": 0.9}, {"index": 1, "score": 0.1}]
    raw_response = MagicMock(spec=httpx.Response)
    raw_response.json.return_value = response_data
    raw_response.status_code = 200
    raw_response.text = json.dumps(response_data)

    result = config.transform_rerank_response(
        model="huggingface/BAAI/bge-reranker-base",
        raw_response=raw_response,
        model_response=RerankResponse(),
        logging_obj=MagicMock(),
        request_data={
            "query": "hello",
            "texts": ["hello", "world"],
            "return_text": False,
        },
    )

    assert result.meta is not None
    billed = result.meta.get("billed_units") or {}
    assert billed.get("search_units") == 1
    # Heuristic must not be presented as provider billed total_tokens
    assert billed.get("total_tokens") in (None, 0)
    tokens = result.meta.get("tokens") or {}
    assert tokens.get("input_tokens") is not None
    assert tokens.get("input_tokens") > 0
