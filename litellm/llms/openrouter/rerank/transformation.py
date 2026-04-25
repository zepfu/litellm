from typing import Any, Dict, List, Optional, Union

import httpx

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.rerank.transformation import BaseRerankConfig
from litellm.secret_managers.main import get_secret, get_secret_str
from litellm.types.rerank import OptionalRerankParams, RerankRequest, RerankResponse

from ..common_utils import OpenRouterException


class OpenRouterRerankConfig(BaseRerankConfig):
    """
    OpenRouter rerank support.

    OpenRouter exposes rerank at /api/v1/rerank while preserving the Cohere-style
    request/response shape used by LiteLLM's rerank API.
    """

    @staticmethod
    def _normalize_api_base(api_base: Optional[str]) -> str:
        if api_base:
            api_base = api_base.rstrip("/")
        else:
            api_base = "https://openrouter.ai/api/v1"

        if api_base.endswith("/rerank"):
            return api_base
        if api_base.endswith("/api"):
            api_base = f"{api_base}/v1"
        return f"{api_base}/rerank"

    def get_complete_url(
        self,
        api_base: Optional[str],
        model: str,
        optional_params: Optional[dict] = None,
    ) -> str:
        return self._normalize_api_base(api_base)

    def get_supported_cohere_rerank_params(self, model: str) -> list:
        return [
            "query",
            "documents",
            "top_n",
            "rank_fields",
            "return_documents",
            "max_chunks_per_doc",
            "max_tokens_per_doc",
        ]

    def map_cohere_rerank_params(
        self,
        non_default_params: Optional[dict],
        model: str,
        drop_params: bool,
        query: str,
        documents: List[Union[str, Dict[str, Any]]],
        custom_llm_provider: Optional[str] = None,
        top_n: Optional[int] = None,
        rank_fields: Optional[List[str]] = None,
        return_documents: Optional[bool] = True,
        max_chunks_per_doc: Optional[int] = None,
        max_tokens_per_doc: Optional[int] = None,
    ) -> Dict:
        return dict(
            OptionalRerankParams(
                query=query,
                documents=documents,
                top_n=top_n,
                rank_fields=rank_fields,
                return_documents=return_documents,
                max_chunks_per_doc=max_chunks_per_doc,
                max_tokens_per_doc=max_tokens_per_doc,
            )
        )

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
        optional_params: Optional[dict] = None,
    ) -> dict:
        api_key = (
            api_key
            or litellm.api_key
            or litellm.openrouter_key
            or get_secret_str("OPENROUTER_API_KEY")
            or get_secret_str("OR_API_KEY")
            or get_secret_str("AAWM_OPENROUTER_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "OpenRouter API key is required. Set OPENROUTER_API_KEY, "
                "OR_API_KEY, AAWM_OPENROUTER_API_KEY, or pass api_key."
            )

        default_headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "accept": "application/json",
            "HTTP-Referer": get_secret("OR_SITE_URL") or "https://litellm.ai",
            "X-Title": get_secret("OR_APP_NAME") or "liteLLM",
        }
        return {**default_headers, **headers}

    def transform_rerank_request(
        self,
        model: str,
        optional_rerank_params: Dict,
        headers: dict,
    ) -> dict:
        if model.startswith("openrouter/"):
            model = model.replace("openrouter/", "", 1)

        if "query" not in optional_rerank_params:
            raise ValueError("query is required for OpenRouter rerank")
        if "documents" not in optional_rerank_params:
            raise ValueError("documents is required for OpenRouter rerank")

        rerank_request = RerankRequest(
            model=model,
            query=optional_rerank_params["query"],
            documents=optional_rerank_params["documents"],
            top_n=optional_rerank_params.get("top_n", None),
            rank_fields=optional_rerank_params.get("rank_fields", None),
            return_documents=optional_rerank_params.get("return_documents", None),
            max_chunks_per_doc=optional_rerank_params.get("max_chunks_per_doc", None),
            max_tokens_per_doc=optional_rerank_params.get("max_tokens_per_doc", None),
        )
        return rerank_request.model_dump(exclude_none=True)

    @staticmethod
    def _coerce_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)):
            return str(value)
        if isinstance(value, list):
            return "\n".join(
                text
                for item in value
                if (text := OpenRouterRerankConfig._coerce_text(item).strip())
            )
        if isinstance(value, dict):
            if "text" in value:
                return OpenRouterRerankConfig._coerce_text(value.get("text"))
            return "\n".join(
                text
                for nested_value in value.values()
                if (
                    text := OpenRouterRerankConfig._coerce_text(nested_value).strip()
                )
            )
        return str(value)

    @staticmethod
    def _estimate_request_tokens(model: str, request_data: dict) -> Optional[int]:
        query = OpenRouterRerankConfig._coerce_text(request_data.get("query")).strip()
        documents = request_data.get("documents")
        if not isinstance(documents, list):
            return None

        document_texts = [
            text
            for document in documents
            if (text := OpenRouterRerankConfig._coerce_text(document).strip())
        ]
        text = "\n\n".join([query, *document_texts]).strip()
        if not text:
            return None

        try:
            token_count = litellm.token_counter(model=model or "", text=text)
            if isinstance(token_count, int) and token_count > 0:
                return token_count
        except Exception:
            pass

        return max(1, (len(text) + 3) // 4)

    @staticmethod
    def _populate_estimated_total_tokens(
        model: str,
        raw_response_json: Dict[str, Any],
        request_data: dict,
        optional_params: dict,
    ) -> None:
        meta = raw_response_json.setdefault("meta", {})
        if not isinstance(meta, dict):
            return
        billed_units = meta.setdefault("billed_units", {})
        if not isinstance(billed_units, dict):
            return
        if billed_units.get("total_tokens"):
            return

        token_source = request_data if request_data else optional_params
        estimated_tokens = OpenRouterRerankConfig._estimate_request_tokens(
            model=model,
            request_data=token_source,
        )
        if estimated_tokens is not None:
            billed_units["total_tokens"] = estimated_tokens

    def transform_rerank_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: RerankResponse,
        logging_obj: Any,
        api_key: Optional[str] = None,
        request_data: dict = {},
        optional_params: dict = {},
        litellm_params: dict = {},
    ) -> RerankResponse:
        logging_obj.post_call(original_response=raw_response.text)
        try:
            raw_response_json = raw_response.json()
        except Exception:
            raise OpenRouterException(
                message=raw_response.text,
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        usage = raw_response_json.get("usage") or {}
        if "meta" not in raw_response_json and usage:
            search_units = usage.get("search_units")
            total_tokens = usage.get("total_tokens")
            raw_response_json["meta"] = {
                "billed_units": {
                    "search_units": search_units,
                    "total_tokens": total_tokens,
                }
            }
        self._populate_estimated_total_tokens(
            model=model,
            raw_response_json=raw_response_json,
            request_data=request_data,
            optional_params=optional_params,
        )

        response = RerankResponse(**raw_response_json)
        if usage:
            response._hidden_params["openrouter_usage"] = usage
            response._hidden_params["openrouter_provider"] = raw_response_json.get(
                "provider"
            )
            response._hidden_params["openrouter_response_model"] = (
                raw_response_json.get("model")
            )
            response_cost = usage.get("cost")
            if response_cost is not None:
                response._hidden_params.setdefault("additional_headers", {})[
                    "llm_provider-x-litellm-response-cost"
                ] = float(response_cost)
        return response

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return OpenRouterException(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )
