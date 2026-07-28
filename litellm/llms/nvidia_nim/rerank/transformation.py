from typing import Any, Dict, List, Literal, Optional, Union

import httpx
from typing_extensions import Required, TypedDict

import litellm
from litellm._uuid import uuid
from litellm.exceptions import UnsupportedParamsError
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.param_adaptation import (
    PROVIDER_PARAMETER_ADAPTATIONS_METADATA_KEY,
    PROVIDER_PARAMETER_ADAPTATIONS_TRUNCATED_COUNT_METADATA_KEY,
    AdaptationCollector,
)
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.rerank.transformation import BaseRerankConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.rerank import (
    RerankBilledUnits,
    RerankResponse,
    RerankResponseMeta,
    RerankResponseResult,
)
from litellm.types.utils import all_litellm_params, get_nvidia_nim_model_metadata

# Key used to embed request-local context inside the optional_params dict.
# This dict flows from map_cohere_rerank_params through the HTTP handler to
# transform_rerank_request (stripped) and transform_rerank_response (read).
_CTX_KEY = "_nvidia_nim_rerank_ctx"


def _raise_unsupported(
    param_names: List[str],
    context: str = "",
    model: Optional[str] = None,
) -> None:
    """Raise canonical UnsupportedParamsError with names-only message (status 400)."""
    names_str = ", ".join(param_names)
    msg = f"Unsupported parameter(s): {names_str}"
    if context:
        msg += f". {context}"
    raise UnsupportedParamsError(
        message=msg,
        model=model,
        llm_provider="nvidia_nim",
    )


class NvidiaNimQueryObject(TypedDict):
    text: Required[str]


class NvidiaNimPassageObject(TypedDict):
    text: Required[str]


class NvidiaNimRerankRequest(TypedDict, total=False):
    model: Required[str]
    query: Required[NvidiaNimQueryObject]
    passages: Required[List[NvidiaNimPassageObject]]
    truncate: Literal["NONE", "END"]
    top_k: int


class NvidiaNimRankingResult(TypedDict):
    index: Required[int]
    logit: Required[float]


class NvidiaNimRerankResponse(TypedDict):
    rankings: Required[List[NvidiaNimRankingResult]]


class NvidiaNimRerankConfig(BaseRerankConfig):
    """
    Reference: https://docs.api.nvidia.com/nim/reference/nvidia-llama-3_2-nv-rerankqa-1b-v2-infer

    Nvidia NIM rerank API uses a different format:
    - query is an object with 'text' field
    - documents are called 'passages' and have 'text' field

    This config is stateless and concurrency-safe.  All request-local state
    (return_documents, adaptation records) is carried inside the returned
    optional_params dict under a hidden context key, never on the instance.

    Model-specific endpoint and body-model decisions are resolved through
    ``get_nvidia_nim_model_metadata()`` (catalog-driven).  Unknown models
    fall back to derived default behavior (prefix strip, underscore-to-dot,
    standard /v1/retrieval/{model}/reranking URL).
    """

    DEFAULT_NIM_RERANK_API_BASE = "https://ai.api.nvidia.com"

    # Provider-wide parameter policy (model-independent)
    VALID_TRUNCATE_VALUES = ("NONE", "END")
    UNSUPPORTED_COHERE_PARAMS = (
        "rank_fields",
        "max_chunks_per_doc",
        "max_tokens_per_doc",
    )

    # NVIDIA NIM-specific provider params allowed through to the request body.
    PROVIDER_SPECIFIC_PARAMS = frozenset({"truncate"})

    # Infrastructure / framework keys that flow through non_default_params
    # from the rerank() caller and @client decorator. These are silently
    # skipped because they are not provider request parameters.
    _INFRA_KEYS = frozenset(all_litellm_params).union(
        GenericLiteLLMParams.model_fields.keys(),
        {
            "arerank",
            "drop_params",
            "litellm_params",
            "user",
        },
    )

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Policy metadata resolver (model-independent)
    # ------------------------------------------------------------------

    @classmethod
    def get_rerank_policy_metadata(cls) -> Dict[str, Any]:
        """Normalized provider-wide rerank parameter policy.

        Separates provider-wide controls from model-specific endpoint/alias
        data.  Does not read or require catalog entries.
        """
        return {
            "provider": "nvidia_nim",
            "supported_cohere_params": [
                "query",
                "documents",
                "top_n",
                "return_documents",
            ],
            "unsupported_cohere_params": list(cls.UNSUPPORTED_COHERE_PARAMS),
            "valid_truncate_values": list(cls.VALID_TRUNCATE_VALUES),
            "param_renames": {"top_n": "top_k"},
            "local_only_params": ["return_documents"],
        }

    # ------------------------------------------------------------------
    # Model name / endpoint resolution (catalog-driven)
    # ------------------------------------------------------------------

    def _get_clean_model_name(self, model: str) -> str:
        """Strip 'nvidia_nim/' prefix from model name if present."""
        if model.startswith("nvidia_nim/"):
            return model[len("nvidia_nim/") :]
        return model

    def _get_rerank_metadata(self, model: str):
        """Return catalog rerank metadata (endpoint_path, body_model) or None."""
        clean_model = self._get_clean_model_name(model)
        metadata = get_nvidia_nim_model_metadata(clean_model)
        return metadata.get("rerank")

    def pop_rerank_response_context(
        self, optional_rerank_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Remove and return response-only context before request logging."""
        context = optional_rerank_params.pop(_CTX_KEY, {})
        return context if isinstance(context, dict) else {}

    def _get_model_name_for_body(self, model: str) -> str:
        """Resolve the model name to send in the provider request body.

        Uses catalog ``body_model`` when available; otherwise applies the
        default derived transform (underscore to dot).
        """
        clean_model = self._get_clean_model_name(model)
        rerank_meta = self._get_rerank_metadata(model)
        if rerank_meta is not None:
            return rerank_meta["body_model"]
        return clean_model.replace("_", ".")

    def get_complete_url(
        self,
        api_base: Optional[str],
        model: str,
        optional_params: Optional[dict] = None,
    ) -> str:
        """
        Construct the Nvidia NIM rerank URL.

        Uses catalog ``endpoint_path`` when available; otherwise falls back
        to the default /v1/retrieval/{model}/reranking pattern.
        """
        if not api_base:
            api_base = self.DEFAULT_NIM_RERANK_API_BASE

        api_base = api_base.rstrip("/")

        # Check if user already provided the full URL with /retrieval/ path
        if "/retrieval/" in api_base:
            return api_base

        # Ensure we don't have duplicate /v1
        if api_base.endswith("/v1"):
            api_base = api_base[:-3]

        rerank_meta = self._get_rerank_metadata(model)
        if rerank_meta is not None:
            # Strip leading slash to avoid double-slash when joining
            path = rerank_meta["endpoint_path"].lstrip("/")
            return f"{api_base}/{path}"

        # Default derived behavior
        clean_model = self._get_clean_model_name(model)
        return f"{api_base}/v1/retrieval/{clean_model}/reranking"

    def get_supported_cohere_rerank_params(self, model: str) -> list:
        """
        Nvidia NIM supports these rerank parameters.
        """
        return [
            "query",
            "documents",
            "top_n",
            "return_documents",
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
        """
        Map Cohere/OpenAI rerank params to Nvidia NIM format.

        Parameter mapping:
        - top_n (Cohere) -> top_k (Nvidia), recorded as renamed/provider_rename
        - return_documents: captured in request-local context, never sent to provider
        - rank_fields, max_chunks_per_doc, max_tokens_per_doc: unsupported,
          rejected (strict) or dropped (permissive) with names-only records

        All request-local state is embedded in the returned dict under a
        hidden context key; nothing is stored on the config instance.
        """
        ndp = non_default_params or {}
        collector = AdaptationCollector()

        # ---- Unsupported params: check both named args and non_default_params
        unsupported_provided: List[str] = []
        if rank_fields is not None:
            unsupported_provided.append("rank_fields")
        for name in self.UNSUPPORTED_COHERE_PARAMS:
            if name == "rank_fields":
                continue  # already checked via named param
            if ndp.get(name) is not None:
                unsupported_provided.append(name)

        if unsupported_provided:
            if not drop_params:
                _raise_unsupported(
                    unsupported_provided,
                    context="Set drop_params=True to silently drop them",
                    model=model,
                )
            collector.add_many(
                unsupported_provided,
                action="dropped",
                reason="unsupported_param",
            )

        optional_nvidia_nim_rerank_params: Dict[str, Any] = {
            "query": query,
            "documents": documents,
        }

        # ---- top_n -> top_k with conflict detection
        direct_top_k = ndp.get("top_k")
        if top_n is not None and direct_top_k is not None:
            if top_n == direct_top_k:
                # Equal: dedup, prefer provider-native top_k, record rename
                collector.add("top_n", action="renamed", reason="provider_rename")
                optional_nvidia_nim_rerank_params["top_k"] = direct_top_k
            else:
                if not drop_params:
                    _raise_unsupported(
                        ["top_n", "top_k"],
                        context="Conflicting values for top_n and top_k",
                        model=model,
                    )
                # Permissive: prefer provider-native top_k, record drop of top_n
                collector.add("top_n", action="dropped", reason="unsupported_param")
                optional_nvidia_nim_rerank_params["top_k"] = direct_top_k
        elif top_n is not None:
            # Standard rename: top_n -> top_k
            collector.add("top_n", action="renamed", reason="provider_rename")
            optional_nvidia_nim_rerank_params["top_k"] = top_n
        elif direct_top_k is not None:
            optional_nvidia_nim_rerank_params["top_k"] = direct_top_k

        # ---- Pass through only known NVIDIA NIM provider params
        _handled_keys = frozenset(
            (
                "query",
                "documents",
                "top_n",
                "top_k",
                "return_documents",
                *self.UNSUPPORTED_COHERE_PARAMS,
            )
        )
        unknown_params: List[str] = []
        if ndp:
            for key, value in ndp.items():
                if key in _handled_keys or key in self._INFRA_KEYS:
                    continue
                if key in self.PROVIDER_SPECIFIC_PARAMS:
                    # Validate truncate at map time (drop_params-aware)
                    if key == "truncate" and value not in self.VALID_TRUNCATE_VALUES:
                        if not drop_params:
                            _raise_unsupported(
                                ["truncate"],
                                context=(
                                    f"Invalid value. "
                                    f"Must be one of: "
                                    f"{', '.join(self.VALID_TRUNCATE_VALUES)}"
                                ),
                                model=model,
                            )
                        collector.add(
                            "truncate",
                            action="dropped",
                            reason="unsupported_param",
                        )
                        continue
                    optional_nvidia_nim_rerank_params[key] = value
                    continue
                unknown_params.append(key)

        if unknown_params:
            if not drop_params:
                _raise_unsupported(
                    sorted(unknown_params),
                    context="Set drop_params=True to silently drop them",
                    model=model,
                )
            collector.add_many(
                sorted(unknown_params),
                action="dropped",
                reason="unsupported_param",
            )

        # ---- Embed request-local context (never sent to provider)
        optional_nvidia_nim_rerank_params[_CTX_KEY] = {
            "return_documents": return_documents,
            "adaptation_records": [
                {"name": r.name, "action": r.action, "reason": r.reason}
                for r in collector.records
            ],
            "adaptation_truncated_count": collector.truncated_count,
        }

        return dict(optional_nvidia_nim_rerank_params)

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
        optional_params: Optional[dict] = None,
    ) -> dict:
        """
        Validate that the Nvidia NIM API key is present.
        """
        if api_key is None:
            api_key = get_secret_str("NVIDIA_NIM_API_KEY") or litellm.api_key

        if api_key is None:
            raise ValueError(
                "Nvidia NIM API key is required. Please set 'NVIDIA_NIM_API_KEY' in your environment"
            )

        default_headers = {
            "Authorization": f"Bearer {api_key}",
            "accept": "application/json",
            "content-type": "application/json",
        }

        # If 'Authorization' is provided in headers, it overrides the default
        if "Authorization" in headers:
            default_headers["Authorization"] = headers["Authorization"]

        # Merge other headers, overriding any default ones except Authorization
        return {**default_headers, **headers}

    def transform_rerank_request(
        self,
        model: str,
        optional_rerank_params: Dict,
        headers: dict,
    ) -> dict:
        """
        Transform request to Nvidia NIM format.

        Nvidia NIM expects:
        - query as {text: "..."}
        - documents as passages: [{text: "..."}, ...]
        - Optional: truncate (NONE or END), top_k

        The hidden context key (_CTX_KEY) is stripped and never sent to the
        provider.
        """
        if "query" not in optional_rerank_params:
            _raise_unsupported(["query"], context="query is required", model=model)
        if "documents" not in optional_rerank_params:
            _raise_unsupported(
                ["documents"], context="documents is required", model=model
            )

        query = optional_rerank_params["query"]
        documents = optional_rerank_params["documents"]

        # Transform query to object format
        query_obj: NvidiaNimQueryObject = {"text": query}

        # Transform documents to passages format
        passages: List[NvidiaNimPassageObject] = []
        for doc in documents:
            if isinstance(doc, str):
                passages.append({"text": doc})
            elif isinstance(doc, dict):
                if "text" in doc:
                    passages.append({"text": doc["text"]})
                else:
                    import json

                    passages.append({"text": json.dumps(doc)})
            else:
                passages.append({"text": str(doc)})

        # Build request using TypedDict
        request_data: NvidiaNimRerankRequest = {
            "model": self._get_model_name_for_body(model),
            "query": query_obj,
            "passages": passages,
        }

        # Add optional top_k parameter if provided (already mapped from top_n)
        if "top_k" in optional_rerank_params and optional_rerank_params.get("top_k") is not None:  # type: ignore
            request_data["top_k"] = optional_rerank_params.get("top_k")  # type: ignore

        # Add Nvidia-specific truncate parameter if provided
        if "truncate" in optional_rerank_params and optional_rerank_params.get("truncate") is not None:  # type: ignore
            truncate_value = optional_rerank_params.get("truncate")  # type: ignore
            if truncate_value not in self.VALID_TRUNCATE_VALUES:
                _raise_unsupported(
                    ["truncate"],
                    context=(
                        f"Invalid value. "
                        f"Must be one of: {', '.join(self.VALID_TRUNCATE_VALUES)}"
                    ),
                    model=model,
                )
            request_data["truncate"] = truncate_value  # type: ignore

        return dict(request_data)

    def transform_rerank_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: RerankResponse,
        logging_obj: LiteLLMLoggingObj,
        api_key: Optional[str] = None,
        request_data: dict = {},
        optional_params: dict = {},
        litellm_params: dict = {},
    ) -> RerankResponse:
        """
        Transform Nvidia NIM rerank response to LiteLLM format.

        Request-local context (return_documents, adaptation records) is passed
        separately from provider and logging optional params. No instance state
        is used.
        """
        try:
            raw_response_json = raw_response.json()
        except Exception:
            raise BaseLLMException(
                status_code=raw_response.status_code,
                message=raw_response.text,
                headers=raw_response.headers,
            )

        nvidia_response: NvidiaNimRerankResponse = raw_response_json

        results: List[RerankResponseResult] = []
        rankings = nvidia_response.get("rankings", [])

        original_passages: List[NvidiaNimPassageObject] = request_data.get(
            "passages", []
        )

        # Read request-local context
        ctx: Dict[str, Any] = optional_params
        include_documents = ctx.get("return_documents") is not False

        for ranking in rankings:
            result_item: RerankResponseResult = {
                "index": ranking["index"],
                "relevance_score": ranking["logit"],
            }

            if include_documents:
                index: int = ranking["index"]
                if index < len(original_passages):
                    result_item["document"] = {"text": original_passages[index]["text"]}  # type: ignore

            results.append(result_item)

        usage = raw_response_json.get("usage", {})
        total_tokens = usage.get("total_tokens", 0)

        billed_units: RerankBilledUnits = {
            "total_tokens": total_tokens if total_tokens > 0 else len(results)
        }

        meta: RerankResponseMeta = {"billed_units": billed_units}

        response = RerankResponse(
            id=raw_response_json.get("id") or str(uuid.uuid4()),
            results=results,
            meta=meta,
        )

        # Attach adaptation metadata as flat keys (not nested)
        adaptation_records = ctx.get("adaptation_records", [])
        if adaptation_records:
            response._hidden_params[
                PROVIDER_PARAMETER_ADAPTATIONS_METADATA_KEY
            ] = adaptation_records
            response._hidden_params[
                PROVIDER_PARAMETER_ADAPTATIONS_TRUNCATED_COUNT_METADATA_KEY
            ] = ctx.get("adaptation_truncated_count", 0)

        return response

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )
