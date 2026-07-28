"""
Nvidia NIM endpoint: https://docs.api.nvidia.com/nim/reference/databricks-dbrx-instruct-infer 

This is OpenAI compatible 

This file only contains param mapping logic

API calling is done using the OpenAI SDK with an api_base
"""
from typing import Callable, Optional

from litellm.exceptions import UnsupportedParamsError
from litellm.litellm_core_utils.param_adaptation import AdaptationCollector
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.types.utils import get_nvidia_nim_model_metadata
from litellm.utils import supports_reasoning

# Catalog follow-up (D1-556): populate ``supports_max_completion_tokens``
# in nvidia_nim model entries in model_prices_and_context_window.json for
# models whose NIM endpoint natively accepts ``max_completion_tokens``.
# Until that metadata exists every model defaults to the conservative
# ``max_tokens``-only policy, preserving the historical mapping.
_METADATA_KEY_NATIVE_MCT = "supports_max_completion_tokens"

_TOKEN_LIMIT_FIELDS = frozenset({"max_tokens", "max_completion_tokens"})

# Injectable resolver signature: (model, provider) -> metadata dict | None.
MetadataResolver = Callable[[str, str], Optional[dict]]


def _default_metadata_resolver(model: str, provider: str) -> Optional[dict]:
    """Resolve NVIDIA NIM metadata via the typed normalized helper.

    Delegates to ``get_nvidia_nim_model_metadata`` which internally uses
    ``get_model_info`` with normalized name resolution, so provider-prefixed
    (``nvidia_nim/vendor/model``) and unprefixed (``vendor/model``) inputs
    resolve identically.

    Returns the ``NvidiaNimModelMetadata`` dict (a plain dict at runtime)
    or ``None`` on unexpected failure.
    """
    try:
        return dict(get_nvidia_nim_model_metadata(model))
    except Exception:
        return None


class NvidiaNimConfig(OpenAIGPTConfig):
    """
    Reference: https://docs.api.nvidia.com/nim/reference/databricks-dbrx-instruct-infer

    The class `NvidiaNimConfig` provides configuration for the Nvidia NIM's Chat Completions API interface. Below are the parameters:
    """

    # The field NVIDIA NIM endpoints accept natively.
    _PROVIDER_NATIVE_TOKEN_LIMIT_FIELD: str = "max_tokens"
    _ALIAS_TOKEN_LIMIT_FIELD: str = "max_completion_tokens"

    def __init__(
        self,
        metadata_resolver: Optional[MetadataResolver] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        # Instance-local resolver; falls back to the module-level default.
        # Never stored on the class -- concurrent instances are isolated.
        self._metadata_resolver: MetadataResolver = (
            metadata_resolver
            if metadata_resolver is not None
            else _default_metadata_resolver
        )

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def _provider_supports_max_completion_tokens(self, model: str) -> bool:
        """Return True when model metadata declares native *max_completion_tokens*.

        Delegates to the instance-local ``_metadata_resolver``.  Returns
        ``False`` for unknown models or when the key is absent
        (conservative default).
        """
        try:
            entry = self._metadata_resolver(model, "nvidia_nim")
            if entry is None:
                return False
            return entry.get(_METADATA_KEY_NATIVE_MCT, False) is True
        except Exception:
            return False

    @staticmethod
    def _add_reasoning_effort_if_supported(model: str, params: list) -> list:
        if supports_reasoning(model=model, custom_llm_provider="nvidia_nim"):
            return [*params, "reasoning_effort"]
        return params

    def get_supported_openai_params(self, model: str) -> list:
        """
        Get the supported OpenAI params for the given model


        Updated on July 5th, 2024 - based on https://docs.api.nvidia.com/nim/reference
        """
        if model in [
            "google/recurrentgemma-2b",
            "google/gemma-2-27b-it",
            "google/gemma-2-9b-it",
            "gemma-2-9b-it",
        ]:
            return self._add_reasoning_effort_if_supported(
                model,
                ["stream", "temperature", "top_p", "max_tokens", "stop", "seed"],
            )
        elif model == "nvidia/nemotron-4-340b-instruct":
            return self._add_reasoning_effort_if_supported(
                model,
                [
                    "stream",
                    "temperature",
                    "top_p",
                    "max_tokens",
                    "max_completion_tokens",
                ],
            )
        elif model == "nvidia/nemotron-4-340b-reward":
            return self._add_reasoning_effort_if_supported(
                model,
                [
                    "stream",
                ],
            )
        elif model in ["google/codegemma-1.1-7b"]:
            # most params - but no 'seed' :(
            return self._add_reasoning_effort_if_supported(
                model,
                [
                    "stream",
                    "temperature",
                    "top_p",
                    "frequency_penalty",
                    "presence_penalty",
                    "max_tokens",
                    "max_completion_tokens",
                    "stop",
                ],
            )
        else:
            # DEFAULT Case - The vast majority of Nvidia NIM Models lie here
            # "upstage/solar-10.7b-instruct",
            # "snowflake/arctic",
            # "seallms/seallm-7b-v2.5",
            # "nvidia/llama3-chatqa-1.5-8b",
            # "nvidia/llama3-chatqa-1.5-70b",
            # "mistralai/mistral-large",
            # "mistralai/mixtral-8x22b-instruct-v0.1",
            # "mistralai/mixtral-8x7b-instruct-v0.1",
            # "mistralai/mistral-7b-instruct-v0.3",
            # "mistralai/mistral-7b-instruct-v0.2",
            # "mistralai/codestral-22b-instruct-v0.1",
            # "microsoft/phi-3-small-8k-instruct",
            # "microsoft/phi-3-small-128k-instruct",
            # "microsoft/phi-3-mini-4k-instruct",
            # "microsoft/phi-3-mini-128k-instruct",
            # "microsoft/phi-3-medium-4k-instruct",
            # "microsoft/phi-3-medium-128k-instruct",
            # "meta/llama3-70b-instruct",
            # "meta/llama3-8b-instruct",
            # "meta/llama2-70b",
            # "meta/codellama-70b",
            return self._add_reasoning_effort_if_supported(
                model,
                [
                    "stream",
                    "temperature",
                    "top_p",
                    "frequency_penalty",
                    "presence_penalty",
                    "max_tokens",
                    "max_completion_tokens",
                    "stop",
                    "seed",
                    "tools",
                    "tool_choice",
                    "parallel_tool_calls",
                    "response_format",
                ],
            )

    # ------------------------------------------------------------------
    # Token-limit resolution
    # ------------------------------------------------------------------

    def _resolve_token_limit_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
        adaptation_collector: Optional[AdaptationCollector] = None,
    ) -> dict:
        """Deterministically resolve *max_tokens* / *max_completion_tokens*.

        Both the native (metadata-driven) and conservative paths share
        the same alias-reconciliation logic:

        * Only *max_tokens*            -> passed through.
        * Only *max_completion_tokens* -> native: kept as-is;
          conservative: mapped to *max_tokens* (historical compat).
        * Both, equal values           -> deduplicated to *max_tokens*.
        * Both, differing values       -> strict/default raises a
          names-only ``UnsupportedParamsError``; drop/permissive retains
          *max_tokens* and records a value-free adaptation when a
          collector is supplied.

        Stale token-limit keys already present in *optional_params* are
        removed atomically before the resolved field is emitted.
        """
        native_field = self._PROVIDER_NATIVE_TOKEN_LIMIT_FIELD
        alias_field = self._ALIAS_TOKEN_LIMIT_FIELD

        has_native = native_field in non_default_params
        has_alias = alias_field in non_default_params

        if not has_native and not has_alias:
            return optional_params

        supports_native_mct = self._provider_supports_max_completion_tokens(model)

        # -- compute resolved fields (no mutation yet) --------------------
        resolved: dict = {}

        if has_native and has_alias:
            native_val = non_default_params[native_field]
            alias_val = non_default_params[alias_field]
            if native_val == alias_val:
                # Equal aliases: deduplicate to provider-native field.
                resolved[native_field] = native_val
                if adaptation_collector is not None:
                    adaptation_collector.add(
                        name=alias_field,
                        action="renamed",
                        reason="provider_rename",
                    )
            elif not drop_params:
                # Differing aliases, strict/default: names-only error.
                if adaptation_collector is not None:
                    adaptation_collector.add(
                        name=alias_field,
                        action="rejected",
                        reason="unsupported_param",
                    )
                raise UnsupportedParamsError(
                    message=(
                        "Conflicting token limit parameters: "
                        f"'{native_field}' and '{alias_field}' "
                        "have different values. Provide only one."
                    ),
                    llm_provider="nvidia_nim",
                    model=model,
                )
            else:
                # Drop/permissive: retain provider-native field.
                resolved[native_field] = native_val
                if adaptation_collector is not None:
                    adaptation_collector.add(
                        name=alias_field,
                        action="dropped",
                        reason="unsupported_param",
                    )
        elif has_alias:
            if supports_native_mct:
                # Native: alias passes through unchanged.
                resolved[alias_field] = non_default_params[alias_field]
            else:
                # Conservative: historical compat mapping.
                resolved[native_field] = non_default_params[alias_field]
                if adaptation_collector is not None:
                    adaptation_collector.add(
                        name=alias_field,
                        action="renamed",
                        reason="provider_rename",
                    )
        else:
            resolved[native_field] = non_default_params[native_field]

        # -- atomic emit: clear stale, apply resolved ---------------------
        optional_params.pop(native_field, None)
        optional_params.pop(alias_field, None)
        optional_params.update(resolved)

        return optional_params

    # ------------------------------------------------------------------
    # Param mapping
    # ------------------------------------------------------------------

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
        adaptation_collector: Optional[AdaptationCollector] = None,
    ) -> dict:
        supported_openai_params = self.get_supported_openai_params(model=model)

        # Resolve token-limit fields with deterministic policy first.
        optional_params = self._resolve_token_limit_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=drop_params,
            adaptation_collector=adaptation_collector,
        )

        # Map remaining non-token params through the supported list.
        for param, value in non_default_params.items():
            if param in _TOKEN_LIMIT_FIELDS:
                continue  # already resolved above
            if param in supported_openai_params:
                optional_params[param] = value
        return optional_params
