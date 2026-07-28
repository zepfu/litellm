"""
Nvidia NIM embeddings endpoint: https://docs.api.nvidia.com/nim/reference/nvidia-nv-embedqa-e5-v5-infer

This is OpenAI compatible 

This file only contains param mapping logic

API calling is done using the OpenAI SDK with an api_base
"""

from typing import Optional


class _NvidiaNimEmbeddingGetConfig:
    def __get__(self, instance, owner):
        if instance is None:
            return self._get_default_config
        return instance._get_instance_config

    @staticmethod
    def _get_default_config():
        return {}


class NvidiaNimEmbeddingConfig:
    """
    Reference: https://docs.api.nvidia.com/nim/reference/nvidia-nv-embedqa-e5-v5-infer
    """

    get_config = _NvidiaNimEmbeddingGetConfig()

    def __init__(
        self,
        encoding_format: Optional[str] = None,
        user: Optional[str] = None,
        input_type: Optional[str] = None,
        truncate: Optional[str] = None,
    ) -> None:
        self.encoding_format = encoding_format
        self.user = user
        self.input_type = input_type
        self.truncate = truncate

    def _get_instance_config(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_") and v is not None}

    def get_supported_openai_params(
        self,
    ):
        return ["encoding_format", "user", "dimensions"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        kwargs: Optional[dict] = None,
    ):
        if "extra_body" not in optional_params:
            optional_params["extra_body"] = {}
        for k, v in non_default_params.items():
            if v is None or k == "max_tokens":
                continue
            if k == "input_type":
                optional_params["extra_body"].update({"input_type": v})
            elif k == "truncate":
                optional_params["extra_body"].update({"truncate": v})
            else:
                optional_params[k] = v

        if kwargs is not None:
            # Pass supported NVIDIA embedding kwargs in extra_body.
            for key, value in kwargs.items():
                if value is None or key == "max_tokens":
                    continue
                optional_params["extra_body"][key] = value
        return optional_params
