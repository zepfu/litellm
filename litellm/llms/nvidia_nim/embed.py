"""
Nvidia NIM embeddings endpoint: https://docs.api.nvidia.com/nim/reference/nvidia-nv-embedqa-e5-v5-infer

This is OpenAI compatible

This file only contains param mapping logic

API calling is done using the OpenAI SDK with an api_base
"""

from typing import Any, Dict, List, Optional, Set, Tuple

from litellm.exceptions import UnsupportedParamsError
from litellm.litellm_core_utils.param_adaptation import (
    MAX_PARAM_NAME_LENGTH,
    MAX_RECORDS,
    AdaptationCollector,
)


class _NvidiaNimEmbeddingGetConfig:
    def __get__(self, instance, owner):
        if instance is None:
            return self._get_default_config
        return instance._get_instance_config

    @staticmethod
    def _get_default_config():
        return {}


# ---------------------------------------------------------------------------
# Provider policy constants (conservative, provider-wide, no model names)
# ---------------------------------------------------------------------------

# Fields allowed inside the provider request body (extra_body).
_NVIDIA_NIM_EMBED_EXTRA_BODY_ALLOWED: Set[str] = {
    "input_type",
    "truncate",
    "modality",
    "embedding_type",
}

# Keys that must never appear in extra_body (reserved / internal / sensitive).
_NVIDIA_NIM_EMBED_EXTRA_BODY_RESERVED: Set[str] = {
    "api_key",
    "api_base",
    "model",
    "input",
    "extra_body",
    "extra_headers",
    "timeout",
    "stream",
    "stream_options",
    "max_retries",
    "organization",
    "project",
}

# OpenAI top-level params that pass through directly.
_OPENAI_TOP_LEVEL_PARAMS: Set[str] = {"dimensions", "encoding_format", "user"}


def _safe_name(key: Any) -> str:
    """Normalize a dict key to a bounded string name for reporting."""
    return str(key)[:MAX_PARAM_NAME_LENGTH]


def _bounded_names_message(names: List[str], prefix: str) -> str:
    """Build a bounded error message from sorted, deduplicated names."""
    display = names[:MAX_RECORDS]
    msg = prefix + ", ".join(display)
    if len(names) > MAX_RECORDS:
        msg += f" (and {len(names) - MAX_RECORDS} more)"
    return msg


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
        return {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("_") and v is not None
        }

    def get_supported_openai_params(
        self,
    ) -> List[str]:
        return ["encoding_format", "user", "dimensions"]

    # ------------------------------------------------------------------
    # Mapper API
    # ------------------------------------------------------------------

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        kwargs: Optional[dict] = None,
        *,
        strict: bool = False,
        adaptation_collector: Optional[AdaptationCollector] = None,
    ) -> dict:
        """Map OpenAI-style params to NVIDIA NIM embedding request shape.

        When neither *strict* nor *adaptation_collector* is provided the
        mapper preserves legacy behavior: all non-None, non-max_tokens
        kwargs are forwarded into ``extra_body``.  This keeps existing
        direct callers (e.g. ``litellm/utils.py``) working unchanged.

        When *strict* is True or *adaptation_collector* is provided, the
        provider-policy allowlist is enforced atomically: all validation
        runs on staged copies first and *optional_params* is mutated only
        on success.

        Args:
            non_default_params: Params explicitly passed by the caller.
            optional_params: Accumulator dict (mutated atomically on success).
            kwargs: Additional provider kwargs (may contain extra_body).
            strict: If True, policy violations raise UnsupportedParamsError
                with names only (no values).
            adaptation_collector: Optional bounded collector for recording
                dropped/rejected adaptations (value-free).  Providing this
                activates policy enforcement in non-strict (drop) mode.

        Returns:
            The mutated *optional_params* dict.

        Raises:
            UnsupportedParamsError: In strict mode when policy violations
                are found.  The message contains parameter names only.
        """
        policy_active = strict or adaptation_collector is not None

        if not policy_active:
            return self._map_legacy(non_default_params, optional_params, kwargs)

        return self._map_policy(
            non_default_params,
            optional_params,
            kwargs,
            strict=strict,
            adaptation_collector=adaptation_collector,
        )

    # ------------------------------------------------------------------
    # Legacy path (backward-compatible, no policy enforcement)
    # ------------------------------------------------------------------

    @staticmethod
    def _map_legacy(
        non_default_params: dict,
        optional_params: dict,
        kwargs: Optional[dict],
    ) -> dict:
        """Original mapping behavior: forward kwargs to extra_body."""
        if "extra_body" not in optional_params:
            optional_params["extra_body"] = {}
        for k, v in non_default_params.items():
            if v is None or k == "max_tokens":
                continue
            if k in _NVIDIA_NIM_EMBED_EXTRA_BODY_ALLOWED:
                optional_params["extra_body"][k] = v
            else:
                optional_params[k] = v

        if kwargs is not None:
            for key, value in kwargs.items():
                if value is None or key == "max_tokens":
                    continue
                optional_params["extra_body"][key] = value
        return optional_params

    # ------------------------------------------------------------------
    # Policy path (atomic, allowlist-enforced)
    # ------------------------------------------------------------------

    def _map_policy(
        self,
        non_default_params: dict,
        optional_params: dict,
        kwargs: Optional[dict],
        *,
        strict: bool,
        adaptation_collector: Optional[AdaptationCollector],
    ) -> dict:
        # Phase 1: Stage copies (read-only).
        preexisting_eb = optional_params.get("extra_body")
        preexisting_is_dict = isinstance(preexisting_eb, dict)
        staged_extra_body: Dict[Any, Any] = (
            dict(preexisting_eb) if isinstance(preexisting_eb, dict) else {}
        )
        staged_top_level: Dict[str, Any] = {}
        violation_names: List[str] = []

        # Validate preexisting extra_body (non-dict or bad keys).
        preexisting_bad_keys: List[Any] = []
        if not preexisting_is_dict and preexisting_eb is not None:
            violation_names.append("extra_body")
        elif preexisting_is_dict:
            preexisting_bad_keys = self._find_bad_extra_body_keys(staged_extra_body)
            violation_names.extend(_safe_name(k) for k in preexisting_bad_keys)

        self._classify_params(
            non_default_params, staged_extra_body, staged_top_level, violation_names
        )

        explicit_extra_body: Any = None
        if kwargs is not None:
            explicit_extra_body = kwargs.get("extra_body")
            kw_items = {k: v for k, v in kwargs.items() if k != "extra_body"}
            self._classify_params(
                kw_items, staged_extra_body, staged_top_level, violation_names
            )

        extra_body_violations, extra_body_merges = self._validate_explicit_extra_body(
            explicit_extra_body, staged_extra_body
        )
        violation_names.extend(extra_body_violations)

        # Phase 2: Decide (raise or record).
        deduped_sorted = sorted(set(violation_names))
        if deduped_sorted:
            if strict:
                raise UnsupportedParamsError(
                    message=_bounded_names_message(
                        deduped_sorted,
                        "Unsupported parameters for nvidia_nim embeddings: ",
                    ),
                    model="",
                    llm_provider="nvidia_nim",
                )
            self._record_violations(
                deduped_sorted,
                extra_body_violations,
                explicit_extra_body,
                preexisting_is_dict,
                preexisting_eb,
                adaptation_collector,
            )

        # Phase 3: Mutate (only reached if no strict error).
        for key in preexisting_bad_keys:
            staged_extra_body.pop(key, None)
        staged_extra_body.update(extra_body_merges)
        optional_params["extra_body"] = staged_extra_body
        optional_params.update(staged_top_level)
        return optional_params

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_params(
        params: dict,
        staged_extra_body: Dict[Any, Any],
        staged_top_level: Dict[str, Any],
        violation_names: List[str],
    ) -> None:
        """Classify params into extra_body, top-level, or violations."""
        for k, v in params.items():
            if v is None or k == "max_tokens":
                continue
            if not isinstance(k, str):
                violation_names.append(_safe_name(k))
                continue
            if k in _NVIDIA_NIM_EMBED_EXTRA_BODY_ALLOWED:
                staged_extra_body[k] = v
            elif k in _OPENAI_TOP_LEVEL_PARAMS:
                staged_top_level[k] = v
            else:
                violation_names.append(_safe_name(k))

    @staticmethod
    def _validate_explicit_extra_body(
        explicit_extra_body: Any,
        staged_extra_body: Dict[Any, Any],
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Validate explicit extra_body; return (violations, merges)."""
        violations: List[str] = []
        merges: Dict[str, Any] = {}
        if explicit_extra_body is None:
            return violations, merges
        if not isinstance(explicit_extra_body, dict):
            violations.append("extra_body")
            return violations, merges
        for key in sorted(explicit_extra_body.keys(), key=lambda k: str(k)):
            sname = _safe_name(key)
            if not isinstance(key, str):
                violations.append(sname)
            elif key in _NVIDIA_NIM_EMBED_EXTRA_BODY_RESERVED:
                violations.append(sname)
            elif key not in _NVIDIA_NIM_EMBED_EXTRA_BODY_ALLOWED:
                violations.append(sname)
            elif key in staged_extra_body:
                violations.append(sname)
            elif explicit_extra_body[key] is not None:
                merges[key] = explicit_extra_body[key]
        return violations, merges

    @staticmethod
    def _record_violations(
        deduped_sorted: List[str],
        extra_body_violations: List[str],
        explicit_extra_body: Any,
        preexisting_is_dict: bool,
        preexisting_eb: Any,
        adaptation_collector: Optional[AdaptationCollector],
    ) -> None:
        """Record violations in the collector with appropriate reasons."""
        if adaptation_collector is None:
            return
        extra_body_set = set(extra_body_violations)
        for name in deduped_sorted:
            if name == "extra_body" and (
                not isinstance(explicit_extra_body, dict)
                or (not preexisting_is_dict and preexisting_eb is not None)
            ):
                adaptation_collector.add(name, action="rejected", reason="invalid_type")
            elif name in extra_body_set:
                adaptation_collector.add(
                    name, action="rejected", reason="extra_body_policy"
                )
            else:
                adaptation_collector.add(
                    name, action="dropped", reason="unsupported_param"
                )

    @staticmethod
    def _find_bad_extra_body_keys(extra_body: Dict[Any, Any]) -> List[Any]:
        """Return original keys that violate policy (for removal)."""
        bad: List[Any] = []
        for key in list(extra_body.keys()):
            if not isinstance(key, str):
                bad.append(key)
            elif key in _NVIDIA_NIM_EMBED_EXTRA_BODY_RESERVED:
                bad.append(key)
            elif key not in _NVIDIA_NIM_EMBED_EXTRA_BODY_ALLOWED:
                bad.append(key)
        return bad
