"""Generic AAWM alias-routing policy constants.

Configured alias identity, candidate order, and fallback behavior belong only
to the compiled routing snapshot. This module retains provider, lane, cooldown,
quota, allowlist, and adapter-capability policy shared by runtime components.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

# Default cooldowns for auto-agent alias candidates.
CODEX_AUTO_AGENT_DEFAULT_COOLDOWN_SECONDS = 3 * 60 * 60.0
CODEX_AUTO_AGENT_DEFAULT_CAPACITY_COOLDOWN_SECONDS = (
    CODEX_AUTO_AGENT_DEFAULT_COOLDOWN_SECONDS
)
CODEX_AUTO_AGENT_DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS = (
    CODEX_AUTO_AGENT_DEFAULT_COOLDOWN_SECONDS
)
CODEX_AUTO_AGENT_DEFAULT_USAGE_LIMIT_COOLDOWN_SECONDS = (
    CODEX_AUTO_AGENT_DEFAULT_COOLDOWN_SECONDS
)
CODEX_AUTO_AGENT_DEFAULT_TRANSIENT_COOLDOWN_SECONDS = 30.0
CODEX_AUTO_AGENT_CONTINUATION_STATE_UNAVAILABLE_ERROR_CLASS = (
    "continuation_state_unavailable"
)
CODEX_AUTO_AGENT_CONTINUATION_STATE_UNAVAILABLE_COOLDOWN_SECONDS = 300.0

OPENROUTER_FREE_DAILY_QUOTA_MODELS = frozenset(
    {
        "openrouter/cohere/north-mini-code:free",
        "openrouter/owl-alpha",
        "openrouter/stealth/ox-alpha",
    }
)

# Provider and lane identity strings.
CODEX_AUTO_AGENT_NATIVE_PROVIDER = "openai"
CODEX_AUTO_AGENT_OPENROUTER_PROVIDER = "openrouter"
CODEX_AUTO_AGENT_XAI_PROVIDER = "xai"
CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER = "kimi_code"
CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER = "alibaba_token_plan"
CODEX_AUTO_AGENT_ZAI_CODING_PLAN_PROVIDER = "zai_coding_plan"
CODEX_AUTO_AGENT_COHERE_PROVIDER = "cohere"
CODEX_AUTO_AGENT_NOUS_PROVIDER = "nous"
CODEX_AUTO_AGENT_CURSOR_AGENT_PROVIDER = "cursor_agent"
CODEX_AUTO_AGENT_NVIDIA_PROVIDER = "nvidia"
CODEX_AUTO_AGENT_MUSE_CODE_PROVIDER = "muse_code"
OPENCODE_ZEN_PROVIDER = "opencode_zen"
OPENCODE_GO_PROVIDER = "opencode_go"
CODEX_AUTO_AGENT_OPENCODE_PROVIDER = OPENCODE_ZEN_PROVIDER
CODEX_AUTO_AGENT_OPENCODE_GO_PROVIDER = OPENCODE_GO_PROVIDER
ANTHROPIC_AUTO_AGENT_NATIVE_PROVIDER = "anthropic"

# Authoritative OpenCode Go namespace and catalog allowlist. Cost-map rows
# with litellm_provider=opencode_go are stored as opencode/<id>; compiled
# snapshots and the live Go handler use the bare <id> only.
OPENCODE_GO_MODEL_PREFIXES = frozenset({"opencode_go", "opencode-go"})
OPENCODE_GO_ALLOWED_MODELS = frozenset(
    {
        "muse-spark-1.3-contributor",
        "omen-alpha",
        "ox-alpha-free",
    }
)
_OPENCODE_GO_FOREIGN_PREFIXES: dict[str, str] = {
    "alibaba_token_plan": CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER,
    "anthropic": ANTHROPIC_AUTO_AGENT_NATIVE_PROVIDER,
    "chatgpt": CODEX_AUTO_AGENT_NATIVE_PROVIDER,
    "cohere": CODEX_AUTO_AGENT_COHERE_PROVIDER,
    "cursor_agent": CODEX_AUTO_AGENT_CURSOR_AGENT_PROVIDER,
    "kimi_code": CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER,
    "muse_code": CODEX_AUTO_AGENT_MUSE_CODE_PROVIDER,
    "nous": CODEX_AUTO_AGENT_NOUS_PROVIDER,
    "nvidia": CODEX_AUTO_AGENT_NVIDIA_PROVIDER,
    "nvidia_nim": CODEX_AUTO_AGENT_NVIDIA_PROVIDER,
    "opencode": OPENCODE_ZEN_PROVIDER,
    "opencode-zen": OPENCODE_ZEN_PROVIDER,
    "opencode_zen": OPENCODE_ZEN_PROVIDER,
    "openai": CODEX_AUTO_AGENT_NATIVE_PROVIDER,
    "openrouter": CODEX_AUTO_AGENT_OPENROUTER_PROVIDER,
    "xai": CODEX_AUTO_AGENT_XAI_PROVIDER,
    "zai_coding_plan": CODEX_AUTO_AGENT_ZAI_CODING_PLAN_PROVIDER,
    "zen": OPENCODE_ZEN_PROVIDER,
}

CODEX_AUTO_AGENT_OPENROUTER_LANE_KEY = "openrouter"
CODEX_AUTO_AGENT_OPENROUTER_RESPONSES_ROUTE_FAMILY = (
    "codex_auto_agent_openrouter_responses"
)
CODEX_OPENROUTER_COMPLETION_ADAPTER_ROUTE_FAMILY = (
    "codex_openrouter_completion_adapter"
)
ANTHROPIC_OPENROUTER_COMPLETION_ADAPTER_ROUTE_FAMILY = (
    "anthropic_openrouter_completion_adapter"
)
_OPENROUTER_NATIVE_RESPONSES_LEGACY_ROUTE_FAMILIES = frozenset(
    {
        "codex_openrouter_responses",
        "codex_openrouter_responses_adapter",
    }
)
CODEX_AUTO_AGENT_XAI_LANE_KEY = "xai_grok_native"
CODEX_AUTO_AGENT_XAI_OAUTH_LANE_KEY = "xai_oauth_managed"
CODEX_AUTO_AGENT_KIMI_CODE_LANE_KEY = "kimi_code_managed_account"
CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_LANE_KEY = "alibaba_token_plan"
CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_ACCOUNT_QUOTA_COOLDOWN_KEY = (
    "alibaba_token_plan:__account_quota__:alibaba_token_plan"
)
CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_FIVE_HOUR_EXHAUSTED_ERROR_CLASS = (
    "alibaba_token_plan_five_hour_exhausted"
)
CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_WEEKLY_EXHAUSTED_ERROR_CLASS = (
    "alibaba_token_plan_weekly_exhausted"
)
CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_EXHAUSTED_ERROR_CLASSES = frozenset(
    {
        CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_FIVE_HOUR_EXHAUSTED_ERROR_CLASS,
        CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_WEEKLY_EXHAUSTED_ERROR_CLASS,
    }
)
CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_EXHAUSTION_BASE_COOLDOWN_SECONDS = 7200.0
CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_EXHAUSTION_JITTER_SECONDS = 3600.0
CODEX_AUTO_AGENT_ZAI_CODING_PLAN_LANE_KEY = "zai_coding_plan"
CODEX_AUTO_AGENT_ZAI_CODING_PLAN_ACCOUNT_QUOTA_COOLDOWN_KEY = (
    "zai_coding_plan:__account_quota__:zai_coding_plan"
)
CODEX_AUTO_AGENT_COHERE_LANE_KEY = "cohere_native"
CODEX_AUTO_AGENT_NOUS_LANE_KEY = "nous"
CODEX_AUTO_AGENT_CURSOR_AGENT_LANE_KEY = "cursor_agent_cli"
CODEX_AUTO_AGENT_NVIDIA_LANE_KEY = "nvidia_nim"
CODEX_AUTO_AGENT_OPENCODE_LANE_KEY = OPENCODE_ZEN_PROVIDER
CODEX_AUTO_AGENT_OPENCODE_GO_LANE_KEY = OPENCODE_GO_PROVIDER

ANTHROPIC_OPENAI_RESPONSES_ADAPTER_ALLOWED_MODELS = frozenset(
    {
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.5",
        "gpt-5.3-codex-spark",
    }
)
ANTHROPIC_NVIDIA_RESPONSES_ADAPTER_ALLOWED_MODELS = frozenset(
    {
        "deepseek-ai/deepseek-v3.1-terminus",
        "deepseek-ai/deepseek-v3.2",
        "minimaxai/minimax-m2.7",
        "mistralai/devstral-2-123b-instruct-2512",
        "z-ai/glm4.7",
    }
)
ANTHROPIC_OPENROUTER_RESPONSES_ADAPTER_ALLOWED_MODELS = frozenset(
    {
        "openrouter/free",
        "google/gemma-4-31b-it:free",
        "google/gemma-4-26b-a4b-it:free",
        "nvidia/nemotron-3-super-120b-a12b:free",
        "meta-llama/llama-3.3-70b-instruct:free",
        "minimax/minimax-m2.5:free",
        "openai/gpt-oss-20b:free",
        "openai/gpt-oss-120b:free",
        "gpt-oss-20b:free",
        "gpt-oss-120b:free",
        "qwen/qwen3.5-flash-02-23",
        "qwen/qwen3.6-flash",
        "qwen/qwen3-coder:free",
    }
)
ANTHROPIC_OPENROUTER_COMPLETION_ADAPTER_ALLOWED_MODELS = frozenset(
    {
        "cohere/north-mini-code:free",
        "deepseek/deepseek-v4-flash:free",
        "openrouter/elephant-alpha",
        "inclusionai/ling-2.6-flash",
        "owl-alpha",
        "stealth/ox-alpha",
    }
)
# Managed Kimi Code model admission is namespace-based: any explicit
# `kimi_code/<nonempty-model-id>` config route is admissible without a Python
# enumeration. This set retains only the semantic compatibility mappings
# (k3-low/k3-high/k3-max select the K3 model with a forced thinking effort);
# it is not a model-admission gate.
KIMI_CODE_CHAT_COMPLETIONS_ADAPTER_COMPATIBILITY_MAPPINGS = frozenset(
    {
        "kimi_code/k3-low",
        "kimi_code/k3-high",
        "kimi_code/k3-max",
    }
)


def normalize_kimi_code_chat_completions_adapter_model_name(model: Any) -> Optional[str]:
    """Return the canonical `kimi_code/<model-id>` adapter key when admissible.

    Any explicit `kimi_code/<nonempty-model-id>` route is admitted without a
    Python model enumeration. Unprefixed names, foreign-provider names, and
    nested suffixes return `None` so provider inference stays fail-closed and
    the policy boundary matches the downstream Kimi model contract.
    """

    if not isinstance(model, str):
        return None
    candidate = model.strip()
    if not candidate:
        return None
    provider_prefix, separator, model_id = candidate.partition("/")
    if not separator or provider_prefix != CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER:
        return None
    if not model_id.strip() or "/" in model_id:
        return None
    return candidate

def normalize_alibaba_token_plan_adapter_model_name(model: Any) -> Optional[str]:
    """Return the canonical `alibaba_token_plan/<model-id>` adapter key when admissible.

    Any structurally valid explicit `alibaba_token_plan/<nonempty-model-id>`
    route is admitted without a Python model enumeration and the exact model
    ID suffix is forwarded upstream. Unprefixed names, foreign-provider names,
    and nested suffixes (for example `alibaba_token_plan/qwen/sub`) return
    `None`, matching `AlibabaTokenPlanChatConfig._model_id`, so provider
    inference stays fail-closed.
    """

    if not isinstance(model, str):
        return None
    candidate = model.strip()
    if not candidate:
        return None
    provider_prefix, separator, model_id = candidate.partition("/")
    if not separator or provider_prefix != CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER:
        return None
    if not model_id.strip() or "/" in model_id:
        return None
    return candidate


def normalize_zai_coding_plan_adapter_model_name(model: Any) -> Optional[str]:
    """Return the canonical `zai_coding_plan/<model-id>` adapter key when admissible.

    Admission is structural (`zai_coding_plan/<nonempty-id>` without nested `/`)
    plus the documented Coding Plan supported set. Unknown ids fail closed.
    """

    if not isinstance(model, str):
        return None
    candidate = model.strip()
    if not candidate:
        return None
    provider_prefix, separator, model_id = candidate.partition("/")
    if not separator or provider_prefix != CODEX_AUTO_AGENT_ZAI_CODING_PLAN_PROVIDER:
        return None
    normalized_id = model_id.strip()
    if not normalized_id or "/" in normalized_id:
        return None
    from litellm.llms.zai_coding_plan.chat.transformation import (
        ZAI_CODING_PLAN_MODEL_IDS,
    )

    if normalized_id not in ZAI_CODING_PLAN_MODEL_IDS:
        return None
    return candidate


NVIDIA_COMPLETION_ADAPTER_ALLOWED_MODELS = frozenset(
    {
        "deepseek-ai/deepseek-v3.1-terminus",
        "deepseek-ai/deepseek-v3.2",
        "minimaxai/minimax-m2.7",
        "mistralai/devstral-2-123b-instruct-2512",
        "z-ai/glm4.7",
    }
)
NVIDIA_COMPLETION_ADAPTER_MODEL_ALIASES = {
    "minimax/minimax-m2.7": "minimaxai/minimax-m2.7",
}


def is_reserved_openrouter_nvidia_nemotron_free_model(model: Any) -> bool:
    """Return True for reserved OpenRouter ``nvidia/nemotron-*:free`` names.

    Accepts ``nvidia/nemotron-<nonempty>:free`` and
    ``openrouter/nvidia/nemotron-<nonempty>:free``. Non-free NVIDIA NIM names
    such as ``nvidia/nemotron-3-ultra`` are not reserved.
    """

    if not isinstance(model, str):
        return False
    candidate = model.strip().casefold()
    if not candidate:
        return False
    if candidate.startswith("openrouter/"):
        candidate = candidate[len("openrouter/") :]
    prefix = "nvidia/nemotron-"
    suffix = ":free"
    if not candidate.startswith(prefix) or not candidate.endswith(suffix):
        return False
    wildcard = candidate[len(prefix) : -len(suffix)]
    return bool(wildcard)


def canonicalize_openrouter_native_responses_route_family(route_family: Any) -> Any:
    """Rewrite legacy native OpenRouter Responses families to the exact Codex family.

    Completion-adapter identities stay distinct. Empty or non-string values are
    returned unchanged.
    """

    if not isinstance(route_family, str):
        return route_family
    cleaned = route_family.strip()
    if not cleaned:
        return route_family
    lowered = cleaned.casefold()
    if (
        lowered == CODEX_AUTO_AGENT_OPENROUTER_RESPONSES_ROUTE_FAMILY
        or lowered in _OPENROUTER_NATIVE_RESPONSES_LEGACY_ROUTE_FAMILIES
    ):
        return CODEX_AUTO_AGENT_OPENROUTER_RESPONSES_ROUTE_FAMILY
    return route_family


def normalize_openrouter_model_namespace(model: Any) -> Optional[str]:
    """Return the OpenRouter-native model id with the LiteLLM prefix removed.

    Strips surrounding whitespace and at most one leading ``openrouter/``
    LiteLLM provider prefix (case-insensitive). Nested vendor namespaces such
    as ``nvidia/``, ``cohere/``, and ``stealth/`` are preserved, so
    ``openrouter/nvidia/nemotron-x:free`` becomes ``nvidia/nemotron-x:free``
    and ``stealth/ox-alpha`` keeps ``stealth/``.
    """

    if not isinstance(model, str):
        return None
    candidate = model.strip()
    if not candidate:
        return None
    prefix = "openrouter/"
    if candidate[: len(prefix)].casefold() == prefix:
        candidate = candidate[len(prefix) :].strip()
        if not candidate:
            return None
    return candidate.casefold()


# Hosted OpenRouter free aliases that do not carry a ``:free`` suffix.
_OPENROUTER_FREE_MODEL_ALIASES = frozenset(
    {
        "free",
        "elephant-alpha",
        "owl-alpha",
        "stealth/ox-alpha",
    }
)


def is_openrouter_free_model(model: Any) -> bool:
    """Return True for every recognized OpenRouter free-model route.

    A route is free when its normalized name ends with a nonempty ``:free``
    suffix, or when it is one of the hosted free aliases (``free``,
    ``elephant-alpha``, ``owl-alpha``, ``stealth/ox-alpha``). Paid siblings
    such as ``inclusionai/ling-2.6-flash`` and ``nvidia/nemotron-3-ultra``
    are never classified as free.
    """

    candidate = normalize_openrouter_model_namespace(model)
    if candidate is None:
        return False
    if candidate.endswith(":free"):
        return len(candidate) > len(":free")
    return candidate in _OPENROUTER_FREE_MODEL_ALIASES


def normalize_nvidia_completion_adapter_model_name(model: Any) -> Optional[str]:
    """Return the canonical `nvidia/<model-id>` adapter key when admissible.

    Explicit `nvidia/<nonempty-id>` routes are admitted except OpenRouter-namespace
    names such as `nvidia/nemotron-3-super-120b-a12b:free`. Unprefixed names are
    admitted only when they match the closed NVIDIA completion allowlist.
    """

    if not isinstance(model, str):
        return None
    candidate = model.strip()
    if not candidate:
        return None
    if is_reserved_openrouter_nvidia_nemotron_free_model(candidate):
        return None
    provider_prefix, separator, model_id = candidate.partition("/")
    if separator and provider_prefix == CODEX_AUTO_AGENT_NVIDIA_PROVIDER:
        normalized_id = NVIDIA_COMPLETION_ADAPTER_MODEL_ALIASES.get(
            model_id, model_id
        ).strip()
        if not normalized_id:
            return None
        return f"{CODEX_AUTO_AGENT_NVIDIA_PROVIDER}/{normalized_id}"
    normalized_id = NVIDIA_COMPLETION_ADAPTER_MODEL_ALIASES.get(
        candidate, candidate
    ).strip()
    if normalized_id in NVIDIA_COMPLETION_ADAPTER_ALLOWED_MODELS:
        return f"{CODEX_AUTO_AGENT_NVIDIA_PROVIDER}/{normalized_id}"
    return None


def nvidia_completion_adapter_upstream_model(model: Any) -> Optional[str]:
    """Return the NVIDIA NIM upstream model id without the `nvidia/` prefix."""

    canonical = normalize_nvidia_completion_adapter_model_name(model)
    if canonical is None:
        return None
    _prefix, _separator, model_id = canonical.partition("/")
    return model_id or None


def _parse_opencode_go_model(
    model: Any,
) -> tuple[str, Optional[str], Optional[str], Optional[str]]:
    """Classify a Go model spelling without rewriting foreign namespaces.

    Returns ``(kind, canonical, claimed_prefix, foreign_provider)``.
    ``kind`` is one of ``empty``, ``bare``, ``go_prefixed``, ``doubled``,
    ``cross_provider``, or ``unsupported``.
    """

    if not isinstance(model, str):
        return ("empty", None, None, None)
    cleaned = model.strip()
    if not cleaned:
        return ("empty", None, None, None)
    if "/" not in cleaned:
        if cleaned in OPENCODE_GO_ALLOWED_MODELS:
            return ("bare", cleaned, None, None)
        return ("unsupported", cleaned, None, None)

    prefix, remainder = cleaned.split("/", 1)
    prefix = prefix.strip()
    remainder = remainder.strip()
    if prefix in OPENCODE_GO_MODEL_PREFIXES:
        if not remainder:
            return ("empty", None, prefix, None)
        nested_prefix, nested_sep, _nested_rest = remainder.partition("/")
        nested_prefix = nested_prefix.strip()
        if nested_sep:
            if nested_prefix in OPENCODE_GO_MODEL_PREFIXES:
                return ("doubled", None, prefix, None)
            foreign = _OPENCODE_GO_FOREIGN_PREFIXES.get(nested_prefix)
            if foreign is not None:
                return ("cross_provider", None, prefix, foreign)
            return ("unsupported", remainder, prefix, None)
        if remainder in OPENCODE_GO_ALLOWED_MODELS:
            return ("go_prefixed", remainder, prefix, None)
        return ("unsupported", remainder, prefix, None)

    foreign = _OPENCODE_GO_FOREIGN_PREFIXES.get(prefix)
    if foreign is not None:
        return ("cross_provider", None, prefix, foreign)
    return ("unsupported", cleaned, None, None)


OpencodeGoDirectDisposition = Literal["unclaimed", "supported", "empty", "unsupported"]


@dataclass(frozen=True)
class OpencodeGoDirectNamespace:
    """A direct Go prefix claim, kept separate from catalog support.

    ``claimed_prefix`` is set only for ``opencode_go`` or ``opencode-go``.
    Bare catalog ids and foreign namespaces stay ``unclaimed``. ``supported``
    carries the canonical bare suffix; ``empty`` and ``unsupported`` do not.
    """

    disposition: OpencodeGoDirectDisposition
    canonical_model: Optional[str]
    claimed_prefix: Optional[str]


def recognize_opencode_go_direct_namespace(model: Any) -> OpencodeGoDirectNamespace:
    """Recognize either direct Go prefix without applying alias precedence.

    Support is the ``supported`` disposition only. Callers that own direct
    routing must reject ``empty`` and ``unsupported`` before egress. This
    does not rewrite foreign namespaces or reserve bare catalog ids.
    """

    kind, canonical, claimed_prefix, _foreign_provider = _parse_opencode_go_model(model)
    if claimed_prefix not in OPENCODE_GO_MODEL_PREFIXES:
        return OpencodeGoDirectNamespace("unclaimed", None, None)
    if kind == "go_prefixed" and canonical is not None:
        return OpencodeGoDirectNamespace("supported", canonical, claimed_prefix)
    if kind == "empty":
        return OpencodeGoDirectNamespace("empty", None, claimed_prefix)
    return OpencodeGoDirectNamespace("unsupported", None, claimed_prefix)


def canonicalize_opencode_go_alias_model(model: Any) -> str:
    """Return the canonical bare Go upstream id for an alias candidate.

    Approved bare catalog ids and a single ``opencode_go/`` or
    ``opencode-go/`` prefix compile to the same bare id. Empty,
    unsupported, doubled, or cross-provider spellings raise ``ValueError``
    with a migration diagnostic. Foreign provider namespaces are not
    rewritten.
    """

    kind, canonical, claimed_prefix, foreign_provider = _parse_opencode_go_model(
        model
    )
    rendered = model.strip() if isinstance(model, str) else model
    if kind in {"bare", "go_prefixed"} and canonical is not None:
        return canonical
    if kind == "empty":
        if claimed_prefix is not None:
            raise ValueError(
                f"OpenCode Go model {rendered!r} has an empty suffix after "
                f"{claimed_prefix!r}; use a bare catalog id "
                f"({', '.join(sorted(OPENCODE_GO_ALLOWED_MODELS))}) or a "
                "single opencode_go/<id> or opencode-go/<id> prefix"
            )
        raise ValueError(
            "OpenCode Go model is empty; use a bare catalog id "
            f"({', '.join(sorted(OPENCODE_GO_ALLOWED_MODELS))}) or a "
            "single opencode_go/<id> or opencode-go/<id> prefix"
        )
    if kind == "doubled":
        raise ValueError(
            f"OpenCode Go model {rendered!r} has a doubled Go prefix; "
            "use a single opencode_go/ or opencode-go/ prefix, or the bare "
            "catalog id"
        )
    if kind == "cross_provider":
        raise ValueError(
            f"OpenCode Go model {rendered!r} uses {claimed_prefix!r}, "
            f"which is the {foreign_provider!r} namespace; do not rewrite "
            "other providers. Use a bare Go catalog id or a single "
            "opencode_go/<id> or opencode-go/<id> prefix"
        )
    raise ValueError(
        f"OpenCode Go model {rendered!r} is not in the Go catalog "
        f"allowlist {sorted(OPENCODE_GO_ALLOWED_MODELS)}; migrate to a "
        "listed bare id or a single opencode_go/<id> or opencode-go/<id> "
        "prefix"
    )


def normalize_opencode_go_adapter_model_name(model: Any) -> Optional[str]:
    """Return the canonical bare Go id for a supported direct Go prefix.

    Direct requests must use ``opencode_go/<id>`` or ``opencode-go/<id>``.
    Bare catalog ids return ``None`` so they do not steal default
    pass-through. Empty and unsupported claims also return ``None`` here;
    direct routing rejects those dispositions before egress.
    """

    recognized = recognize_opencode_go_direct_namespace(model)
    if recognized.disposition == "supported":
        return recognized.canonical_model
    return None


# Generic compatibility publication for the pass-through integration module.
COMPAT_ALIAS_MAP: dict[str, str] = {
    "_CODEX_AUTO_AGENT_NATIVE_PROVIDER": "CODEX_AUTO_AGENT_NATIVE_PROVIDER",
    "_CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER": "CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER",
    "_CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER": (
        "CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER"
    ),
    "_CODEX_AUTO_AGENT_ZAI_CODING_PLAN_PROVIDER": (
        "CODEX_AUTO_AGENT_ZAI_CODING_PLAN_PROVIDER"
    ),
    "_CODEX_AUTO_AGENT_OPENROUTER_PROVIDER": "CODEX_AUTO_AGENT_OPENROUTER_PROVIDER",
    "_CODEX_AUTO_AGENT_XAI_PROVIDER": "CODEX_AUTO_AGENT_XAI_PROVIDER",
    "_CODEX_AUTO_AGENT_OPENCODE_PROVIDER": "CODEX_AUTO_AGENT_OPENCODE_PROVIDER",
    "_CODEX_AUTO_AGENT_OPENCODE_GO_PROVIDER": (
        "CODEX_AUTO_AGENT_OPENCODE_GO_PROVIDER"
    ),
    "_CODEX_AUTO_AGENT_OPENROUTER_LANE_KEY": (
        "CODEX_AUTO_AGENT_OPENROUTER_LANE_KEY"
    ),
    "_CODEX_AUTO_AGENT_XAI_LANE_KEY": "CODEX_AUTO_AGENT_XAI_LANE_KEY",
    "_CODEX_AUTO_AGENT_XAI_OAUTH_LANE_KEY": (
        "CODEX_AUTO_AGENT_XAI_OAUTH_LANE_KEY"
    ),
    "_CODEX_AUTO_AGENT_KIMI_CODE_LANE_KEY": (
        "CODEX_AUTO_AGENT_KIMI_CODE_LANE_KEY"
    ),
    "_CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_LANE_KEY": (
        "CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_LANE_KEY"
    ),
    "_CODEX_AUTO_AGENT_ZAI_CODING_PLAN_LANE_KEY": (
        "CODEX_AUTO_AGENT_ZAI_CODING_PLAN_LANE_KEY"
    ),
    "_CODEX_AUTO_AGENT_COHERE_PROVIDER": "CODEX_AUTO_AGENT_COHERE_PROVIDER",
    "_CODEX_AUTO_AGENT_COHERE_LANE_KEY": (
        "CODEX_AUTO_AGENT_COHERE_LANE_KEY"
    ),
    "_CODEX_AUTO_AGENT_NOUS_PROVIDER": "CODEX_AUTO_AGENT_NOUS_PROVIDER",
    "_CODEX_AUTO_AGENT_NOUS_LANE_KEY": "CODEX_AUTO_AGENT_NOUS_LANE_KEY",
    "_CODEX_AUTO_AGENT_CURSOR_AGENT_PROVIDER": (
        "CODEX_AUTO_AGENT_CURSOR_AGENT_PROVIDER"
    ),
    "_CODEX_AUTO_AGENT_CURSOR_AGENT_LANE_KEY": (
        "CODEX_AUTO_AGENT_CURSOR_AGENT_LANE_KEY"
    ),
    "_CODEX_AUTO_AGENT_NVIDIA_PROVIDER": "CODEX_AUTO_AGENT_NVIDIA_PROVIDER",
    "_CODEX_AUTO_AGENT_NVIDIA_LANE_KEY": "CODEX_AUTO_AGENT_NVIDIA_LANE_KEY",
    "_CODEX_AUTO_AGENT_OPENCODE_LANE_KEY": "CODEX_AUTO_AGENT_OPENCODE_LANE_KEY",
    "_CODEX_AUTO_AGENT_OPENCODE_GO_LANE_KEY": (
        "CODEX_AUTO_AGENT_OPENCODE_GO_LANE_KEY"
    ),
    "_CODEX_AUTO_AGENT_DEFAULT_COOLDOWN_SECONDS": (
        "CODEX_AUTO_AGENT_DEFAULT_COOLDOWN_SECONDS"
    ),
    "_CODEX_AUTO_AGENT_DEFAULT_CAPACITY_COOLDOWN_SECONDS": (
        "CODEX_AUTO_AGENT_DEFAULT_CAPACITY_COOLDOWN_SECONDS"
    ),
    "_CODEX_AUTO_AGENT_DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS": (
        "CODEX_AUTO_AGENT_DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS"
    ),
    "_CODEX_AUTO_AGENT_DEFAULT_USAGE_LIMIT_COOLDOWN_SECONDS": (
        "CODEX_AUTO_AGENT_DEFAULT_USAGE_LIMIT_COOLDOWN_SECONDS"
    ),
    "_CODEX_AUTO_AGENT_DEFAULT_TRANSIENT_COOLDOWN_SECONDS": (
        "CODEX_AUTO_AGENT_DEFAULT_TRANSIENT_COOLDOWN_SECONDS"
    ),
    "_CODEX_AUTO_AGENT_CONTINUATION_STATE_UNAVAILABLE_ERROR_CLASS": (
        "CODEX_AUTO_AGENT_CONTINUATION_STATE_UNAVAILABLE_ERROR_CLASS"
    ),
    "_CODEX_AUTO_AGENT_CONTINUATION_STATE_UNAVAILABLE_COOLDOWN_SECONDS": (
        "CODEX_AUTO_AGENT_CONTINUATION_STATE_UNAVAILABLE_COOLDOWN_SECONDS"
    ),
    "_ANTHROPIC_AUTO_AGENT_NATIVE_PROVIDER": "ANTHROPIC_AUTO_AGENT_NATIVE_PROVIDER",
    "_ANTHROPIC_OPENAI_RESPONSES_ADAPTER_ALLOWED_MODELS": (
        "ANTHROPIC_OPENAI_RESPONSES_ADAPTER_ALLOWED_MODELS"
    ),
    "_ANTHROPIC_NVIDIA_RESPONSES_ADAPTER_ALLOWED_MODELS": (
        "ANTHROPIC_NVIDIA_RESPONSES_ADAPTER_ALLOWED_MODELS"
    ),
    "_ANTHROPIC_OPENROUTER_RESPONSES_ADAPTER_ALLOWED_MODELS": (
        "ANTHROPIC_OPENROUTER_RESPONSES_ADAPTER_ALLOWED_MODELS"
    ),
    "_ANTHROPIC_OPENROUTER_COMPLETION_ADAPTER_ALLOWED_MODELS": (
        "ANTHROPIC_OPENROUTER_COMPLETION_ADAPTER_ALLOWED_MODELS"
    ),
    "_KIMI_CODE_CHAT_COMPLETIONS_ADAPTER_COMPATIBILITY_MAPPINGS": (
        "KIMI_CODE_CHAT_COMPLETIONS_ADAPTER_COMPATIBILITY_MAPPINGS"
    ),
    "_normalize_kimi_code_chat_completions_adapter_model_name": (
        "normalize_kimi_code_chat_completions_adapter_model_name"
    ),
    "_normalize_alibaba_token_plan_adapter_model_name": (
        "normalize_alibaba_token_plan_adapter_model_name"
    ),
    "_normalize_zai_coding_plan_adapter_model_name": (
        "normalize_zai_coding_plan_adapter_model_name"
    ),
    "_normalize_nvidia_completion_adapter_model_name": (
        "normalize_nvidia_completion_adapter_model_name"
    ),
    "_normalize_opencode_go_adapter_model_name": (
        "normalize_opencode_go_adapter_model_name"
    ),
}
COMPAT_ALIAS_COUNT = len(COMPAT_ALIAS_MAP)


def install_policy_compat_aliases(host_globals: dict[str, Any]) -> None:
    """Install retained generic policy aliases into ``host_globals``."""
    if not isinstance(host_globals, dict):
        raise TypeError(
            f"host_globals must be a dict, got {type(host_globals).__name__}"
        )
    policy_namespace = globals()
    for local_name, policy_name in COMPAT_ALIAS_MAP.items():
        host_globals[local_name] = policy_namespace[policy_name]


__all__ = [
    "ANTHROPIC_AUTO_AGENT_NATIVE_PROVIDER",
    "ANTHROPIC_OPENROUTER_COMPLETION_ADAPTER_ROUTE_FAMILY",
    "ANTHROPIC_NVIDIA_RESPONSES_ADAPTER_ALLOWED_MODELS",
    "ANTHROPIC_OPENAI_RESPONSES_ADAPTER_ALLOWED_MODELS",
    "ANTHROPIC_OPENROUTER_COMPLETION_ADAPTER_ALLOWED_MODELS",
    "ANTHROPIC_OPENROUTER_RESPONSES_ADAPTER_ALLOWED_MODELS",
    "CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_LANE_KEY",
    "CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER",
    "CODEX_AUTO_AGENT_ZAI_CODING_PLAN_LANE_KEY",
    "CODEX_AUTO_AGENT_ZAI_CODING_PLAN_ACCOUNT_QUOTA_COOLDOWN_KEY",
    "CODEX_AUTO_AGENT_ZAI_CODING_PLAN_PROVIDER",
    "CODEX_AUTO_AGENT_COHERE_LANE_KEY",
    "CODEX_AUTO_AGENT_COHERE_PROVIDER",
    "CODEX_AUTO_AGENT_NOUS_LANE_KEY",
    "CODEX_AUTO_AGENT_NOUS_PROVIDER",
    "CODEX_AUTO_AGENT_CURSOR_AGENT_LANE_KEY",
    "CODEX_AUTO_AGENT_CURSOR_AGENT_PROVIDER",
    "CODEX_AUTO_AGENT_NVIDIA_LANE_KEY",
    "CODEX_AUTO_AGENT_NVIDIA_PROVIDER",
    "CODEX_AUTO_AGENT_MUSE_CODE_PROVIDER",
    "CODEX_AUTO_AGENT_DEFAULT_CAPACITY_COOLDOWN_SECONDS",
    "CODEX_AUTO_AGENT_DEFAULT_COOLDOWN_SECONDS",
    "CODEX_AUTO_AGENT_DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS",
    "CODEX_AUTO_AGENT_DEFAULT_TRANSIENT_COOLDOWN_SECONDS",
    "CODEX_AUTO_AGENT_DEFAULT_USAGE_LIMIT_COOLDOWN_SECONDS",
    "CODEX_AUTO_AGENT_CONTINUATION_STATE_UNAVAILABLE_ERROR_CLASS",
    "CODEX_AUTO_AGENT_CONTINUATION_STATE_UNAVAILABLE_COOLDOWN_SECONDS",
    "CODEX_AUTO_AGENT_KIMI_CODE_LANE_KEY",
    "CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER",
    "CODEX_AUTO_AGENT_NATIVE_PROVIDER",
    "CODEX_AUTO_AGENT_OPENCODE_GO_LANE_KEY",
    "CODEX_AUTO_AGENT_OPENCODE_GO_PROVIDER",
    "CODEX_AUTO_AGENT_OPENCODE_LANE_KEY",
    "CODEX_AUTO_AGENT_OPENCODE_PROVIDER",
    "CODEX_AUTO_AGENT_OPENROUTER_LANE_KEY",
    "CODEX_AUTO_AGENT_OPENROUTER_PROVIDER",
    "CODEX_AUTO_AGENT_OPENROUTER_RESPONSES_ROUTE_FAMILY",
    "CODEX_OPENROUTER_COMPLETION_ADAPTER_ROUTE_FAMILY",
    "canonicalize_openrouter_native_responses_route_family",
    "CODEX_AUTO_AGENT_XAI_LANE_KEY",
    "CODEX_AUTO_AGENT_XAI_OAUTH_LANE_KEY",
    "CODEX_AUTO_AGENT_XAI_PROVIDER",
    "COMPAT_ALIAS_COUNT",
    "COMPAT_ALIAS_MAP",
    "KIMI_CODE_CHAT_COMPLETIONS_ADAPTER_COMPATIBILITY_MAPPINGS",
    "NVIDIA_COMPLETION_ADAPTER_ALLOWED_MODELS",
    "NVIDIA_COMPLETION_ADAPTER_MODEL_ALIASES",
    "OPENCODE_GO_ALLOWED_MODELS",
    "OPENCODE_GO_MODEL_PREFIXES",
    "OPENCODE_GO_PROVIDER",
    "OpencodeGoDirectNamespace",
    "OpencodeGoDirectDisposition",
    "OPENCODE_ZEN_PROVIDER",
    "OPENROUTER_FREE_DAILY_QUOTA_MODELS",
    "canonicalize_opencode_go_alias_model",
    "install_policy_compat_aliases",
    "is_openrouter_free_model",
    "is_reserved_openrouter_nvidia_nemotron_free_model",
    "normalize_alibaba_token_plan_adapter_model_name",
    "normalize_kimi_code_chat_completions_adapter_model_name",
    "normalize_nvidia_completion_adapter_model_name",
    "normalize_opencode_go_adapter_model_name",
    "recognize_opencode_go_direct_namespace",
    "normalize_openrouter_model_namespace",
    "normalize_zai_coding_plan_adapter_model_name",
    "nvidia_completion_adapter_upstream_model",
]
