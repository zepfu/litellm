from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

from litellm.llms.base_llm.chat.transformation import BaseLLMException

OPENROUTER_COST_STATUS_UNMAPPED = "unmapped"
OPENROUTER_COST_STATUS_FREE = "free"
OPENROUTER_COST_STATUS_PRICED = "priced"

_logger = logging.getLogger("litellm")

OPENROUTER_PROFILE_SOURCE_AAWM = "aawm"
OPENROUTER_PROFILE_SOURCE_OPENROUTER = "openrouter"
OPENROUTER_PROFILE_SOURCE_OR_ALIAS = "or_alias"
OPENROUTER_PROFILE_SOURCE_REQUEST = "request"
OPENROUTER_PROFILE_SOURCE_LITELLM_API_KEY = "litellm_api_key"
OPENROUTER_PROFILE_SOURCE_LITELLM_OPENROUTER_KEY = "litellm_openrouter_key"
OPENROUTER_PROFILE_SOURCE_NONE = "none"
OPENROUTER_PROFILE_TARGET_FAMILY = "openrouter"

OPENROUTER_API_BASE_VERSION_SEGMENT = "/v1"
"""The single API version segment appended when building OpenRouter transport URLs."""

OPENROUTER_TARGET_BASE_DEFAULT = "https://openrouter.ai/api"
"""Canonical default OpenRouter target root, stored without a version segment."""

_OPENROUTER_KNOWN_ENDPOINT_SUFFIXES: Tuple[str, ...] = (
    "/chat/completions",
    "/responses",
    "/embeddings",
    "/rerank",
)


class OpenRouterException(BaseLLMException):
    pass


def authoritative_openrouter_usage_cost(cost: Any, model: Any) -> Optional[float]:
    """Return OpenRouter usage cost only when it is an authoritative price.

    Positive values are provider-priced. Numeric zero is reserved for models
    explicitly identified as free. Zero on any other model is unavailable
    price metadata, not free, so callers must omit cost instead of recording
    ``0.0``.
    """

    if cost is None or cost == "":
        return None
    try:
        value = float(cost)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value < 0:
        return None
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.policy import (
        is_openrouter_free_model,
    )

    if value == 0 and not is_openrouter_free_model(model):
        return None
    return value


def openrouter_cost_status(*, model: Any, response_cost: Optional[float]) -> str:
    """Stable OpenRouter cost status for session-history and passthrough logs."""

    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.policy import (
        is_openrouter_free_model,
    )

    if is_openrouter_free_model(model):
        return OPENROUTER_COST_STATUS_FREE
    if response_cost is None:
        return OPENROUTER_COST_STATUS_UNMAPPED
    return OPENROUTER_COST_STATUS_PRICED


class OpenRouterConfigError(ValueError):
    """Provider-specific configuration error for OpenRouter.

    Compatible with ValueError-style config handling so callers that catch
    ValueError will also catch this.  Does NOT reuse the HTTP-response
    OpenRouterException constructor.
    """

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


@dataclass(frozen=True)
class OpenRouterProfileNamespace:
    """One OpenRouter configuration namespace: keys, optional custom bases, source id."""

    source: str
    key_envs: Tuple[str, ...]
    base_envs: Tuple[str, ...]


@dataclass(frozen=True)
class OpenRouterResolutionPolicy:
    """Policy knobs for the shared OpenRouter credential and base resolver."""

    include_sdk_globals: bool = True


@dataclass(frozen=True)
class OpenRouterSecretHooks:
    """Optional secret-manager / environment adapters used by proxy runtime tests."""

    get_secret_str: Callable[[str], Optional[Any]]
    getenv: Optional[Callable[[str], Optional[str]]] = None
    clean_secret_string: Optional[Callable[[Optional[str]], Optional[str]]] = None


@dataclass(frozen=True)
class OpenRouterCredentialTargetProfile:
    """Atomic OpenRouter credential-target profile. ``api_key`` is never logged."""

    source: str
    api_key: Optional[str]
    key_env: Optional[str]
    target_base: str
    target_base_env: Optional[str]
    target_family: str = OPENROUTER_PROFILE_TARGET_FAMILY

    def observability(self) -> Dict[str, str]:
        """Secret-safe source identity for logs, spans, and metadata."""

        return {
            "openrouter_profile_source": self.source,
            "openrouter_profile_key_env": self.key_env or "none",
            "openrouter_profile_target_base_env": self.target_base_env or "default",
            "openrouter_profile_target_family": self.target_family,
        }


NATIVE_POLICY = OpenRouterResolutionPolicy(include_sdk_globals=True)
SERVICE_OWNED_POLICY = OpenRouterResolutionPolicy(include_sdk_globals=False)

_OPENROUTER_PROFILE_NAMESPACES: Tuple[OpenRouterProfileNamespace, ...] = (
    OpenRouterProfileNamespace(
        source=OPENROUTER_PROFILE_SOURCE_AAWM,
        key_envs=("AAWM_OPENROUTER_API_KEY",),
        base_envs=("AAWM_OPENROUTER_API_BASE",),
    ),
    OpenRouterProfileNamespace(
        source=OPENROUTER_PROFILE_SOURCE_OPENROUTER,
        key_envs=("OPENROUTER_API_KEY", "OR_API_KEY"),
        base_envs=("OPENROUTER_API_BASE", "OR_API_BASE"),
    ),
)

OPENROUTER_CREDENTIAL_ENV_VARS: Tuple[str, ...] = tuple(
    key_env
    for namespace in _OPENROUTER_PROFILE_NAMESPACES
    for key_env in namespace.key_envs
)


def _contains_invalid_credential_characters(value: str) -> bool:
    return any(
        character.isspace() or not character.isprintable() for character in value
    )


def _contains_non_printable_characters(value: str) -> bool:
    return any(not character.isprintable() for character in value)


def _normalize_configured_credential(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise OpenRouterConfigError(
            "OpenRouter: configured credential must be a string."
        )

    stripped_value = value.strip()
    if not stripped_value:
        return None
    if _contains_invalid_credential_characters(stripped_value):
        raise OpenRouterConfigError(
            "OpenRouter: configured credential must be one printable token "
            "without whitespace or control characters."
        )
    return stripped_value


def _default_get_secret_str(secret_name: str) -> Optional[Any]:
    from litellm.secret_managers.main import get_secret_str

    return get_secret_str(secret_name=secret_name)


def _clean_hook_secret(
    value: Optional[Any],
    hooks: Optional[OpenRouterSecretHooks],
) -> Optional[Any]:
    if hooks is None or hooks.clean_secret_string is None:
        return value
    if value is None or isinstance(value, str):
        return hooks.clean_secret_string(value)
    return value


def _read_secret(
    secret_name: str,
    hooks: Optional[OpenRouterSecretHooks],
) -> Optional[Any]:
    getter = (
        hooks.get_secret_str if hooks is not None else _default_get_secret_str
    )
    value = getter(secret_name)
    if value is None and hooks is not None and hooks.getenv is not None:
        value = hooks.getenv(secret_name)
    return _clean_hook_secret(value, hooks)


def _normalize_configured_url(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise OpenRouterConfigError(
            "OpenRouter: configured API base must be a string."
        )
    stripped_value = value.strip()
    if not stripped_value:
        return None
    if _contains_non_printable_characters(stripped_value) or any(
        character.isspace() for character in stripped_value
    ):
        raise OpenRouterConfigError(
            "OpenRouter: configured API base must be one printable URL "
            "without whitespace or control characters."
        )
    return stripped_value


def _strip_known_endpoint_suffix(path: str) -> str:
    normalized = path if path.startswith("/") else f"/{path}"
    for suffix in _OPENROUTER_KNOWN_ENDPOINT_SUFFIXES:
        if normalized.endswith(suffix):
            remainder = normalized[: -len(suffix)]
            return remainder if remainder else "/"
    return normalized


def _openrouter_target_base_api_version_violation(target_base: str) -> Optional[str]:
    """Return a deterministic rejection reason for ambiguous version paths.

    The canonical OpenRouter target root never carries a version segment:
    exactly one trailing ``/v1`` is normalized away, and any additional
    version segment nested inside the path is rejected instead of silently
    rewritten.
    """

    parsed = urlsplit(target_base)
    path = _strip_known_endpoint_suffix(parsed.path or "/")
    segments = [segment for segment in path.split("/") if segment]
    version_indexes = [
        index
        for index, segment in enumerate(segments)
        if segment == OPENROUTER_API_BASE_VERSION_SEGMENT.lstrip("/")
    ]
    if not version_indexes:
        return None
    if version_indexes == [len(segments) - 1]:
        return None
    return (
        "OpenRouter API base target must carry at most one trailing "
        f"'{OPENROUTER_API_BASE_VERSION_SEGMENT}' version segment; nested "
        "version paths are ambiguous and are rejected instead of rewritten."
    )


def canonical_openrouter_target_base(target_base: str) -> str:
    """Normalize one OpenRouter target root without any version segment."""

    cleaned = _normalize_configured_url(target_base)
    if cleaned is None:
        return OPENROUTER_TARGET_BASE_DEFAULT

    try:
        parsed = urlsplit(cleaned)
    except ValueError as exc:
        raise OpenRouterConfigError(
            "OpenRouter: configured API base is not a valid URL."
        ) from exc

    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise OpenRouterConfigError(
            "OpenRouter: configured API base must be an absolute http(s) URL."
        )
    if parsed.username is not None or parsed.password is not None:
        raise OpenRouterConfigError(
            "OpenRouter: configured API base must not embed credentials."
        )
    if parsed.query or parsed.fragment:
        raise OpenRouterConfigError(
            "OpenRouter: configured API base must not include a query or fragment."
        )

    violation = _openrouter_target_base_api_version_violation(cleaned)
    if violation is not None:
        raise OpenRouterConfigError(violation)

    path = _strip_known_endpoint_suffix(parsed.path or "/")
    segments = [segment for segment in path.split("/") if segment]
    if segments and segments[-1] == OPENROUTER_API_BASE_VERSION_SEGMENT.lstrip("/"):
        segments = segments[:-1]
    normalized_path = "/" + "/".join(segments) if segments else ""
    canonical = urlunsplit((parsed.scheme, parsed.netloc, normalized_path, "", ""))
    return canonical.rstrip("/") or OPENROUTER_TARGET_BASE_DEFAULT


def openrouter_api_base_from_target_base(target_base: str) -> str:
    """Build the OpenRouter API base by appending the version segment once."""

    return (
        f"{canonical_openrouter_target_base(target_base)}"
        f"{OPENROUTER_API_BASE_VERSION_SEGMENT}"
    )


def openrouter_complete_url(target_base: str, endpoint: str) -> str:
    """Join a canonical OpenRouter root with one versioned endpoint exactly once."""

    api_base = openrouter_api_base_from_target_base(target_base)
    suffix = "/" + endpoint.lstrip("/")
    if api_base.endswith(suffix):
        return api_base
    return f"{api_base}{suffix}"


def _empty_openrouter_credential_target_profile() -> OpenRouterCredentialTargetProfile:
    return OpenRouterCredentialTargetProfile(
        source=OPENROUTER_PROFILE_SOURCE_NONE,
        api_key=None,
        key_env=None,
        target_base=OPENROUTER_TARGET_BASE_DEFAULT,
        target_base_env=None,
        target_family=OPENROUTER_PROFILE_TARGET_FAMILY,
    )


def _namespace_source_for_key(namespace: OpenRouterProfileNamespace, key_env: str) -> str:
    if key_env == "OR_API_KEY":
        return OPENROUTER_PROFILE_SOURCE_OR_ALIAS
    return namespace.source


def _read_namespace_base(
    namespace: OpenRouterProfileNamespace,
    hooks: Optional[OpenRouterSecretHooks],
) -> Tuple[Optional[str], Optional[str]]:
    for base_env in namespace.base_envs:
        raw_base = _normalize_configured_url(_read_secret(base_env, hooks))
        if raw_base is not None:
            return raw_base, base_env
    return None, None


def _profile_for_key(
    *,
    source: str,
    api_key: str,
    key_env: str,
    raw_base: Optional[str],
    target_base_env: Optional[str],
) -> OpenRouterCredentialTargetProfile:
    target_base = (
        canonical_openrouter_target_base(raw_base)
        if raw_base is not None
        else OPENROUTER_TARGET_BASE_DEFAULT
    )
    return OpenRouterCredentialTargetProfile(
        source=source,
        api_key=api_key,
        key_env=key_env,
        target_base=target_base,
        target_base_env=target_base_env if raw_base is not None else None,
        target_family=OPENROUTER_PROFILE_TARGET_FAMILY,
    )


def _log_openrouter_profile_observability(
    profile: OpenRouterCredentialTargetProfile,
) -> None:
    observability = profile.observability()
    _logger.debug(
        "OpenRouter credential-target profile source=%s key_env=%s "
        "target_base_env=%s target_family=%s",
        observability["openrouter_profile_source"],
        observability["openrouter_profile_key_env"],
        observability["openrouter_profile_target_base_env"],
        observability["openrouter_profile_target_family"],
    )


def resolve_openrouter_credential_target(
    *,
    explicit_api_key: Optional[str] = None,
    explicit_api_base: Optional[str] = None,
    policy: OpenRouterResolutionPolicy = NATIVE_POLICY,
    hooks: Optional[OpenRouterSecretHooks] = None,
) -> OpenRouterCredentialTargetProfile:
    """Select one service-owned OpenRouter key+base profile; never mix namespaces.

    Precedence:
      1. Explicit request ``api_key`` (deployment/request configuration).
      2. Optional SDK globals ``litellm.api_key`` then ``litellm.openrouter_key``
         when ``policy.include_sdk_globals`` is true.
      3. Env/secret namespaces in AAWM, then OpenRouter, then ``OR_API_KEY``
         alias order.

    The winning key namespace owns the target. A secret-managed custom base is
    honored only when it belongs to that namespace. ``explicit_api_base`` is
    request-scoped and always wins over the namespace base. Caller Authorization
    headers are not a credential source.
    """

    explicit_base = _normalize_configured_url(
        _clean_hook_secret(explicit_api_base, hooks)
    )

    resolved_key = _normalize_configured_credential(
        _clean_hook_secret(explicit_api_key, hooks)
    )
    if resolved_key is not None:
        profile = _profile_for_key(
            source=OPENROUTER_PROFILE_SOURCE_REQUEST,
            api_key=resolved_key,
            key_env="api_key",
            raw_base=explicit_base,
            target_base_env="api_base" if explicit_base is not None else None,
        )
        _log_openrouter_profile_observability(profile)
        return profile

    if policy.include_sdk_globals:
        import litellm

        for attr, source, key_env in (
            (
                "api_key",
                OPENROUTER_PROFILE_SOURCE_LITELLM_API_KEY,
                "litellm.api_key",
            ),
            (
                "openrouter_key",
                OPENROUTER_PROFILE_SOURCE_LITELLM_OPENROUTER_KEY,
                "litellm.openrouter_key",
            ),
        ):
            resolved_key = _normalize_configured_credential(
                getattr(litellm, attr, None)
            )
            if resolved_key is None:
                continue
            profile = _profile_for_key(
                source=source,
                api_key=resolved_key,
                key_env=key_env,
                raw_base=explicit_base,
                target_base_env="api_base" if explicit_base is not None else None,
            )
            _log_openrouter_profile_observability(profile)
            return profile

    for namespace in _OPENROUTER_PROFILE_NAMESPACES:
        winning_key: Optional[str] = None
        winning_key_env: Optional[str] = None
        for key_env in namespace.key_envs:
            resolved_key = _normalize_configured_credential(
                _read_secret(key_env, hooks)
            )
            if resolved_key is None:
                continue
            winning_key = resolved_key
            winning_key_env = key_env
            break
        if winning_key is None or winning_key_env is None:
            continue
        raw_base, base_env = (explicit_base, "api_base")
        if raw_base is None:
            raw_base, base_env = _read_namespace_base(namespace, hooks)
        profile = _profile_for_key(
            source=_namespace_source_for_key(namespace, winning_key_env),
            api_key=winning_key,
            key_env=winning_key_env,
            raw_base=raw_base,
            target_base_env=base_env,
        )
        _log_openrouter_profile_observability(profile)
        return profile

    if explicit_base is not None:
        profile = OpenRouterCredentialTargetProfile(
            source=OPENROUTER_PROFILE_SOURCE_NONE,
            api_key=None,
            key_env=None,
            target_base=canonical_openrouter_target_base(explicit_base),
            target_base_env="api_base",
            target_family=OPENROUTER_PROFILE_TARGET_FAMILY,
        )
        _log_openrouter_profile_observability(profile)
        return profile

    profile = _empty_openrouter_credential_target_profile()
    _log_openrouter_profile_observability(profile)
    return profile


def _reject_duplicate_authorization_headers(
    extra_headers: Optional[Mapping[str, Any]],
) -> None:
    if not extra_headers:
        return
    auth_keys = [key for key in extra_headers if key.lower() == "authorization"]
    if len(auth_keys) > 1:
        raise OpenRouterConfigError(
            "OpenRouter: multiple Authorization headers detected "
            "(case-insensitive). Provide exactly one."
        )


def get_openrouter_auth_headers(
    api_key: Optional[str] = None,
    extra_headers: Optional[Mapping[str, Any]] = None,
    *,
    api_base: Optional[str] = None,
    policy: OpenRouterResolutionPolicy = NATIVE_POLICY,
    hooks: Optional[OpenRouterSecretHooks] = None,
) -> Dict[str, str]:
    """Resolve OpenRouter authorization headers from the service-owned profile.

    Caller Authorization is detected only to reject duplicate header names.
    It is never admitted as a credential and cannot override the service key.
    ``api_base`` is the same request-scoped explicit base used for URL
    resolution, so an unused environment base is not canonicalized.

    Raises:
      OpenRouterConfigError: if multiple Authorization headers are present, a
        configured credential/base is malformed, or no key source is available.
    """
    _reject_duplicate_authorization_headers(extra_headers)
    profile = resolve_openrouter_credential_target(
        explicit_api_key=api_key,
        explicit_api_base=api_base,
        policy=policy,
        hooks=hooks,
    )
    if not profile.api_key:
        raise OpenRouterConfigError(
            "OpenRouter API key is required. OpenRouter: no valid credential "
            "found. Provide a non-empty credential through request "
            "configuration, LiteLLM configuration, or the environment."
        )
    return {"Authorization": f"Bearer {profile.api_key}"}


def apply_openrouter_auth_headers(
    headers: Mapping[str, Any],
    api_key: Optional[str] = None,
    *,
    api_base: Optional[str] = None,
    policy: OpenRouterResolutionPolicy = NATIVE_POLICY,
    hooks: Optional[OpenRouterSecretHooks] = None,
) -> Dict[str, Any]:
    """Return headers with exactly one service-owned Authorization value."""

    validated_headers = dict(headers)
    auth_headers = get_openrouter_auth_headers(
        api_key=api_key,
        extra_headers=validated_headers,
        api_base=api_base,
        policy=policy,
        hooks=hooks,
    )
    for key in [key for key in validated_headers if key.lower() == "authorization"]:
        del validated_headers[key]
    validated_headers.update(auth_headers)
    return validated_headers


def resolve_openrouter_complete_url(
    endpoint: str,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    *,
    policy: OpenRouterResolutionPolicy = NATIVE_POLICY,
    hooks: Optional[OpenRouterSecretHooks] = None,
) -> str:
    """Resolve a versioned OpenRouter endpoint URL from the shared profile.

    The URL is derived from the same key+base profile that owns Authorization.
    Passing ``api_base`` without ``api_key`` would otherwise select an env
    namespace target while a request key is sent to that host.
    """

    profile = resolve_openrouter_credential_target(
        explicit_api_key=api_key,
        explicit_api_base=api_base,
        policy=policy,
        hooks=hooks,
    )
    return openrouter_complete_url(profile.target_base, endpoint)
