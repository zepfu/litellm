from typing import Any, Dict, Mapping, Optional

from litellm.llms.base_llm.chat.transformation import BaseLLMException


class OpenRouterException(BaseLLMException):
    pass


class OpenRouterConfigError(ValueError):
    """Provider-specific configuration error for OpenRouter.

    Compatible with ValueError-style config handling so callers that catch
    ValueError will also catch this.  Does NOT reuse the HTTP-response
    OpenRouterException constructor.
    """

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


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


def get_openrouter_auth_headers(
    api_key: Optional[str] = None,
    extra_headers: Optional[Mapping[str, Any]] = None,
) -> Dict[str, str]:
    """Resolve OpenRouter authorization headers with strict precedence.

    Typed, synchronous, dependency-light helper for chat/embedding/image
    validators.

    Precedence:
      1. Caller-supplied Authorization header (case-insensitive lookup in
         *extra_headers*) -- highest precedence ONLY when it is a valid
         non-empty ``Bearer <token>`` credential.  Its actual value/header
         semantics are preserved.
      2. First non-empty/non-whitespace source from:
         explicit *api_key*, ``litellm.api_key``, ``litellm.openrouter_key``,
         env ``OPENROUTER_API_KEY``, ``OR_API_KEY``, ``AAWM_OPENROUTER_API_KEY``.

    Raises:
      OpenRouterConfigError: if the caller Authorization header is present but
        malformed (empty, whitespace-only, non-Bearer, empty token), or if no
        key source is available.
    """
    # --- 1. Caller-supplied Authorization header -------------------------
    caller_auth_value: Any = None
    caller_auth_key: Optional[str] = None
    if extra_headers:
        auth_keys = [k for k in extra_headers if k.lower() == "authorization"]
        if len(auth_keys) > 1:
            raise OpenRouterConfigError(
                "OpenRouter: multiple Authorization headers detected "
                "(case-insensitive). Provide exactly one."
            )
        if auth_keys:
            caller_auth_key = auth_keys[0]
            caller_auth_value = extra_headers[caller_auth_key]

    if caller_auth_key is not None:
        if not isinstance(caller_auth_value, str):
            raise OpenRouterConfigError(
                "OpenRouter: caller-supplied Authorization header must be a "
                "string Bearer credential."
            )
        if _contains_non_printable_characters(caller_auth_value):
            raise OpenRouterConfigError(
                "OpenRouter: caller-supplied Authorization header must not "
                "contain control or non-printable characters."
            )
        stripped = caller_auth_value.strip(" ")
        if not stripped:
            raise OpenRouterConfigError(
                "OpenRouter: caller-supplied Authorization header is empty or "
                "whitespace-only."
            )
        scheme, separator, token_value = stripped.partition(" ")
        if separator != " " or scheme.lower() != "bearer":
            raise OpenRouterConfigError(
                "OpenRouter: caller-supplied Authorization header is not a "
                "valid Bearer credential."
            )
        token = token_value.strip(" ")
        if not token:
            raise OpenRouterConfigError(
                "OpenRouter: caller-supplied Authorization header has an empty "
                "Bearer token."
            )
        if _contains_invalid_credential_characters(token):
            raise OpenRouterConfigError(
                "OpenRouter: caller-supplied Authorization Bearer token must "
                "not contain whitespace or control characters."
            )
        # Preserve the caller's original header key casing and value.
        assert caller_auth_key is not None
        return {caller_auth_key: caller_auth_value}

    # --- 2. Key-source precedence chain ----------------------------------
    resolved_key = _resolve_openrouter_api_key(api_key)
    if resolved_key is None:
        raise OpenRouterConfigError(
            "OpenRouter: no valid credential found. Provide a non-empty "
            "credential through request configuration, LiteLLM configuration, "
            "or the environment."
        )
    return {"Authorization": f"Bearer {resolved_key}"}


def _resolve_openrouter_api_key(
    explicit_key: Optional[str],
) -> Optional[str]:
    """Return the first non-empty/non-whitespace key from the precedence chain.

    Uses lazy imports to avoid litellm initialization cycles.
    """
    # 2a. Explicit parameter
    resolved_key = _normalize_configured_credential(explicit_key)
    if resolved_key is not None:
        return resolved_key

    # 2b. litellm.api_key / litellm.openrouter_key (lazy import)
    import litellm

    for attr in ("api_key", "openrouter_key"):
        resolved_key = _normalize_configured_credential(
            getattr(litellm, attr, None)
        )
        if resolved_key is not None:
            return resolved_key

    # 2c. Environment / secret-manager sources (lazy import)
    from litellm.secret_managers.main import get_secret_str

    for env_name in (
        "OPENROUTER_API_KEY",
        "OR_API_KEY",
        "AAWM_OPENROUTER_API_KEY",
    ):
        resolved_key = _normalize_configured_credential(
            get_secret_str(secret_name=env_name)
        )
        if resolved_key is not None:
            return resolved_key

    return None
