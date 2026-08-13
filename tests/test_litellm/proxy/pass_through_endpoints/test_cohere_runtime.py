"""Direct tests for the Cohere credential/runtime plumbing (COHERE-001)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional

import pytest

from litellm.proxy.pass_through_endpoints.providers.cohere import (
    runtime as cohere_runtime,
)


@pytest.fixture(autouse=True)
def _reset_runtime_dependencies() -> Any:
    cohere_runtime.configure_cohere_runtime(
        cohere_runtime.DEFAULT_COHERE_RUNTIME_DEPENDENCIES
    )
    yield
    cohere_runtime.configure_cohere_runtime(
        cohere_runtime.DEFAULT_COHERE_RUNTIME_DEPENDENCIES
    )


def _dependencies(
    **overrides: Any,
) -> cohere_runtime.CohereRuntimeDependencies:
    return replace(
        cohere_runtime.DEFAULT_COHERE_RUNTIME_DEPENDENCIES,
        **overrides,
    )


def test_credential_resolution_uses_only_canonical_name() -> None:
    requested_names: list[str] = []

    def get_secret(name: str) -> Optional[str]:
        requested_names.append(name)
        return (
            "cohere-canonical-key"
            if name == cohere_runtime.COHERE_CANONICAL_API_KEY_ENV_VAR
            else "legacy-value-should-not-be-used"
        )

    cohere_runtime.configure_cohere_runtime(
        _dependencies(get_secret=get_secret)
    )

    assert cohere_runtime._require_cohere_api_key() == "cohere-canonical-key"
    assert requested_names == [
        cohere_runtime.COHERE_CANONICAL_API_KEY_ENV_VAR
    ]


def test_credential_cleanup_strips_whitespace_and_quotes() -> None:
    for raw_value, expected in (
        ("  cohere-key-123  ", "cohere-key-123"),
        ('"cohere-key-456"', "cohere-key-456"),
        ("'cohere-key-789'", "cohere-key-789"),
        ('  "cohere-key-012"  ', "cohere-key-012"),
    ):
        cohere_runtime.configure_cohere_runtime(
            _dependencies(
                get_secret=lambda name: (
                    raw_value
                    if name == cohere_runtime.COHERE_CANONICAL_API_KEY_ENV_VAR
                    else None
                )
            )
        )
        assert cohere_runtime._get_cohere_api_key() == expected


def test_legacy_key_names_are_never_consulted() -> None:
    requested_names: list[str] = []

    def get_secret(name: str) -> Optional[str]:
        requested_names.append(name)
        if name in {"COHERE_KEY", "CO_API_KEY"}:
            return "legacy-credential-value"
        return None

    cohere_runtime.configure_cohere_runtime(
        _dependencies(get_secret=get_secret)
    )

    with pytest.raises(cohere_runtime.CohereMissingCredentialError):
        cohere_runtime._require_cohere_api_key()

    assert requested_names == [
        cohere_runtime.COHERE_CANONICAL_API_KEY_ENV_VAR
    ]


def test_missing_credential_fails_with_sanitized_typed_error() -> None:
    cohere_runtime.configure_cohere_runtime(
        _dependencies(get_secret=lambda name: None)
    )

    with pytest.raises(cohere_runtime.CohereMissingCredentialError) as excinfo:
        cohere_runtime._require_cohere_api_key()

    error = excinfo.value
    assert error.type == "authentication_error"
    assert error.code == "401"
    message = error.message
    assert "COHERE_API_KEY" in message
    assert "legacy-credential-value" not in message
    assert "cohere-canonical-key" not in message
    # No credential material may appear in the typed error surface.
    assert "key=" not in message.lower()
    assert "bearer" not in message.lower()


@pytest.mark.parametrize(
    "configured_base,expected",
    [
        (None, "https://api.cohere.com/v2/chat"),
        ("   ", "https://api.cohere.com/v2/chat"),
        ('"  "', "https://api.cohere.com/v2/chat"),
        (
            "https://api.cohere.com/v2/chat",
            "https://api.cohere.com/v2/chat",
        ),
        (
            "  https://api.cohere.com:443/v2/chat  ",
            "https://api.cohere.com/v2/chat",
        ),
        (
            "https://api.cohere.ai/v2/chat",
            "https://api.cohere.ai/v2/chat",
        ),
    ],
)
def test_target_base_normalization(
    configured_base: Optional[str], expected: str
) -> None:
    cohere_runtime.configure_cohere_runtime(
        _dependencies(
            get_secret=lambda name: (
                configured_base
                if name == cohere_runtime.COHERE_API_BASE_ENV_VAR
                else None
            )
        )
    )
    assert cohere_runtime._get_cohere_target_base() == expected


@pytest.mark.parametrize(
    "configured_base",
    [
        "not-a-url",
        "////",
        "http://api.cohere.com/v2/chat",
        "https://127.0.0.1/v2/chat",
        "https://corp.example.test/v2/chat",
        "https://edge.api.cohere.com/v2/chat",
        "https://api.cohere.com/v2/generate",
        "https://api.cohere.com/v2/chat/",
        "https://user:password@api.cohere.com/v2/chat",
        "https://api.cohere.com:8443/v2/chat",
        "https://api.cohere.com/v2/chat?foo=bar",
        "https://api.cohere.com/v2/chat?",
        "https://api.cohere.com/v2/chat#fragment",
        "https://api.cohere.com/v2/chat#",
        "https://[api.cohere.com/v2/chat",
    ],
)
def test_invalid_target_base_fails_with_sanitized_typed_error(
    configured_base: str,
) -> None:
    credential_sentinel = "cohere-credential-sentinel"
    log_messages: list[str] = []

    def get_secret(name: str) -> Optional[str]:
        if name == cohere_runtime.COHERE_API_BASE_ENV_VAR:
            return configured_base
        if name == cohere_runtime.COHERE_CANONICAL_API_KEY_ENV_VAR:
            return credential_sentinel
        return None

    def log_debug(*args: Any, **kwargs: Any) -> None:
        log_messages.append(" ".join(str(arg) for arg in args))
        log_messages.extend(str(value) for value in kwargs.values())

    cohere_runtime.configure_cohere_runtime(
        _dependencies(get_secret=get_secret, log_debug=log_debug)
    )

    with pytest.raises(cohere_runtime.CohereInvalidTargetBaseError) as excinfo:
        cohere_runtime._get_cohere_target_base()

    error = excinfo.value
    assert error.type == "invalid_request_error"
    assert error.code == "400"
    message = error.message
    logged = " ".join(log_messages)
    assert configured_base not in message
    assert configured_base not in logged
    assert credential_sentinel not in message
    assert credential_sentinel not in logged
    assert "COHERE_API_BASE" in message
    assert "/v2/chat" in message
