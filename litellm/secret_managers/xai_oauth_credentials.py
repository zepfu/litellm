"""Shared xAI OAuth credential-record selection.

This module is intentionally side-effect free. Request, refresh, health, and
preflight callers all use the same exact-scope policy, while refresh callers
retain ownership of file locking and publication.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any


def looks_like_xai_oauth_credential(value: Mapping[str, Any]) -> bool:
    """Return whether a mapping contains at least one credential field."""

    return any(
        value.get(field_name)
        for field_name in ("key", "access_token", "refresh_token")
    )


def _is_unambiguous_flat_record(payload: Mapping[str, Any]) -> bool:
    if not looks_like_xai_oauth_credential(payload):
        return False

    # A top-level token alongside another credential-like mapping is a mixed
    # document, not a legacy flat record. Require its configured scope.
    return not any(
        isinstance(value, Mapping) and looks_like_xai_oauth_credential(value)
        for value in payload.values()
    )


def select_xai_oauth_credential_record(
    payload: Mapping[str, Any],
    scope: str,
    *,
    provider_label: str = "xAI OAuth",
) -> MutableMapping[str, Any]:
    """Select a mutable credential record without an order-based fallback.

    A legacy flat object remains valid only when it is unambiguous. Every
    multi-record object must contain the exact configured scope key; missing
    scopes fail before callers can make provider/token calls or write files.
    """

    if not isinstance(payload, Mapping):
        raise ValueError(f"{provider_label} auth file must contain a JSON object.")
    if not isinstance(scope, str) or not scope.strip():
        raise ValueError(f"{provider_label} credential scope must not be empty.")

    if _is_unambiguous_flat_record(payload):
        if isinstance(payload, MutableMapping):
            return payload
        raise ValueError(f"{provider_label} credential record is not mutable.")

    scoped_record = payload.get(scope)
    if not isinstance(scoped_record, MutableMapping):
        raise ValueError(
            f"{provider_label} auth file does not contain the configured "
            "credential scope. Exact scope matching is required."
        )
    if not looks_like_xai_oauth_credential(scoped_record):
        raise ValueError(
            f"{provider_label} credential scope does not contain a usable "
            "credential record."
        )
    return scoped_record
