"""Read-only Hermes Nous Portal invoke-JWT loader.

LiteLLM request handling is a consumer of ``~/.hermes/auth.json``. The
provider-status sidecar is the sole automatic writer. This module never
writes Hermes state and never treats xAI or Copilot slots as Nous.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from litellm.secret_managers.credential_error_sanitizer import (
    sanitize_credential_error_message,
)

_DEFAULT_HERMES_AUTH_PATH = os.path.join("~", ".hermes", "auth.json")
write_and_publish_private_text = None


def resolve_hermes_nous_auth_path() -> str:
    litellm_path = os.environ.get("LITELLM_HERMES_AUTH_FILE")
    if litellm_path:
        return litellm_path
    aawm_path = os.environ.get("AAWM_HERMES_AUTH_FILE")
    if aawm_path:
        return aawm_path
    return os.path.expanduser(_DEFAULT_HERMES_AUTH_PATH)


def _usable_access_token(slot: Any) -> Optional[str]:
    if not isinstance(slot, dict):
        return None
    token = slot.get("access_token")
    if isinstance(token, str) and token.strip():
        return token
    return None


def _slot_from_document(payload: Any, collection: str) -> Any:
    if not isinstance(payload, dict):
        return None
    group = payload.get(collection)
    if not isinstance(group, dict):
        return None
    return group.get("nous")


def load_nous_invoke_jwt() -> str:
    path = resolve_hermes_nous_auth_path()
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        token = _usable_access_token(_slot_from_document(payload, "providers"))
        if token is not None:
            return token
        token = _usable_access_token(_slot_from_document(payload, "credential_pool"))
        if token is not None:
            return token
        raise RuntimeError("Nous Portal invoke JWT is missing from Hermes auth")
    except Exception as exc:
        message = sanitize_credential_error_message(str(exc))
        raise type(exc)(message) from None
