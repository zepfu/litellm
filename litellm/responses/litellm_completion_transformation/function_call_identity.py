"""
Responses function_call identity helpers for Chat Completions adapters.

OpenAI Responses function_call items carry two distinct identifiers:
- ``id``: the Responses output-item id (``fc_...``)
- ``call_id``: the upstream tool/call id used to correlate function_call_output

Chat Completions providers typically expose a single tool-call id. Non-native
providers (Kimi/Alibaba/OpenRouter/OpenCode, etc.) often reuse that value for
both fields. This module keeps the provider id exclusively on ``call_id`` and
assigns a stable Responses ``fc_*`` item id when the provider id is not already
a native-looking ``fc_*`` value.
"""

from __future__ import annotations

import hashlib
import re
from typing import Optional, Tuple

# OpenAI-native Responses function_call item ids look like either:
#   fc_685c42deefc0819a822b6936faaa30be0c76bc1491ab6619  (contiguous hex)
#   fc_1fe70e2a-a596-45ef-b72c-9b8567c460e5              (UUID-shaped)
# Require the fc_ prefix plus a reasonably long native body so short/provider
# placeholders (e.g. "fc_1" in fixtures) are treated as non-native and still
# get a stable distinct generated id when needed by production adapters.
_NATIVE_RESPONSES_FUNCTION_CALL_ITEM_ID_RE = re.compile(
    r"^fc_(?:"
    r"[0-9a-fA-F]{24,}"
    r"|"
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
    r")$"
)
_GENERATED_FC_HEX_LEN = 48


def is_native_responses_function_call_item_id(value: Optional[str]) -> bool:
    """Return True when ``value`` already looks like a native Responses fc_* id."""
    if not isinstance(value, str) or not value:
        return False
    return _NATIVE_RESPONSES_FUNCTION_CALL_ITEM_ID_RE.fullmatch(value) is not None


def generate_responses_function_call_item_id(provider_tool_id: str) -> str:
    """
    Build one deterministic stable ``fc_*`` item id from a provider tool id.

    The digest is derived solely from the exact provider id bytes so stream
    deltas, done events, and non-stream conversion agree without session-local
    state, and whitespace-distinct ids cannot collapse.
    """
    digest = hashlib.sha256(provider_tool_id.encode("utf-8")).hexdigest()
    return f"fc_{digest[:_GENERATED_FC_HEX_LEN]}"


def resolve_responses_function_call_identity(
    provider_tool_id: Optional[str],
) -> Tuple[str, str]:
    """
    Map a Chat Completions tool-call id onto Responses ``(id, call_id)``.

    Contract:
    - Empty / whitespace-only provider ids resolve to ``("", "")``.
    - Every other non-empty provider id is preserved byte-for-byte as
      ``call_id`` (no strip/normalize) and is hashed exactly as given.
    - Valid native contiguous-hex or UUID-shaped ``fc_*`` provider ids are left
      unchanged for both fields.
    - Non-native provider ids keep ``call_id`` as the provider value and receive
      a deterministic distinct ``fc_*`` Responses item ``id``.
    """
    if provider_tool_id is None:
        return "", ""
    if not isinstance(provider_tool_id, str):
        provider_tool_id = str(provider_tool_id)

    # Explicit empty behavior: blank / whitespace-only is not a provider id.
    if not provider_tool_id.strip():
        return "", ""

    # Preserve every non-empty provider id byte-for-byte as call_id.
    call_id = provider_tool_id

    if is_native_responses_function_call_item_id(call_id):
        return call_id, call_id

    item_id = generate_responses_function_call_item_id(call_id)
    # Guaranteed distinct for non-native provider ids under the native predicate.
    if item_id == call_id:
        item_id = generate_responses_function_call_item_id(f"item:{call_id}")
    return item_id, call_id
