"""Shared credential/OAuth error-message value redaction.

Used by AAWM OAuth refresh scripts so secret *values* keyed by known field
names are redacted consistently (RR-065/074/075/092).

Redacts bare ``key=value`` / ``key: value`` forms, single- and double-quoted
values (including escape/even-backslash cases that would otherwise leave a
quoted suffix), JSON ``"key": "value"`` and unquoted JSON values, and query
or form ``key=value&…`` boundaries. Optionally redacts scoped
``Authorization`` / Bearer credentials in header, assignment, JSON, and
dict/quoted forms without bare-matching English "bearer" prose or ``*_key``
substrings.
"""

from __future__ import annotations

import re
from typing import Callable, Iterable, Optional

DEFAULT_SECRET_FIELD_NAMES = frozenset(
    {
        "access_token",
        "client_secret",
        "id_token",
        "key",
        "refresh_token",
    }
)

# Backward-compatible alias for callers that expect the historical name.
SECRET_FIELD_NAMES = DEFAULT_SECRET_FIELD_NAMES

_REDACTED_VALUE = "[REDACTED]"

# Delimiters that end bare / unquoted secret values (query/form, JSON, prose).
_BARE_VALUE_DELIMS = frozenset(" \t\r\n,;&}]")


def _field_replacement(field_name: str) -> str:
    return f"{field_name}={_REDACTED_VALUE}"


def _consume_quoted_value_for_redaction(message: str, start: int) -> int:
    """Consume a quoted secret value starting at ``start`` (the opening quote).

    Uses escape-aware scanning. When an even-backslash sequence would close the
    string early and leave an immediate non-delimiter suffix that itself ends in
    a matching quote (for example ``"secret\\\\"suffix"``), extend through
    that suffix so partial secret material is not left in the clear.
    """
    if start >= len(message):
        return start
    quote = message[start]
    if quote not in "\"'":
        return start

    i = start + 1
    n = len(message)
    while i < n:
        ch = message[i]
        if ch == "\\" and i + 1 < n:
            # Skip escaped pair (\" \\ etc.); do not treat the second char as close.
            i += 2
            continue
        if ch == quote:
            # Unescaped quote: normal end unless an even-backslash-style suffix
            # leak remains immediately after (non-delimiter run ending in quote).
            j = i + 1
            if j < n and message[j] not in _BARE_VALUE_DELIMS and message[j] != quote:
                k = j
                while k < n and message[k] not in _BARE_VALUE_DELIMS:
                    if message[k] == "\\" and k + 1 < n:
                        k += 2
                        continue
                    if message[k] == quote:
                        # Include the suffix through this later quote and keep
                        # scanning (another suffix may follow).
                        i = k
                        break
                    k += 1
                else:
                    return i + 1
                continue
            return i + 1
        i += 1
    return n


def _consume_secret_value(message: str, start: int) -> int:
    """Return the index just past a secret value starting at ``start``."""
    if start >= len(message):
        return start
    if message.startswith(_REDACTED_VALUE, start):
        return start + len(_REDACTED_VALUE)
    if message[start] in "\"'":
        return _consume_quoted_value_for_redaction(message, start)
    i = start
    n = len(message)
    while i < n and message[i] not in _BARE_VALUE_DELIMS:
        i += 1
    return i


def _redact_field_values(message: str, field_name: str) -> str:
    """Redact every secret value associated with ``field_name``.

    Matches JSON keys (double/single quoted) or bare word-boundary keys with
    ``=``/``:``, then consumes quoted or unquoted values. Replaces each full
    ``key…value`` span with ``field_name=[REDACTED]`` so no secret suffix
    remains.
    """
    escaped = re.escape(field_name)
    key_re = re.compile(
        rf'(?i)(?:"{escaped}"|\'{escaped}\'|\b{escaped}\b)(\s*[:=]\s*)'
    )
    out: list[str] = []
    pos = 0
    for match in key_re.finditer(message):
        if match.start() < pos:
            continue
        value_start = match.end()
        if value_start > len(message):
            continue
        if message.startswith(_REDACTED_VALUE, value_start):
            continue
        value_end = _consume_secret_value(message, value_start)
        out.append(message[pos : match.start()])
        out.append(_field_replacement(field_name))
        pos = value_end
    out.append(message[pos:])
    return "".join(out)


def _authorization_bearer_replacement_header(match: re.Match[str]) -> str:
    prefix = match.group(1)
    sep = match.group(2)
    return f"{prefix}{sep}Bearer {_REDACTED_VALUE}"


def _authorization_bearer_replacement_quoted(match: re.Match[str]) -> str:
    prefix = match.group(1)
    sep = match.group(2)
    quote = match.group(3)
    return f"{prefix}{sep}{quote}Bearer {_REDACTED_VALUE}{quote}"


def _build_authorization_bearer_rules() -> list[
    tuple[re.Pattern[str], Callable[[re.Match[str]], str]]
]:
    # Build patterns without nested-quote footguns.
    dq_body = r'(?:\\.|[^"\\])*'
    sq_body = r"(?:\\.|[^'\\])*"
    bare_token = r"[^\s,;&}\]]+"
    return [
        # Authorization: Bearer token / Authorization=Bearer token
        (
            re.compile(
                rf"(?i)\b(Authorization)\b(\s*[:=]\s*)Bearer\s+{bare_token}"
            ),
            _authorization_bearer_replacement_header,
        ),
        # Authorization="Bearer …" / Authorization='Bearer …'
        (
            re.compile(
                rf'(?i)\b(Authorization)\b(\s*[:=]\s*)(")Bearer\s+{dq_body}"'
            ),
            _authorization_bearer_replacement_quoted,
        ),
        (
            re.compile(
                rf"(?i)\b(Authorization)\b(\s*[:=]\s*)(')Bearer\s+{sq_body}'"
            ),
            _authorization_bearer_replacement_quoted,
        ),
        # JSON / dict: "Authorization": "Bearer …" and quote-style mixes
        (
            re.compile(
                rf'(?i)("Authorization")(\s*:\s*)(")Bearer\s+{dq_body}"'
            ),
            _authorization_bearer_replacement_quoted,
        ),
        (
            re.compile(
                rf"(?i)('Authorization')(\s*:\s*)(')Bearer\s+{sq_body}'"
            ),
            _authorization_bearer_replacement_quoted,
        ),
        (
            re.compile(
                rf'(?i)("Authorization")(\s*:\s*)(\')Bearer\s+{sq_body}\''
            ),
            _authorization_bearer_replacement_quoted,
        ),
        (
            re.compile(
                rf"(?i)('Authorization')(\s*:\s*)(\")Bearer\s+{dq_body}\""
            ),
            _authorization_bearer_replacement_quoted,
        ),
    ]


_AUTHORIZATION_BEARER_RULES = _build_authorization_bearer_rules()


def sanitize_credential_error_message(
    message: str,
    *,
    limit: Optional[int] = None,
    field_names: Optional[Iterable[str]] = None,
    redact_authorization_bearer: bool = True,
) -> str:
    """Redact secret *values* keyed by known field names (not just the labels).

    Handles:

    - ``access_token=…`` / ``refresh_token: …``
    - ``access_token="…"`` / ``access_token='…'`` (escape / even-backslash safe)
    - JSON ``"access_token": "…"`` and unquoted ``"access_token": ya29…``
    - query/form ``access_token=…&refresh_token=…`` (``&`` is a value boundary)
    - optional scoped ``Authorization`` Bearer credentials in header,
      ``Authorization=Bearer …``, quoted assignment, and JSON/dict forms

    Replacements use ``field_name=[REDACTED]`` (canonical name from
    ``field_names``) so no secret suffix remains. When
    ``redact_authorization_bearer`` is true, Authorization Bearer credentials
    become ``…Bearer [REDACTED]`` (key/separator/quote style preserved).

    When ``limit`` is set and the sanitized text is longer, truncate with
    ``...`` so the result length is at most ``limit``.
    """
    sanitized = str(message)
    names = (
        DEFAULT_SECRET_FIELD_NAMES
        if field_names is None
        else frozenset(str(name) for name in field_names if str(name))
    )
    for field_name in names:
        sanitized = _redact_field_values(sanitized, field_name)
    if redact_authorization_bearer:
        for pattern, replacer in _AUTHORIZATION_BEARER_RULES:
            sanitized = pattern.sub(replacer, sanitized)
    if limit is not None and limit > 0 and len(sanitized) > limit:
        if limit <= 3:
            sanitized = sanitized[:limit]
        else:
            sanitized = sanitized[: limit - 3] + "..."
    return sanitized
