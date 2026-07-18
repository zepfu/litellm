"""Provider request-text shaping primitives extracted for RR-054 #1/#44/#54."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class TextSpan:
    start: int
    end: int


def decode_json_prefix(
    text: str,
    *,
    fallback_transform: Optional[Callable[[str], str]] = None,
) -> tuple[str, int]:
    """Return exactly one leading JSON value and its end offset."""
    leading = len(text) - len(text.lstrip())
    source = text[leading:]
    candidates = [source]
    if fallback_transform is not None:
        transformed = fallback_transform(source)
        if transformed != source:
            candidates.append(transformed)
    last_error: Optional[json.JSONDecodeError] = None
    for candidate in candidates:
        try:
            _, decoded_end = json.JSONDecoder().raw_decode(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        return candidate[:decoded_end].strip(), leading + decoded_end
    if last_error is not None:
        raise last_error
    raise json.JSONDecodeError("No JSON value", text, 0)


def iter_delimited_spans(text: str, opener: str, closer: str) -> list[TextSpan]:
    """Find closed delimiter blocks in one forward pass.

    Unclosed openers are skipped without restarting a DOTALL regex from every
    opener, keeping adversarial reminder payloads linear.
    """
    if not text or opener not in text or closer not in text:
        return []
    spans: list[TextSpan] = []
    cursor = 0
    while True:
        start = text.find(opener, cursor)
        if start < 0:
            break
        close_start = text.find(closer, start + len(opener))
        if close_start < 0:
            break
        end = close_start + len(closer)
        while end < len(text) and text[end] in "\r\n":
            end += 1
        spans.append(TextSpan(start=start, end=end))
        cursor = end
    return spans
