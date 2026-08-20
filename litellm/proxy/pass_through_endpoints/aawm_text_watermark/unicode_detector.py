"""Deterministic Unicode-carrier inspection and conservative sanitation."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Optional

ZWSP = "\u200b"
ZWNJ = "\u200c"
ZWJ = "\u200d"
WORD_JOINER = "\u2060"
BOM = "\ufeff"
MONGOLIAN_VOWEL_SEPARATOR = "\u180e"
COMBINING_ENCLOSING_KEYCAP = "\u20e3"

UNICODE_CARRIER_DETECTOR_NAME = "unicode_carrier"
UNICODE_CARRIER_DETECTOR_VERSION = 1

_EXOTIC_SPACES = frozenset(
    {
        "\u1680",
        "\u2000",
        "\u2001",
        "\u2002",
        "\u2003",
        "\u2004",
        "\u2005",
        "\u2006",
        "\u2007",
        "\u2008",
        "\u2009",
        "\u200a",
        "\u202f",
        "\u205f",
        "\u3000",
    }
)

_UNSAFE_ISOLATED_FORMAT = frozenset(
    {
        ZWSP,
        WORD_JOINER,
        BOM,
        MONGOLIAN_VOWEL_SEPARATOR,
    }
)

_EMOJI_RANGES = (
    (0x231A, 0x231B),
    (0x23E9, 0x23F3),
    (0x23F8, 0x23FA),
    (0x25AA, 0x25AB),
    (0x25B6, 0x25B6),
    (0x25C0, 0x25C0),
    (0x25FB, 0x25FE),
    (0x2600, 0x27BF),
    (0x2934, 0x2935),
    (0x2B05, 0x2B07),
    (0x2B1B, 0x2B1C),
    (0x2B50, 0x2B50),
    (0x2B55, 0x2B55),
    (0x3030, 0x3030),
    (0x303D, 0x303D),
    (0x3297, 0x3297),
    (0x3299, 0x3299),
    (0x1F000, 0x1F02F),
    (0x1F0A0, 0x1F0FF),
    (0x1F100, 0x1F1FF),
    (0x1F200, 0x1F2FF),
    (0x1F300, 0x1FAFF),
    (0x1FAC0, 0x1FAFF),
)

_CJK_RANGES = (
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0x20000, 0x2A6DF),
    (0x2A700, 0x2B73F),
    (0x2B740, 0x2B81F),
    (0x2B820, 0x2CEAF),
    (0x2CEB0, 0x2EBEF),
    (0x30000, 0x3134F),
    (0x2F800, 0x2FA1F),
)

_KEYCAP_BASES = frozenset("0123456789*#")
_REGIONAL_INDICATOR = range(0x1F1E6, 0x1F1FF + 1)
_SKIN_TONE = range(0x1F3FB, 0x1F3FF + 1)
_TAGS = range(0xE0020, 0xE007F + 1)
_VS_BMP = range(0xFE00, 0xFE0F + 1)
_VS_SUPP = range(0xE0100, 0xE01EF + 1)
_MONGOLIAN_FVS = frozenset({"\u180b", "\u180c", "\u180d", "\u180f"})
_MONGOLIAN = range(0x1800, 0x18AF + 1)

CarrierAction = Literal["report", "remove", "replace_space"]


@dataclass(frozen=True)
class CarrierHit:
    index: int
    kind: str
    action: CarrierAction


@dataclass(frozen=True)
class UnicodeCarrierDetection:
    signal_detected: bool
    confirmed_watermark_detected: bool = False
    vendor_attribution: str = "unknown"
    hit_kinds: tuple[str, ...] = ()
    hit_count: int = 0
    hits: tuple[CarrierHit, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class UnicodeSanitizeResult:
    text: str
    removed_count: int = 0
    replaced_count: int = 0
    detection: UnicodeCarrierDetection = field(
        default_factory=lambda: UnicodeCarrierDetection(signal_detected=False)
    )


def _in_ranges(cp: int, ranges: Iterable[tuple[int, int]]) -> bool:
    return any(start <= cp <= end for start, end in ranges)


def _is_noncharacter(cp: int) -> bool:
    if 0xFDD0 <= cp <= 0xFDEF:
        return True
    return cp >= 0xFFFE and (cp & 0xFFFE) == 0xFFFE and cp <= 0x10FFFF


def _is_variation_selector(ch: str) -> bool:
    cp = ord(ch)
    return cp in _VS_BMP or cp in _VS_SUPP or ch in _MONGOLIAN_FVS


def _is_emoji_extender(ch: str) -> bool:
    cp = ord(ch)
    return (
        _is_variation_selector(ch)
        or cp in _SKIN_TONE
        or cp in _TAGS
        or ch == COMBINING_ENCLOSING_KEYCAP
    )


def _is_emoji_base(ch: str) -> bool:
    cp = ord(ch)
    if ch in _KEYCAP_BASES:
        return True
    if cp in _REGIONAL_INDICATOR or cp in _SKIN_TONE:
        return True
    if _in_ranges(cp, _EMOJI_RANGES):
        return True
    name = unicodedata.name(ch, "")
    return "EMOJI" in name


def _is_cjk(ch: str) -> bool:
    return _in_ranges(ord(ch), _CJK_RANGES)


def _is_mongolian(ch: str) -> bool:
    return ord(ch) in _MONGOLIAN


def _prev_base_index(text: str, index: int) -> Optional[int]:
    j = index - 1
    while j >= 0 and _is_emoji_extender(text[j]):
        j -= 1
    return j if j >= 0 else None


def _next_base_index(text: str, index: int) -> Optional[int]:
    j = index + 1
    n = len(text)
    while j < n and _is_emoji_extender(text[j]):
        j += 1
    return j if j < n else None


def _is_emoji_zwj_glue(text: str, index: int) -> bool:
    left = _prev_base_index(text, index)
    right = _next_base_index(text, index)
    if left is None or right is None:
        return False
    return _is_emoji_base(text[left]) and _is_emoji_base(text[right])


def _is_context_valid_variation_selector(text: str, index: int) -> bool:
    left = _prev_base_index(text, index)
    if left is None:
        return False
    ch = text[left]
    if _is_emoji_base(ch) or ch in _KEYCAP_BASES:
        return True
    if _is_cjk(ch) or _is_mongolian(ch):
        return True
    return False


def _is_letter_joiner_context(text: str, index: int) -> bool:
    if index <= 0 or index + 1 >= len(text):
        return False
    left = unicodedata.category(text[index - 1])
    right = unicodedata.category(text[index + 1])
    return left.startswith("L") and right.startswith("L")


def _normalize_policy(policy: Any) -> str:
    if policy is None:
        return "conservative"
    name = getattr(policy, "policy", policy)
    text = str(name).strip().lower()
    if text not in {"conservative", "aggressive"}:
        raise ValueError(f"unsupported unicode watermark policy: {policy!r}")
    return text


def iter_carrier_hits(
    text: str,
    *,
    policy: str = "conservative",
    normalize_spaces: bool = True,
) -> tuple[CarrierHit, ...]:
    """Return ordered carrier hits. Never includes raw characters in ``kind``."""

    policy_name = _normalize_policy(policy)
    aggressive = policy_name == "aggressive"
    hits: list[CarrierHit] = []
    for index, ch in enumerate(text):
        cp = ord(ch)
        if _is_noncharacter(cp):
            hits.append(CarrierHit(index, "noncharacter", "remove"))
            continue
        if ch in _EXOTIC_SPACES:
            action: CarrierAction = (
                "replace_space" if normalize_spaces else "report"
            )
            hits.append(CarrierHit(index, "exotic_space", action))
            continue
        if ch == ZWSP:
            hits.append(CarrierHit(index, "zero_width_space", "remove"))
            continue
        if ch == ZWJ:
            if not aggressive and _is_emoji_zwj_glue(text, index):
                continue
            hits.append(CarrierHit(index, "zwj", "remove"))
            continue
        if ch == ZWNJ:
            if not aggressive and _is_letter_joiner_context(text, index):
                continue
            hits.append(CarrierHit(index, "format_control", "remove"))
            continue
        if _is_variation_selector(ch):
            context_valid = _is_context_valid_variation_selector(text, index)
            if not aggressive and context_valid:
                continue
            action = "remove" if aggressive else "report"
            hits.append(CarrierHit(index, "variation_selector", action))
            continue
        if ch in _UNSAFE_ISOLATED_FORMAT:
            hits.append(CarrierHit(index, "format_control", "remove"))
            continue
        if aggressive and unicodedata.category(ch) == "Cf":
            hits.append(CarrierHit(index, "format_control", "remove"))
            continue
        if aggressive and cp in _TAGS:
            hits.append(CarrierHit(index, "format_control", "remove"))
    return tuple(hits)


def _unique_kinds(hits: Iterable[CarrierHit]) -> tuple[str, ...]:
    seen: set[str] = set()
    kinds: list[str] = []
    for hit in hits:
        if hit.kind in seen:
            continue
        seen.add(hit.kind)
        kinds.append(hit.kind)
    return tuple(kinds)


def detect_unicode_carriers(
    text: str,
    policy: str = "conservative",
    *,
    normalize_spaces: bool = True,
) -> UnicodeCarrierDetection:
    """Inspect text for configured Unicode carriers. Never claims vendor attribution."""

    hits = iter_carrier_hits(
        text, policy=policy, normalize_spaces=normalize_spaces
    )
    return UnicodeCarrierDetection(
        signal_detected=bool(hits),
        confirmed_watermark_detected=False,
        vendor_attribution="unknown",
        hit_kinds=_unique_kinds(hits),
        hit_count=len(hits),
        hits=hits,
    )


def sanitize_unicode_carriers(
    text: str,
    policy: str = "conservative",
    *,
    normalize_spaces: bool = True,
    nfkc: bool = False,
) -> UnicodeSanitizeResult:
    """Remove or replace configured carriers. Conservative keeps valid emoji glue."""

    policy_name = _normalize_policy(policy)
    detection = detect_unicode_carriers(
        text, policy=policy_name, normalize_spaces=normalize_spaces
    )
    if not detection.hits:
        cleaned = unicodedata.normalize("NFKC" if nfkc else "NFC", text)
        return UnicodeSanitizeResult(text=cleaned, detection=detection)

    removable = {hit.index: hit for hit in detection.hits}
    chars: list[str] = []
    removed = 0
    replaced = 0
    for index, ch in enumerate(text):
        hit = removable.get(index)
        if hit is None or hit.action == "report":
            chars.append(ch)
            continue
        if hit.action == "replace_space":
            chars.append(" ")
            replaced += 1
            continue
        removed += 1
    cleaned = "".join(chars)
    form = "NFKC" if nfkc else "NFC"
    cleaned = unicodedata.normalize(form, cleaned)
    return UnicodeSanitizeResult(
        text=cleaned,
        removed_count=removed,
        replaced_count=replaced,
        detection=detection,
    )
