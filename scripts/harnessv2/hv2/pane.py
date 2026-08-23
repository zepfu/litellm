"""Pane needle matching. Needles that only appear in the sent prompt are ignored."""

from __future__ import annotations

_EXACT_PONG = "PONG"


def _latest_prompt_echo_index(pane: str, prompt: str) -> int:
    prompt_line = prompt.strip()
    last = -1
    if not prompt_line:
        return last
    for index, raw_line in enumerate(pane.splitlines()):
        if raw_line.strip() == prompt_line:
            last = index
    return last


def _standalone_line_after_latest_prompt(
    pane: str, prompt: str, token: str, *, after_echo_index: int | None = None
) -> bool:
    """True when a standalone *token* line appears after the latest prompt echo."""
    prompt_line = prompt.strip()
    if not token or not prompt_line or token == prompt_line:
        return False
    latest_echo = _latest_prompt_echo_index(pane, prompt)
    if latest_echo < 0:
        return False
    # Watermark: only count a PONG after an echo that is newer than pre-send.
    if after_echo_index is not None and latest_echo <= after_echo_index:
        return False
    for raw_line in pane.splitlines()[latest_echo + 1 :]:
        if raw_line.strip() == token:
            return True
    return False


def _pane_exact_pong(
    pane: str, prompt: str, *, after_echo_index: int | None = None
) -> bool:
    """True when a standalone PONG line follows the latest prompt echo."""
    return _standalone_line_after_latest_prompt(
        pane, prompt, _EXACT_PONG, after_echo_index=after_echo_index
    )


def _pane_has_any(
    pane: str,
    needles: list[str],
    *,
    prompt: str | None = None,
    after_echo_index: int | None = None,
) -> bool:
    """True if a needle is in the pane and is not just the sent prompt."""
    sent = prompt or ""
    for token in needles:
        if not token:
            continue
        if token not in pane:
            continue
        # Leftover session-dir PONG before this turn's echo is not a live reply.
        if token == _EXACT_PONG and prompt is not None:
            if _standalone_line_after_latest_prompt(
                pane, prompt, token, after_echo_index=after_echo_index
            ):
                return True
            continue
        if token not in sent:
            return True
        # Needle is inside the sent prompt (H-6). Still accept a standalone
        # exact pane line equal to the needle that is not the whole prompt.
        for raw_line in pane.splitlines():
            line = raw_line.strip()
            if line == token and line != sent.strip():
                return True
    return False
