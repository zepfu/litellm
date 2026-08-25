"""Pane needle matching. Needles that only appear in the sent prompt are ignored."""

from __future__ import annotations

import re

_EXACT_PONG = "PONG"
# Codex TUI prefixes assistant lines with a list bullet. Standalone pass
# tokens such as `hv2-codex-child` still count when the line is `• token`.
_LEADING_LIST_MARKER = re.compile(r"^(?:[•●▪▸›*]|\d+[.)]|-)\s+")
_TREE_PREFIX = re.compile(r"^[└├│]\s*")
_DATE_STDOUT_LINE = re.compile(
    r"^(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun) "
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) "
    r"\d{1,2} \d{2}:\d{2}:\d{2}(?: [A-Z]{2,5})? \d{4}$"
)
_ISO_DATE_STDOUT_LINE = re.compile(
    r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}"
)
_PWD_STDOUT_LINE = re.compile(r"^/\S+$")
_CHILD_STDOUT_REJECT = re.compile(r"(?:hv2_(?:codex_)?child|…|\.\.\.)")
_RAN_DATE_LINE = re.compile(r"^Ran date(?:\s|$)")
_RAN_PWD_LINE = re.compile(r"^Ran pwd(?:\s|$)")
_CHILD_SPAWN_LINE = re.compile(r"^(?:Spawned|Started)\b")
_CODEX_IDLE_COMPOSER_LINE = re.compile(
    r"^(?:[›>]\s*Ask Codex to do anything|>)\s*$"
)
_CODEX_TOOL_PASS_TOKEN = "hv2-codex-child"


def _latest_prompt_echo_index(pane: str, prompt: str) -> int:
    """Return the last line index of the latest prompt echo, including wraps.

    Codex may paint the sent prompt across visually wrapped lines. Current-turn
    scanning starts after that wrapped echo so a missing exact one-line match
    never falls back to line 0.
    """

    prompt_line = prompt.strip()
    sent = " ".join(prompt_line.split())
    last = -1
    if not sent:
        return last
    lines = pane.splitlines()
    for start, raw_start in enumerate(lines):
        raw_stripped = raw_start.strip()
        start_text = " ".join(_line_token_text(raw_start).split())
        if raw_stripped == prompt_line or start_text == sent:
            last = start
            continue
        if not start_text or not sent.startswith(start_text):
            continue
        acc = start_text
        for end, raw_line in enumerate(lines[start + 1 :], start=start + 1):
            piece = " ".join(_line_token_text(raw_line).split())
            if not piece:
                break
            joined = f"{acc} {piece}"
            if joined == sent:
                last = end
                break
            if not sent.startswith(joined):
                break
            acc = joined
    return last


def _line_token_text(raw_line: str) -> str:
    """Return *raw_line* without leading TUI list markers such as ``• ``."""

    line = raw_line.strip()
    while True:
        stripped = _LEADING_LIST_MARKER.sub("", line, count=1).strip()
        if stripped == line:
            return line
        line = stripped


def _line_equals_token(raw_line: str, token: str) -> bool:
    return bool(token) and _line_token_text(raw_line) == token


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
        if _line_equals_token(raw_line, token):
            return True
    return False


def _pane_exact_pong(
    pane: str, prompt: str, *, after_echo_index: int | None = None
) -> bool:
    """True when a standalone PONG line follows the latest prompt echo."""
    return _standalone_line_after_latest_prompt(
        pane, prompt, _EXACT_PONG, after_echo_index=after_echo_index
    )


def _pane_line_command_text(raw_line: str) -> str:
    """Strip TUI list markers and Codex tree prefixes from *raw_line*."""

    text = _line_token_text(raw_line)
    while True:
        stripped = _TREE_PREFIX.sub("", text, count=1).strip()
        if stripped == text:
            return text
        text = stripped


def _pane_line_has_tree_prefix(raw_line: str) -> bool:
    """True when *raw_line* is a Codex tree child such as ``└ /tmp/...``."""

    return bool(_TREE_PREFIX.match(_line_token_text(raw_line)))


def _pane_scan_start(
    pane: str,
    prompt: str | None,
    *,
    after_echo_index: int | None = None,
) -> int:
    """Return the first line index that belongs to the current turn."""

    sent = (prompt or "").strip()
    latest_echo = _latest_prompt_echo_index(pane, sent) if sent else -1
    if after_echo_index is not None:
        # A supplied watermark, including -1, is current-turn only after a
        # newer prompt echo. Missing or unchanged echoes fail closed.
        if latest_echo >= 0 and latest_echo > after_echo_index:
            return latest_echo + 1
        return len(pane.splitlines())
    if latest_echo >= 0:
        return latest_echo + 1
    return 0


def _pane_is_rejected_child_text(text: str) -> bool:
    if not text:
        return True
    lowered = text.lower()
    if "directory:" in lowered or "·" in text:
        return True
    return bool(_CHILD_STDOUT_REJECT.search(text))


def _pane_is_child_spawn_line(text: str) -> bool:
    """True for a current-turn ``Spawned`` / ``Started`` child, not leftover chrome."""

    if _pane_is_rejected_child_text(text):
        return False
    if not _CHILD_SPAWN_LINE.match(text):
        return False
    return "hv2_child" not in text


def _pane_has_current_turn_child_spawn(
    pane: str,
    prompt: str | None = None,
    *,
    after_echo_index: int | None = None,
) -> bool:
    """True when a non-chrome child spawn appears after the latest prompt echo."""

    sent = (prompt or "").strip()
    start = _pane_scan_start(pane, prompt, after_echo_index=after_echo_index)
    for raw_line in pane.splitlines()[start:]:
        text = _pane_line_command_text(raw_line)
        if not text or text == sent:
            continue
        if _pane_is_child_spawn_line(text):
            return True
    return False


def _pane_has_child_command_stdout(
    pane: str,
    prompt: str | None = None,
    *,
    after_echo_index: int | None = None,
) -> bool:
    """True when current-turn child ``date`` / ``pwd`` stdout is in the pane.

    Workspace chrome, leftover ``/root/hv2_child_*`` or ``/root/hv2_codex_child*`` lines, and truncated
    directory headers are not command stdout. ``Ran date`` / ``Ran pwd`` context
    still counts, and a standalone current-turn ``/tmp/...`` pwd line that is not
    tree-attached spawn chrome also counts. A standalone ``hv2-codex-child``
    token without this evidence is not a tool-bearing pass.
    """

    sent = (prompt or "").strip()
    start = _pane_scan_start(pane, prompt, after_echo_index=after_echo_index)
    ran_context: str | None = None
    for raw_line in pane.splitlines()[start:]:
        text = _pane_line_command_text(raw_line)
        if not text or text == sent:
            ran_context = None
            continue
        if _RAN_DATE_LINE.match(text):
            ran_context = "date"
            continue
        if _RAN_PWD_LINE.match(text):
            ran_context = "pwd"
            continue
        if _pane_is_child_spawn_line(text) or _line_equals_token(
            raw_line, _CODEX_TOOL_PASS_TOKEN
        ):
            ran_context = None
            continue
        if _pane_is_rejected_child_text(text):
            continue
        if ran_context is None:
            # Tree-attached paths without Ran date/pwd are spawn chrome.
            if _pane_line_has_tree_prefix(raw_line):
                continue
            if _DATE_STDOUT_LINE.match(text) or _ISO_DATE_STDOUT_LINE.match(text):
                return True
            if _PWD_STDOUT_LINE.match(text):
                return True
            continue
        if ran_context == "date" and (
            _DATE_STDOUT_LINE.match(text) or _ISO_DATE_STDOUT_LINE.match(text)
        ):
            return True
        if ran_context == "pwd" and _PWD_STDOUT_LINE.match(text):
            return True
    return False


def _pane_has_current_turn_pass_token(
    pane: str,
    prompt: str | None,
    token: str,
    *,
    after_echo_index: int | None = None,
) -> bool:
    """True when a standalone pass token appears on the current turn."""

    if not token:
        return False
    sent = (prompt or "").strip()
    start = _pane_scan_start(pane, prompt, after_echo_index=after_echo_index)
    for raw_line in pane.splitlines()[start:]:
        if _line_equals_token(raw_line, token) and token != sent:
            return True
    return False


def _pane_has_codex_idle_prompt(
    pane: str,
    prompt: str | None = None,
    *,
    after_echo_index: int | None = None,
) -> bool:
    """True when the current Codex turn has returned to an idle prompt line."""

    start = _pane_scan_start(pane, prompt, after_echo_index=after_echo_index)
    for raw_line in pane.splitlines()[start:]:
        if _CODEX_IDLE_COMPOSER_LINE.match(raw_line.strip()):
            return True
    return False


def _pane_tool_command_pass(
    pane: str,
    prompt: str,
    tokens: list[str],
    *,
    after_echo_index: int | None = None,
) -> bool:
    """True when the pass token and current-turn child stdout are present.

    Local ``/root/hv2_child*`` chrome and tree-attached workspace paths without
    ``Ran date`` / ``Ran pwd`` are not stdout. A current-turn spawn line is not
    required when standalone child stdout is already visible.
    """

    if not tokens:
        return False
    if _CODEX_TOOL_PASS_TOKEN in tokens:
        if not _pane_has_current_turn_pass_token(
            pane,
            prompt,
            _CODEX_TOOL_PASS_TOKEN,
            after_echo_index=after_echo_index,
        ):
            return False
        if not _pane_has_child_command_stdout(
            pane, prompt, after_echo_index=after_echo_index
        ):
            return False
        return True
    return _pane_has_any(
        pane, tokens, prompt=prompt, after_echo_index=after_echo_index
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
    lines = pane.splitlines()
    if after_echo_index is not None:
        start = _pane_scan_start(
            pane, prompt, after_echo_index=after_echo_index
        )
        scan_lines = lines[start:]
        scan_text = "\n".join(scan_lines)
    else:
        scan_lines = lines
        scan_text = pane
    for token in needles:
        if not token:
            continue
        # Leftover session-dir PONG before this turn's echo is not a live reply.
        if token == _EXACT_PONG and prompt is not None:
            if _standalone_line_after_latest_prompt(
                pane, prompt, token, after_echo_index=after_echo_index
            ):
                return True
            continue
        if token not in scan_text:
            continue
        # A supplied watermark applies to every needle, including 404 text.
        # Stale matches before the current-turn scan start cannot pass.
        if token not in sent:
            return True
        # Needle is inside the sent prompt (H-6). Still accept a standalone
        # pane line equal to the needle (Codex may prefix `• `) that is not
        # the whole prompt and not a wrapped prompt sentence containing it.
        for raw_line in scan_lines:
            if _line_equals_token(raw_line, token) and token != sent.strip():
                return True
    return False
