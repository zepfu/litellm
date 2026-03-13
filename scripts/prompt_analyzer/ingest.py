"""Local capture file loader.

Reads directories from captures_dir, each containing:
  meta.json, system.json, messages.json, tools.json
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from scripts.prompt_analyzer.models import RequestSnapshot

logger = logging.getLogger(__name__)


def _read_json(path: Path) -> Any:
    """Read a JSON file, returning None on any error."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        logger.debug("File not found: %s", path)
        return None
    except json.JSONDecodeError as exc:
        logger.warning("Malformed JSON at %s: %s", path, exc)
        return None


def _load_snapshot(directory: Path) -> Optional[RequestSnapshot]:
    """Load a single RequestSnapshot from a capture directory."""
    meta = _read_json(directory / "meta.json")
    if meta is None or not isinstance(meta, dict):
        logger.warning("Skipping %s: missing or invalid meta.json", directory.name)
        return None

    system_raw = _read_json(directory / "system.json")
    messages_raw = _read_json(directory / "messages.json")
    tools_raw = _read_json(directory / "tools.json")

    system_blocks: list[dict] = []
    if isinstance(system_raw, list):
        system_blocks = [b for b in system_raw if isinstance(b, dict)]

    messages: list[dict] = []
    if isinstance(messages_raw, list):
        messages = [m for m in messages_raw if isinstance(m, dict)]

    tools: list[dict] = []
    if isinstance(tools_raw, list):
        tools = [t for t in tools_raw if isinstance(t, dict)]

    try:
        snapshot = RequestSnapshot(
            request_id=directory.name,
            timestamp=meta.get("timestamp", ""),
            agent=meta.get("agent", "unknown"),
            model=meta.get("model", ""),
            stream=bool(meta.get("stream", False)),
            call_type=meta.get("call_type", ""),
            litellm_call_id=meta.get("litellm_call_id", ""),
            max_tokens=int(meta.get("max_tokens", 0)),
            system_blocks=system_blocks,
            messages=messages,
            tools=tools,
        )
    except Exception as exc:
        logger.warning("Failed to build snapshot for %s: %s", directory.name, exc)
        return None

    return snapshot


def load_captures(captures_dir: str) -> list[RequestSnapshot]:
    """Load all capture directories from captures_dir, sorted chronologically.

    Directories are sorted by name (timestamp-prefixed names sort naturally
    into chronological order).  Missing or malformed directories are skipped
    with a warning.
    """
    base = Path(captures_dir)
    if not base.exists():
        logger.error("Captures directory does not exist: %s", captures_dir)
        return []

    entries = sorted(
        [e for e in base.iterdir() if e.is_dir()],
        key=lambda e: e.name,
    )

    snapshots: list[RequestSnapshot] = []
    for entry in entries:
        snapshot = _load_snapshot(entry)
        if snapshot is not None:
            snapshots.append(snapshot)
        else:
            logger.warning("Skipped directory: %s", entry.name)

    logger.info("Loaded %d capture(s) from %s", len(snapshots), captures_dir)
    return snapshots
