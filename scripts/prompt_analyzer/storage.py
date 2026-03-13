"""Pluggable storage backend for the replacement table.

The abstract base makes it straightforward to add a PostgreSQL backend later.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from scripts.prompt_analyzer.models import FragmentType, ReplacementEntry

logger = logging.getLogger(__name__)


def _entry_to_dict(entry: ReplacementEntry) -> dict[str, Any]:
    return {
        "fingerprint": entry.fingerprint,
        "fragment_type": entry.fragment_type.value,
        "catalog_match": entry.catalog_match,
        "original_preview": entry.original_preview,
        "replacement_text": entry.replacement_text,
        "enabled": entry.enabled,
        "observation_count": entry.observation_count,
        "first_seen": entry.first_seen,
        "last_seen": entry.last_seen,
    }


def _dict_to_entry(d: dict[str, Any]) -> ReplacementEntry:
    return ReplacementEntry(
        fingerprint=d["fingerprint"],
        fragment_type=FragmentType(d.get("fragment_type", FragmentType.UNKNOWN.value)),
        catalog_match=d.get("catalog_match"),
        original_preview=d.get("original_preview", ""),
        replacement_text=d.get("replacement_text"),
        enabled=bool(d.get("enabled", False)),
        observation_count=int(d.get("observation_count", 0)),
        first_seen=d.get("first_seen", ""),
        last_seen=d.get("last_seen", ""),
    )


class StorageBackend:
    """Abstract base for replacement table storage."""

    def load_entries(self) -> dict[str, ReplacementEntry]:
        """Return all entries keyed by fingerprint."""
        raise NotImplementedError

    def save_entry(self, entry: ReplacementEntry) -> None:
        """Upsert a single entry."""
        raise NotImplementedError

    def save_all(self, entries: dict[str, ReplacementEntry]) -> None:
        """Replace all entries."""
        raise NotImplementedError


class JsonFileStorage(StorageBackend):
    """JSON file backend — list of serialized ReplacementEntry objects."""

    def __init__(self, path: str) -> None:
        self._path = Path(path)

    def load_entries(self) -> dict[str, ReplacementEntry]:
        if not self._path.exists():
            return {}
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                raw: list[dict] = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load replacement table from %s: %s", self._path, exc)
            return {}
        entries: dict[str, ReplacementEntry] = {}
        for item in raw:
            try:
                entry = _dict_to_entry(item)
                entries[entry.fingerprint] = entry
            except Exception as exc:
                logger.warning("Skipping malformed entry: %s", exc)
        return entries

    def save_entry(self, entry: ReplacementEntry) -> None:
        entries = self.load_entries()
        entries[entry.fingerprint] = entry
        self.save_all(entries)

    def save_all(self, entries: dict[str, ReplacementEntry]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = [_entry_to_dict(e) for e in entries.values()]
        try:
            with open(self._path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
            logger.debug("Saved %d entries to %s", len(entries), self._path)
        except OSError as exc:
            logger.error("Failed to save replacement table: %s", exc)
