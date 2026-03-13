"""Catalog loader and index for the marckrenn/claude-code-changelog repo.

TODO: The actual marckrenn/claude-code-changelog repo structure needs exploration
      before real catalog loading can be implemented.  The interface below is
      correct so the rest of the pipeline works with real results once the repo
      structure is understood and the fetch/parse logic is filled in.
"""
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

from scripts.prompt_analyzer.models import fingerprint as compute_fingerprint

logger = logging.getLogger(__name__)

_CATALOG_REPO_URL = "https://github.com/marckrenn/claude-code-changelog"
_DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/prompt_analyzer/catalog")


def fetch_catalog(cache_dir: str = _DEFAULT_CACHE_DIR) -> "CatalogIndex":
    """Clone or pull the marckrenn catalog into cache_dir.

    Returns an empty CatalogIndex on any network or git error so the pipeline
    continues to work without catalog data.
    """
    cache_path = Path(cache_dir)
    repo_path = cache_path / "claude-code-changelog"

    try:
        cache_path.mkdir(parents=True, exist_ok=True)
        if repo_path.exists():
            subprocess.run(
                ["git", "-C", str(repo_path), "pull", "--quiet"],
                check=True,
                capture_output=True,
                timeout=30,
            )
            logger.debug("Pulled catalog repo at %s", repo_path)
        else:
            subprocess.run(
                ["git", "clone", "--quiet", "--depth=1", _CATALOG_REPO_URL, str(repo_path)],
                check=True,
                capture_output=True,
                timeout=60,
            )
            logger.debug("Cloned catalog repo to %s", repo_path)
    except Exception as exc:
        logger.warning("Could not fetch catalog (network/git error): %s", exc)
        return CatalogIndex({})

    return CatalogIndex.from_repo(repo_path)


def _ngrams(text: str, n: int = 4) -> set[str]:
    """Return set of n-grams from text."""
    words = text.split()
    if len(words) < n:
        return {" ".join(words)}
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


class CatalogIndex:
    """Maps prompt fingerprints to catalog entry names."""

    def __init__(self, entries: dict[str, str]) -> None:
        # fingerprint -> entry_name
        self._fp_index: dict[str, str] = entries
        # normalized_text -> entry_name (for n-gram fallback)
        self._text_index: dict[str, str] = {}
        self._ngram_cache: dict[str, set[str]] = {}

    @classmethod
    def from_repo(cls, repo_path: Path) -> "CatalogIndex":
        """Load all markdown files from the repo and build the index.

        TODO: The marckrenn repo structure has not been explored yet.
              Once the directory layout is confirmed, implement proper parsing
              of the markdown changelog entries (each entry has a version, date,
              and text block describing what changed in the system prompt).
              For now this returns an empty index.
        """
        logger.debug(
            "CatalogIndex.from_repo called for %s — stub, returning empty index", repo_path
        )
        return cls({})

    def match(self, text: str) -> tuple[Optional[str], float]:
        """Return (entry_name, confidence) for the best catalog match.

        Exact fingerprint match -> 1.0
        Key-phrase overlap (4-gram Jaccard on normalized text) -> 0.5-0.9
        Returns (None, 0.0) when no match found.
        """
        fp = compute_fingerprint(text)
        if fp in self._fp_index:
            return self._fp_index[fp], 1.0

        # Fallback: 4-gram Jaccard similarity
        if not self._text_index:
            return None, 0.0

        import re as _re

        normalized = _re.sub(r"\s+", " ", text.lower()).strip()
        query_grams = _ngrams(normalized)
        best_name: Optional[str] = None
        best_score = 0.0

        for candidate_text, name in self._text_index.items():
            candidate_grams = self._ngram_cache.get(candidate_text)
            if candidate_grams is None:
                candidate_grams = _ngrams(candidate_text)
                self._ngram_cache[candidate_text] = candidate_grams
            score = _jaccard(query_grams, candidate_grams)
            if score > best_score:
                best_score = score
                best_name = name

        if best_score >= 0.5:
            # Scale 0.5-1.0 -> 0.5-0.9
            confidence = 0.5 + (best_score - 0.5) * 0.8
            return best_name, min(confidence, 0.9)

        return None, 0.0
