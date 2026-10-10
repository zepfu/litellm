"""Resolve the latest served Grok model id from the cost map."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from hv2.errors import PlanError
from hv2.load_config import repo_root_from_harness_dir

_COST_MAP = "model_prices_and_context_window.json"
_KEY = re.compile(r"^xai/grok-4\.(\d+)$")
_SENTINEL = "latest_grok"


def latest_grok_model_id(repo_root: Path | str) -> str:
    """Bare ``grok-4.<N>`` with the greatest integer minor in the cost map.

    Only ``xai/grok-4.<integer>`` keys count. Dated or suffixed ids such as
    ``grok-4.20-0309-reasoning`` are ignored. Missing or unreadable catalogs
    fail closed; there is no hardcoded fallback.
    """

    root = Path(repo_root)
    path = root / _COST_MAP
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PlanError(
            f"cannot read Grok catalog {path}: {exc.__class__.__name__}"
        ) from exc
    if not isinstance(payload, dict):
        raise PlanError(f"Grok catalog {path} is not a JSON object")
    best = _max_grok_minor(payload)
    if best is None:
        raise PlanError(
            f"Grok catalog {path} has no xai/grok-4.<integer> key"
        )
    return f"grok-4.{best}"


def latest_grok_served_ids(repo_root: Path | str) -> tuple[str, str]:
    """Native and managed catalog ids for the resolved latest Grok model."""

    model_id = latest_grok_model_id(repo_root)
    return (f"xai/{model_id}", f"oa_xai/{model_id}")


def resolve_served_concrete_ids(
    ids: list[str], repo_root: Path | str
) -> list[str]:
    """Replace the ``latest_grok`` served pair with the resolved catalog ids."""

    if _SENTINEL not in ids:
        return list(ids)
    native, managed = latest_grok_served_ids(repo_root)
    out: list[str] = []
    seen: set[str] = set()
    for item in ids:
        if item == _SENTINEL:
            replacements = (native, managed)
        else:
            replacements = (item,)
        for name in replacements:
            if name not in seen:
                seen.add(name)
                out.append(name)
    return out


def repo_root_for(config: Mapping[str, Any] | None) -> Path:
    if isinstance(config, Mapping):
        meta = config.get("_meta")
        if isinstance(meta, dict) and meta.get("repo_root"):
            return Path(str(meta["repo_root"]))
    return repo_root_from_harness_dir(Path(__file__).resolve().parents[1])


@lru_cache(maxsize=8)
def _cached_latest(repo_root: str) -> str:
    return latest_grok_model_id(repo_root)


def cached_latest_grok_model_id(repo_root: Path | str) -> str:
    return _cached_latest(str(Path(repo_root)))


def _max_grok_minor(payload: Mapping[str, Any]) -> int | None:
    best: int | None = None
    for key in payload:
        match = _KEY.fullmatch(str(key))
        if match is None:
            continue
        minor = int(match.group(1))
        if best is None or minor > best:
            best = minor
    return best
