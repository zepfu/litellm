"""YAML-driven soft-fail signatures (OpenRouter 404, MS-037)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from hv2.load_config import as_str_list


def signatures(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    block = config.get("soft_fail") if isinstance(config.get("soft_fail"), dict) else {}
    if not block:
        checks = config.get("checks") if isinstance(config.get("checks"), dict) else {}
        nested = checks.get("soft_fail")
        block = nested if isinstance(nested, dict) else {}
    rows = block.get("signatures") if isinstance(block, dict) else None
    if not isinstance(rows, list):
        return []
    return [dict(row) for row in rows if isinstance(row, dict)]


def matching_signatures(
    config: Mapping[str, Any],
    *,
    text: str,
    model: str | Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Return YAML signature dicts whose `match` substring hits `text`.

    When a signature lists `models` and a model id is given, the id must be
    in that list. `traceback_null` and `client_status` are metadata only.
    """
    haystack = text or ""
    given = [item for item in as_str_list(model) if item]
    hits: list[dict[str, Any]] = []
    for index, row in enumerate(signatures(config)):
        needle = str(row.get("match") or "")
        if not needle or needle not in haystack:
            continue
        allowed = as_str_list(row.get("models"))
        if allowed and given and not any(_model_in_list(item, allowed) for item in given):
            continue
        hits.append(
            {
                **row,
                "name": str(row.get("name") or f"soft_fail.signatures[{index}]"),
                "match": needle,
                "models": allowed,
                "traceback_null": row.get("traceback_null"),
                "client_status": row.get("client_status"),
            }
        )
    return hits


def _model_in_list(model: str, allowed: Sequence[str]) -> bool:
    if not model:
        return False
    for token in allowed:
        if not token:
            continue
        if model == token:
            return True
        if model.endswith("/" + token) or token.endswith("/" + model):
            return True
    return False
