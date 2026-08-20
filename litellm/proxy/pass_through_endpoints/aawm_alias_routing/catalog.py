"""Passthrough /models catalog: compiled YAML aliases plus served concrete ids."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from starlette.responses import JSONResponse

from .config_snapshot import get_active_snapshot


AAWM_ALIAS_OWNED_BY = "aawm_alias"

# (published catalog id, cost-map lookup key, owned_by)
SERVED_CONCRETE_MODELS: Tuple[Tuple[str, str, str], ...] = (
    ("cursor_agent/composer-2.5", "cursor_agent/composer-2.5", "cursor_agent"),
    (
        "cursor_agent/cursor-grok-4.6-high",
        "cursor_agent/cursor-grok-4.6-high",
        "cursor_agent",
    ),
    ("oa_xai/grok-4.6", "oa_xai/grok-4.6", "oa_xai"),
    ("kimi_code/k3", "kimi_code/k3", "kimi_code"),
    ("cohere/north-mini-code-1-0", "cohere/north-mini-code-1-0", "cohere"),
    ("alibaba_token_plan/glm-5.2", "alibaba_token_plan/glm-5.2", "alibaba_token_plan"),
    (
        "alibaba_token_plan/qwen3.8-max",
        "alibaba_token_plan/qwen3.8-max",
        "alibaba_token_plan",
    ),
    (
        "openrouter/qwen/qwen3.6-flash",
        "openrouter/qwen/qwen3.6-flash",
        "openrouter",
    ),
    (
        "openrouter/qwen/qwen3.5-flash-02-23",
        "openrouter/qwen/qwen3.5-flash-02-23",
        "openrouter",
    ),
    (
        "deepseek-v4-flash-free",
        "opencode/deepseek-v4-flash-free",
        "opencode_zen",
    ),
    ("big-pickle", "opencode/big-pickle", "opencode_zen"),
)

PROVENANCE_BUNDLED_MAP = "bundled_map"
PROVENANCE_AAWM_REFERENCE = "aawm_reference_pricing"
PROVENANCE_UNKNOWN = "unknown"

_NON_CHAT_MODES = frozenset({"embedding", "rerank"})

_COST_MAP_CACHE: Optional[Mapping[str, Any]] = None


def iter_compiled_alias_names(snapshot: Optional[Any] = None) -> tuple[str, ...]:
    """Return YAML alias spellings from the compiled routing snapshot.

    Names are taken from ``snapshot.aliases`` keys only. Fail closed on None.
    Do not invent ``aawm-`` prefixed ids.
    """

    if snapshot is None:
        return ()
    aliases = getattr(snapshot, "aliases", None)
    if not isinstance(aliases, Mapping):
        return ()
    names: list[str] = []
    for key in aliases.keys():
        name = str(key)
        if name and not name.startswith("aawm-"):
            names.append(name)
    return tuple(names)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _load_cost_map() -> Mapping[str, Any]:
    """Load this worktree's cost map; do not prefer host ``litellm.model_cost``."""

    global _COST_MAP_CACHE
    if _COST_MAP_CACHE is not None:
        return _COST_MAP_CACHE

    cost_map_path = _repo_root() / "model_prices_and_context_window.json"
    with cost_map_path.open(encoding="utf-8") as handle:
        parsed = json.load(handle)
    _COST_MAP_CACHE = parsed if isinstance(parsed, Mapping) else {}
    return _COST_MAP_CACHE


def _lookup_cost_row(lookup_key: str) -> Dict[str, Any]:
    row = _load_cost_map().get(lookup_key)
    return dict(row) if isinstance(row, Mapping) else {}


def _without_invoice(value: Any) -> Any:
    if isinstance(value, Mapping):
        cleaned: Dict[str, Any] = {}
        for key, nested in value.items():
            if "invoice" in str(key).lower():
                continue
            if isinstance(nested, str) and "invoice" in nested.lower():
                continue
            cleaned[str(key)] = _without_invoice(nested)
        return cleaned
    if isinstance(value, list):
        return [
            _without_invoice(item)
            for item in value
            if not (isinstance(item, str) and "invoice" in item.lower())
        ]
    return value


def _nested_aawm_reference_pricing(info: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    provider_specific = info.get("provider_specific_entry")
    if not isinstance(provider_specific, Mapping):
        return None
    for nested in provider_specific.values():
        if not isinstance(nested, Mapping):
            continue
        reference = nested.get("aawm_reference_pricing")
        if isinstance(reference, Mapping):
            cleaned = _without_invoice(reference)
            return cleaned if isinstance(cleaned, dict) else None
    return None


def _catalog_mode(raw_mode: Any) -> Optional[str]:
    if raw_mode is None:
        return None
    mode = str(raw_mode)
    if mode == "responses":
        return "chat"
    return mode


def _top_level_cost_fields(info: Mapping[str, Any]) -> Dict[str, Any]:
    fields: Dict[str, Any] = {}
    for key in (
        "input_cost_per_token",
        "output_cost_per_token",
        "max_tokens",
        "max_input_tokens",
        "max_output_tokens",
    ):
        if key in info and info[key] is not None:
            fields[key] = info[key]
    return fields


def _pricing_provenance(
    info: Mapping[str, Any],
    nested_reference: Optional[Mapping[str, Any]],
) -> str:
    if nested_reference:
        return PROVENANCE_AAWM_REFERENCE
    if "input_cost_per_token" in info or "output_cost_per_token" in info:
        return PROVENANCE_BUNDLED_MAP
    return PROVENANCE_UNKNOWN


def _concrete_row(
    published_id: str,
    lookup_key: str,
    owned_by: str,
) -> Optional[Dict[str, Any]]:
    info = _lookup_cost_row(lookup_key)
    mode = _catalog_mode(info.get("mode"))
    if mode in _NON_CHAT_MODES:
        return None

    nested_reference = _nested_aawm_reference_pricing(info)
    row: Dict[str, Any] = {
        "id": published_id,
        "object": "model",
        "owned_by": owned_by,
        "pricing_provenance": _pricing_provenance(info, nested_reference),
    }
    if mode is not None:
        row["mode"] = mode
    row.update(_top_level_cost_fields(info))
    if nested_reference:
        row["aawm_reference_pricing"] = nested_reference
    return row


def _alias_rows(snapshot: Optional[Any]) -> List[Dict[str, Any]]:
    return [
        {
            "id": name,
            "object": "model",
            "owned_by": AAWM_ALIAS_OWNED_BY,
            "mode": "chat",
            "model_info": {"aawm_alias": True, "db_model": False},
        }
        for name in iter_compiled_alias_names(snapshot)
    ]


def _overlay_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for published_id, lookup_key, owned_by in SERVED_CONCRETE_MODELS:
        row = _concrete_row(published_id, lookup_key, owned_by)
        if row is not None:
            rows.append(row)
    return rows


def build_passthrough_model_list(snapshot: Optional[Any] = None) -> dict[str, Any]:
    """OpenAI-shaped {object: list, data: [...]} of aliases plus served concrete ids."""

    if snapshot is None:
        snapshot = get_active_snapshot()

    data: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def _append(rows: Iterable[Dict[str, Any]]) -> None:
        for row in rows:
            model_id = row.get("id")
            if not model_id or str(model_id) in seen:
                continue
            seen.add(str(model_id))
            data.append(row)

    _append(_alias_rows(snapshot))
    _append(_overlay_rows())
    return {"object": "list", "data": data}


def passthrough_models_json_response(snapshot: Optional[Any] = None) -> JSONResponse:
    return JSONResponse(build_passthrough_model_list(snapshot))
