"""Publish compiled YAML alias names on Codex passthrough GET /models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

from starlette.responses import JSONResponse

from .config_snapshot import get_active_snapshot


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


def build_passthrough_model_list(snapshot: Optional[Any] = None) -> dict[str, Any]:
    """OpenAI-shaped {object: list, data: [...]} of compiled YAML alias rows."""

    if snapshot is None:
        snapshot = get_active_snapshot()
    if snapshot is None:
        return {"object": "list", "data": []}
    return {
        "object": "list",
        "data": [
            {
                "id": name,
                "object": "model",
                "owned_by": "aawm_alias",
                "mode": "chat",
                "model_info": {"aawm_alias": True, "db_model": False},
            }
            for name in iter_compiled_alias_names(snapshot)
        ],
    }


def passthrough_models_json_response(snapshot: Optional[Any] = None) -> JSONResponse:
    return JSONResponse(build_passthrough_model_list(snapshot))
