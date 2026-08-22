"""Harness-owned user id and identity headers. Names live in YAML."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from hv2.errors import HarnessError
from hv2.load_config import as_str_list, expand_string, repo_root_from_harness_dir


def resolve_harness_user_id(
    config: Mapping[str, Any],
    *,
    environ: Mapping[str, str] | None = None,
) -> str:
    source = os.environ if environ is None else environ
    for key in as_str_list(config.get("harness_user_id_env")):
        raw = source.get(key)
        if raw is None:
            continue
        value = str(raw).strip()
        if value:
            return value
    default = str(config.get("default_harness_user_id") or "").strip()
    if not default:
        raise HarnessError(
            "resolved harness user id is empty; set a value in one of "
            "harness_user_id_env or a non-empty default_harness_user_id"
        )
    return default


harness_user_id = resolve_harness_user_id


def _expand_context(
    config: Mapping[str, Any],
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    meta = config.get("_meta") if isinstance(config.get("_meta"), Mapping) else {}
    harness_dir = Path(__file__).resolve().parents[1]
    repo = str(meta.get("repo_root") or repo_root_from_harness_dir(harness_dir))
    return {
        "home": str(Path.home()),
        "repo": repo,
        "harness_user_id": resolve_harness_user_id(config, environ=environ),
    }


def identity_headers(
    config: Mapping[str, Any],
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    checks = config.get("checks") if isinstance(config.get("checks"), dict) else {}
    raw = checks.get("identity_headers")
    if not isinstance(raw, Mapping):
        return {}
    context = _expand_context(config, environ=environ)
    headers: dict[str, str] = {}
    for key, value in raw.items():
        name = expand_string(str(key), context).strip()
        text = expand_string("" if value is None else str(value), context)
        if name and text:
            headers[name] = text
    return headers


def merge_request_headers(
    config: Mapping[str, Any],
    caller: Mapping[str, str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Identity YAML first; non-empty caller values win on name conflict."""
    caller_nonempty = {
        str(key): str(value)
        for key, value in (caller or {}).items()
        if str(key) and value is not None and str(value)
    }
    return {**identity_headers(config, environ=environ), **caller_nonempty}
