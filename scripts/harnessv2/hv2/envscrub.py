"""Scrubbed child environment. Allow/deny lists live in YAML."""

from __future__ import annotations

import os
from typing import Any, Mapping

from hv2.identity import harness_user_id
from hv2.load_config import as_str_list


def _child_env_block(config: Mapping[str, Any]) -> dict[str, Any]:
    block = config.get("child_env") if isinstance(config.get("child_env"), dict) else {}
    checks = config.get("checks") if isinstance(config.get("checks"), dict) else {}
    nested = checks.get("child_env") if isinstance(checks.get("child_env"), dict) else {}
    # checks.yaml nests child_env under checks; tolerate a top-level copy.
    return nested or block or {}


def _is_denied(key: str, spec: Mapping[str, Any]) -> bool:
    deny_keys = set(as_str_list(spec.get("deny_keys")))
    if key in deny_keys:
        return True
    for prefix in as_str_list(spec.get("deny_prefixes")):
        if key.startswith(prefix):
            return True
    if key.startswith("LITELLM_"):
        allow_keys = set(as_str_list(spec.get("allow_keys")))
        if key not in allow_keys:
            return True
    upper = key.upper()
    for fragment in as_str_list(spec.get("deny_substrings")):
        if fragment and fragment in upper:
            return True
    return False


def _is_allowed(key: str, spec: Mapping[str, Any]) -> bool:
    if _is_denied(key, spec):
        return False
    if key in set(as_str_list(spec.get("base_keys"))):
        return True
    if key in set(as_str_list(spec.get("allow_keys"))):
        return True
    for prefix in as_str_list(spec.get("allow_prefixes")):
        if key.startswith(prefix):
            return True
    return False


def _inject_harness_user_id(
    env: dict[str, str],
    config: Mapping[str, Any],
    spec: Mapping[str, Any],
    *,
    environ: Mapping[str, str],
) -> None:
    keys = as_str_list(config.get("harness_user_id_env"))
    if not keys:
        return
    key = keys[0]
    if str(env.get(key) or "").strip():
        return
    if _is_denied(key, spec):
        return
    env[key] = harness_user_id(config, environ=environ)


def scrubbed_child_env(
    config: Mapping[str, Any],
    extra_env: Mapping[str, str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    spec = _child_env_block(config)
    source = os.environ if environ is None else environ
    env: dict[str, str] = {}
    for key, value in source.items():
        if value is None:
            continue
        if _is_allowed(key, spec):
            env[key] = value
    if extra_env:
        for key, value in extra_env.items():
            if value is None:
                continue
            key_str = str(key)
            if _is_denied(key_str, spec):
                continue
            env[key_str] = str(value)
    _inject_harness_user_id(env, config, spec, environ=source)
    return env
