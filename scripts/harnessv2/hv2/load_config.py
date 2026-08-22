"""YAML/JSON include + overlay + placeholder expand."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, MutableMapping

from hv2.errors import ConfigError

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ConfigError("PyYAML is required to load harness v2 config") from exc

_PLACEHOLDER = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def repo_root_from_harness_dir(harness_dir: Path) -> Path:
    return harness_dir.resolve().parents[1]


def default_config_path(harness_dir: Path | None = None) -> Path:
    root = harness_dir or Path(__file__).resolve().parents[1]
    return root / "config" / "harness.yaml"


def deep_merge(base: Any, overlay: Any) -> Any:
    if not isinstance(base, dict) or not isinstance(overlay, dict):
        return overlay
    out: dict[str, Any] = {**base}
    for key, value in overlay.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _read_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        loaded = yaml.safe_load(text)
    elif suffix == ".json":
        loaded = json.loads(text)
    else:
        try:
            loaded = yaml.safe_load(text)
        except Exception:
            loaded = json.loads(text)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"config root must be a mapping: {path}")
    return loaded


def expand_string(value: str, context: Mapping[str, str]) -> str:
    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        if key in context:
            return context[key]
        return match.group(0)

    return _PLACEHOLDER.sub(repl, value)


def expand_tree(value: Any, context: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        return expand_string(value, context)
    if isinstance(value, list):
        return [expand_tree(item, context) for item in value]
    if isinstance(value, dict):
        return {str(key): expand_tree(item, context) for key, item in value.items()}
    return value


def _load_with_includes(path: Path, *, seen: set[Path]) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in seen:
        raise ConfigError(f"cyclic config include: {resolved}")
    seen = set(seen)
    seen.add(resolved)
    data = _read_mapping(resolved)
    includes = data.get("include") or []
    if includes and not isinstance(includes, list):
        raise ConfigError(f"include must be a list: {resolved}")
    merged: dict[str, Any] = {}
    config_dir = resolved.parent
    for item in includes:
        rel = Path(str(item))
        child = rel if rel.is_absolute() else config_dir / rel
        merged = deep_merge(merged, _load_with_includes(child, seen=seen))
    body = {key: value for key, value in data.items() if key != "include"}
    return deep_merge(merged, body)


def load_config(
    path: Path | None = None,
    *,
    overlay: Path | None = None,
    extra_context: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    harness_dir = Path(__file__).resolve().parents[1]
    config_path = Path(path) if path is not None else default_config_path(harness_dir)
    data = _load_with_includes(config_path, seen=set())
    if overlay is not None:
        data = deep_merge(data, _load_with_includes(Path(overlay), seen=set()))
    context: dict[str, str] = {
        "home": str(Path.home()),
        "repo": str(repo_root_from_harness_dir(harness_dir)),
        "config_dir": str(config_path.parent),
    }
    if extra_context:
        context.update({str(k): str(v) for k, v in extra_context.items()})
    expanded = expand_tree(data, context)
    if not isinstance(expanded, dict):
        raise ConfigError("expanded config is not a mapping")
    expanded["_meta"] = {
        "config_path": str(config_path.resolve()),
        "config_dir": str(config_path.parent.resolve()),
        "repo_root": context["repo"],
        "harness_dir": str(harness_dir),
    }
    return expanded


def config_timeouts(config: Mapping[str, Any]) -> dict[str, int]:
    raw = config.get("timeouts") if isinstance(config.get("timeouts"), dict) else {}
    assert isinstance(raw, MutableMapping)
    return {
        "docker_seconds": int(raw.get("docker_seconds") or 30),
        "http_seconds": int(raw.get("http_seconds") or 15),
        "tui_seconds": int(raw.get("tui_seconds") or 180),
    }


def as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    raise ConfigError(f"expected list of strings, got {type(value).__name__}")
