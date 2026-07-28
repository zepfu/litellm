"""Wave 7 Claude agent-spec owner extraction.

Owns agent-spec directory resolution, markdown frontmatter model
extraction, markdown reading with encoding fallback, and declared-model
loading with mtime-based caching.

Does NOT import ``llm_passthrough_endpoints`` at module scope.  All
host-global dependencies are injected through ``bind_runtime`` /
``install``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from types import FunctionType
from typing import Any, Mapping, Optional

# ---------------------------------------------------------------------------
# Default configuration (mirrors god-module constants exactly)
# ---------------------------------------------------------------------------

_CLAUDE_AGENT_SPEC_DIR_ENV_VARS: tuple[str, ...] = (
    "LITELLM_CLAUDE_AGENTS_DIR",
    "CLAUDE_AGENTS_DIR",
)
_CLAUDE_AGENT_SPEC_DEFAULT_DIRS: tuple[str, ...] = (
    "~/.claude/agents",
    "~/.claude/agents",
)

# Module-local cache; rebound to host cache via bind_runtime for
# monkeypatch-safe shared state.
_claude_agent_model_cache: dict[Path, tuple[Optional[int], Optional[str]]] = {}

_RUNTIME_DEPENDENCY_NAMES = (
    "_CLAUDE_AGENT_SPEC_DIR_ENV_VARS",
    "_CLAUDE_AGENT_SPEC_DEFAULT_DIRS",
    "_claude_agent_model_cache",
)

_HOST_FUNCTION_NAMES = (
    "_get_claude_agent_spec_dir",
    "_extract_model_from_markdown_frontmatter",
    "_read_claude_agent_markdown",
    "_load_claude_agent_declared_model",
)


# ---------------------------------------------------------------------------
# Runtime binding / host installation
# ---------------------------------------------------------------------------


def bind_runtime(namespace: Mapping[str, object]) -> None:
    """Bind explicit host callbacks/configuration for direct module use."""
    module_globals = globals()
    for name in _RUNTIME_DEPENDENCY_NAMES:
        if name in namespace:
            module_globals[name] = namespace[name]


def install(host_globals: dict[str, Any]) -> None:
    """Publish rebound facades into *host_globals* only.

    Contract:

    * Seeds *host_globals* with any names missing from the owner module so
      facades are functional even when called with an incomplete namespace.
    * Publishes rebound function objects (whose ``__globals__`` is
      *host_globals*) into *host_globals* for each name in
      ``_HOST_FUNCTION_NAMES``, giving the host live monkeypatch reachability.
    * Does **not** modify the owner module's own globals; direct owner-module
      usability (``spec._load_claude_agent_declared_model(...)``) is preserved
      regardless of how many times ``install`` is called or what state
      *host_globals* is in.
    """
    module_globals = globals()
    # Seed host_globals so facades can resolve all module-level names.
    for name, value in module_globals.items():
        if name not in host_globals:
            host_globals[name] = value

    for name in _HOST_FUNCTION_NAMES:
        function = module_globals[name]
        rebound = FunctionType(
            function.__code__,
            host_globals,
            function.__name__,
            function.__defaults__,
            function.__closure__,
        )
        rebound.__kwdefaults__ = function.__kwdefaults__
        rebound.__annotations__ = function.__annotations__
        rebound.__doc__ = function.__doc__
        rebound.__module__ = function.__module__
        rebound.__qualname__ = function.__qualname__
        if function.__dict__:
            rebound.__dict__.update(function.__dict__)
        host_globals[name] = rebound


# ---------------------------------------------------------------------------
# Owner implementations
# ---------------------------------------------------------------------------


def _get_claude_agent_spec_dir() -> Optional[Path]:
    for env_var in _CLAUDE_AGENT_SPEC_DIR_ENV_VARS:
        value = os.getenv(env_var)
        if not isinstance(value, str) or not value.strip():
            continue
        candidate = Path(value).expanduser()
        if candidate.is_dir():
            return candidate

    for raw_path in _CLAUDE_AGENT_SPEC_DEFAULT_DIRS:
        candidate = Path(raw_path).expanduser()
        if candidate.is_dir():
            return candidate

    return None


def _extract_model_from_markdown_frontmatter(markdown_text: str) -> Optional[str]:
    if not markdown_text.startswith("---\n"):
        return None

    closing_index = markdown_text.find("\n---", 4)
    if closing_index == -1:
        return None

    frontmatter = markdown_text[4:closing_index]
    match = re.search(r"(?m)^model:\s*(?P<model>.+?)\s*$", frontmatter)
    if match is None:
        return None

    model_value = match.group("model").strip().strip('"').strip("'")
    return model_value or None


def _read_claude_agent_markdown(candidate_path: Path) -> Optional[str]:
    try:
        markdown_bytes = candidate_path.read_bytes()
    except OSError:
        return None

    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            return markdown_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue

    return markdown_bytes.decode("utf-8", errors="replace")


def _load_claude_agent_declared_model(agent_name: str) -> Optional[str]:
    normalized_agent_name = agent_name.strip()
    if not normalized_agent_name:
        return None

    if normalized_agent_name != Path(normalized_agent_name).name:
        return None

    agents_dir = _get_claude_agent_spec_dir()
    if agents_dir is None:
        return None

    candidate_path = agents_dir / f"{normalized_agent_name}.md"
    if not candidate_path.is_file():
        return None

    try:
        stat_result = candidate_path.stat()
    except OSError:
        return None

    cache_entry = _claude_agent_model_cache.get(candidate_path)
    cache_key = getattr(stat_result, "st_mtime_ns", None)
    if cache_entry is not None and cache_entry[0] == cache_key:
        return cache_entry[1]

    markdown_text = _read_claude_agent_markdown(candidate_path)
    if markdown_text is None:
        return None

    model_name = _extract_model_from_markdown_frontmatter(markdown_text)
    _claude_agent_model_cache[candidate_path] = (cache_key, model_name)
    return model_name
