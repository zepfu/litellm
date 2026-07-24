"""Repository identity extraction and codex-memory workflow helpers.

Behavior-preserving Wave A2 extraction from the identity package
``__init__``. Function bodies resolve free names through the identity
host namespace after :func:`install` rebinds ``__globals__`` (record.py
contract), so most module-level imports are intentionally absent here.
"""

# ruff: noqa: F821 - free names resolve via host globals after rebind
from __future__ import annotations


def _normalize_repository_identity(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None

    cleaned = _clean_non_empty_string(value)
    if not cleaned:
        return None
    cleaned = cleaned.strip("`'\"")

    if "..." in cleaned:
        return None

    if cleaned.startswith("git@") and ":" in cleaned:
        cleaned = cleaned.split(":", 1)[1]
    elif "://" in cleaned:
        try:
            parsed = urlsplit(cleaned)
            netloc = parsed.netloc.split("@", 1)[-1]
            path = parsed.path.strip("/")
            if parsed.scheme == "file" and path:
                cleaned = path.rstrip("/").rsplit("/", 1)[-1]
            elif netloc.lower().endswith("github.com") and path:
                cleaned = path
            else:
                cleaned = urlunsplit(("", netloc, path, "", "")).strip("/")
        except Exception:
            pass
    elif cleaned.startswith("/"):
        cleaned = _normalize_repository_identity_from_absolute_path(cleaned.rstrip("/"))
        if cleaned is None:
            return None

    if cleaned.lower() in _AAWM_REPO_INSTRUCTION_FILENAMES:
        return None

    if cleaned.endswith(".git"):
        cleaned = cleaned[:-4]
    cleaned = cleaned.strip().strip("/")
    if _is_bare_file_basename_with_reject_extension(cleaned):
        return None
    if _is_bare_dot_directory(cleaned):
        return None
    if not cleaned or not _is_valid_repository_identity(cleaned) or _is_disallowed_repository_identity(cleaned):
        return None
    return cleaned


def _normalize_repository_identity_from_absolute_path(
    normalized_path: str,
) -> Optional[str]:
    codex_memory_root = _get_codex_memory_root_path()
    workspace_prefix = _aawm_workspace_root_prefix()
    if normalized_path == codex_memory_root:
        return _CODEX_MEMORY_ROOT_REPOSITORY

    path_parts = normalized_path.rsplit("/", 1)
    basename = path_parts[-1]
    if basename.lower() in _AAWM_REPO_INSTRUCTION_FILENAMES and len(path_parts) > 1:
        parent_path = path_parts[0].rstrip("/")
        if parent_path == codex_memory_root:
            return _CODEX_MEMORY_ROOT_REPOSITORY
        if parent_path.startswith(workspace_prefix):
            return parent_path.rsplit("/", 1)[-1]
        return None

    # Trusted workspace roots map to repos; nested prompt-text file paths under
    # a project are references, not session ownership.
    if normalized_path.startswith(workspace_prefix):
        sub = normalized_path[len(workspace_prefix) :].strip("/")
        if not sub:
            return None
        if "/" not in sub:
            return sub
        first, sep, rest = sub.partition("/")
        if sep and rest and rest.lower() in _AAWM_REPO_INSTRUCTION_FILENAMES:
            return first
        return None

    return basename


def _extract_repository_identity_from_text(value: str) -> Optional[str]:
    repository, _source = _extract_repository_identity_from_text_with_source(value)
    return repository


def _extract_repository_identity_from_value(
    value: Any,
    *,
    _seen: Optional[set[int]] = None,
    _depth: int = 0,
) -> Optional[str]:
    repository, _source = _extract_repository_identity_from_value_with_source(
        value,
        source_prefix="value",
        _seen=_seen,
        _depth=_depth,
    )
    return repository


def _extract_repository_identity_from_metadata_sources(
    *sources: Tuple[str, Any],
) -> Optional[str]:
    repository, _source = _extract_repository_identity_from_metadata_sources_with_source(*sources)
    return repository


def _extract_repository_identity_from_kwargs(
    kwargs: Dict[str, Any],
    *,
    metadata: Optional[Dict[str, Any]] = None,
    standard_logging_object: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    repository, _source = _extract_repository_identity_from_kwargs_with_source(
        kwargs,
        metadata=metadata,
        standard_logging_object=standard_logging_object,
    )
    return repository


def _extract_repository_identity_from_langfuse_trace_observation(
    trace: Dict[str, Any],
    observation: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    repository, _source = _extract_repository_identity_from_langfuse_trace_observation_with_source(
        trace,
        observation,
        metadata,
    )
    return repository


def _is_codex_memory_workflow_request(
    kwargs: Dict[str, Any],
    metadata: Dict[str, Any],
    *,
    request_body: Optional[Dict[str, Any]] = None,
) -> bool:
    headers = _extract_request_headers_from_kwargs(kwargs)
    if not _is_native_codex_passthrough_context(metadata, headers):
        return False

    payload = request_body
    if payload is None:
        payload = _extract_provider_cache_request_body(kwargs)
    return _payload_contains_codex_memory_workflow_markers(payload)


def _apply_codex_memory_workflow_repository(
    kwargs: Dict[str, Any],
    metadata: Dict[str, Any],
    repository: Optional[str],
    *,
    request_body: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    if not repository:
        return repository
    if not _is_codex_memory_workflow_request(
        kwargs,
        metadata,
        request_body=request_body,
    ):
        return repository

    source_repository = repository
    if source_repository.endswith(_CODEX_MEMORY_REPOSITORY_SUFFIX):
        source_repository = source_repository[: -len(_CODEX_MEMORY_REPOSITORY_SUFFIX)]

    metadata["workload_type"] = "agent_memory"
    metadata["workload_subtype"] = "codex_memory_writer"
    metadata["source_repository"] = source_repository
    metadata["repository"] = source_repository
    metadata["memory_workload_label"] = _format_memory_repository_identity(source_repository)
    _merge_tags(metadata, ["codex-memory-workflow", "agent-memory-workload"])
    return source_repository


_HOST_FUNCTION_NAMES = (
    "_normalize_repository_identity",
    "_normalize_repository_identity_from_absolute_path",
    "_extract_repository_identity_from_text",
    "_extract_repository_identity_from_value",
    "_extract_repository_identity_from_metadata_sources",
    "_extract_repository_identity_from_kwargs",
    "_extract_repository_identity_from_langfuse_trace_observation",
    "_is_codex_memory_workflow_request",
    "_apply_codex_memory_workflow_repository",
)


from types import FunctionType as _FunctionType


def _rebind_to_host_globals(fn, host_globals):
    rebound = _FunctionType(
        fn.__code__,
        host_globals,
        name=fn.__name__,
        argdefs=fn.__defaults__,
        closure=fn.__closure__,
    )
    rebound.__kwdefaults__ = fn.__kwdefaults__
    rebound.__annotations__ = getattr(fn, "__annotations__", {})
    rebound.__dict__.update(fn.__dict__)
    rebound.__module__ = __name__
    rebound.__qualname__ = fn.__qualname__
    rebound.__doc__ = fn.__doc__
    return rebound


def install(host_globals):
    """Publish this module's helpers onto the identity host namespace.

    Plain functions are rebound so their ``__globals__`` is the identity
    package dict (record.py contract) -- free-name lookups then resolve
    through the identity namespace and monkeypatches on it stay effective.
    ``functools.lru_cache`` wrappers keep this module's globals (their bodies
    only reference module-local names) and are published by reference so the
    facade-identity invariant holds.
    """
    mod = globals()
    for _name in _HOST_FUNCTION_NAMES:
        _original = mod[_name]
        if isinstance(_original, _FunctionType):
            _installed = _rebind_to_host_globals(_original, host_globals)
            mod[_name] = _installed
            host_globals[_name] = _installed
        else:
            host_globals[_name] = _original
