"""\"You are '<agent>'\" agent-name paths and codex/grok context predicates.

Behavior-preserving Wave A2 extraction from the identity package
``__init__``. Function bodies resolve free names through the identity
host namespace after :func:`install` rebinds ``__globals__`` (record.py
contract), so most module-level imports are intentionally absent here.
"""

# ruff: noqa: F821 - free names resolve via host globals after rebind
from __future__ import annotations


def _extract_agent_name(kwargs: Dict[str, Any]) -> str:
    agent_name, _tenant_id = _extract_agent_context(kwargs)
    return agent_name or _DEFAULT_AGENT


def _is_native_codex_passthrough_context(metadata: Dict[str, Any], headers: Dict[str, Any]) -> bool:
    route_family = _clean_non_empty_string(metadata.get("passthrough_route_family"))
    if route_family and route_family.lower() == "codex_responses":
        return True

    trace_name = _first_non_empty_string(
        metadata.get("trace_name"),
        _get_header_value(headers, "langfuse_trace_name"),
    )
    user_agent = _get_header_value(headers, "user-agent")
    return bool(trace_name and trace_name.lower() == "codex" and user_agent and "codex" in user_agent.lower())


def _is_codex_client_identity(metadata: Dict[str, Any], headers: Dict[str, Any]) -> bool:
    user_agent = _first_non_empty_string(
        metadata.get("client_user_agent"),
        metadata.get("user_agent"),
        metadata.get("http_user_agent"),
        _get_header_value(headers, "user-agent", "User-Agent"),
    )
    parsed_client_name, _parsed_client_version = _parse_client_identity_from_user_agent(user_agent)
    client_name = _first_non_empty_string(metadata.get("client_name"), parsed_client_name)
    return bool((client_name and "codex" in client_name.lower()) or (user_agent and "codex" in user_agent.lower()))


def _is_codex_default_agent_context(
    kwargs: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    metadata = metadata or _ensure_mutable_metadata(kwargs)
    headers = _extract_request_headers_from_kwargs(kwargs)
    return bool(
        _is_native_codex_passthrough_context(metadata, headers)
        and _is_codex_client_identity(metadata, headers)
        and not _is_codex_subagent_context(kwargs, metadata)
    )


def _is_codex_subagent_context(
    kwargs: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    litellm_params = kwargs.get("litellm_params") or {}
    metadata = metadata or litellm_params.get("metadata") or {}
    standard_logging_object = kwargs.get("standard_logging_object") or {}
    passthrough_payload = kwargs.get("passthrough_logging_payload") or {}
    proxy_request = _coerce_mapping(litellm_params.get("proxy_server_request"))
    proxy_body = _coerce_mapping(proxy_request.get("body"))
    passthrough_body = _coerce_mapping(passthrough_payload.get("request_body"))
    sources = (
        metadata,
        standard_logging_object.get("metadata"),
        proxy_body,
        proxy_body.get("metadata"),
        proxy_body.get("litellm_metadata"),
        passthrough_payload,
        passthrough_body,
        passthrough_body.get("metadata"),
        passthrough_body.get("litellm_metadata"),
    )
    for raw_source in sources:
        source = _coerce_mapping(raw_source)
        if not source:
            continue
        thread_source = _clean_non_empty_string(source.get("thread_source"))
        if thread_source and thread_source.lower() == "subagent":
            return True
        nested_source = _coerce_mapping(source.get("source"))
        nested_thread_source = _clean_non_empty_string(nested_source.get("thread_source"))
        if nested_thread_source and nested_thread_source.lower() == "subagent":
            return True
        if nested_source.get("subagent"):
            return True
    return False


def _is_native_grok_passthrough_context(metadata: Dict[str, Any], headers: Dict[str, Any]) -> bool:
    route_family = str(metadata.get("passthrough_route_family") or "").lower()
    if "grok" in route_family or "xai" in route_family:
        return True

    client_name = str(metadata.get("client_name") or "").lower()
    if client_name == "grok-build":
        return True

    trace_name = _first_non_empty_string(
        metadata.get("trace_name"),
        _get_header_value(headers, "langfuse_trace_name"),
    )
    if trace_name and str(trace_name).lower().startswith("grok-build"):
        return True

    return any(
        str(header_name).lower().startswith("x-grok-") or str(header_name).lower() == "x-xai-token-auth"
        for header_name in headers
    )


def _promote_grok_repository_trace_identity(
    kwargs: Dict[str, Any],
    metadata: Dict[str, Any],
    headers: Dict[str, Any],
) -> None:
    if not _is_native_grok_passthrough_context(metadata, headers):
        return

    repository = _extract_repository_identity_from_kwargs(
        kwargs,
        metadata=metadata,
    )
    if repository:
        metadata["repository"] = repository

    tenant_id, tenant_source = _extract_tenant_identity_from_kwargs(
        kwargs,
        metadata=metadata,
    )
    if not tenant_id:
        _agent_name, agent_context_tenant_id = _extract_agent_context(kwargs)
        if agent_context_tenant_id:
            tenant_id = agent_context_tenant_id
            tenant_source = "agent_context_text"
    if tenant_id and not metadata.get("tenant_id"):
        metadata["tenant_id"] = tenant_id
    if tenant_id and tenant_source and not metadata.get("tenant_id_source"):
        metadata["tenant_id_source"] = tenant_source

    metadata_trace_user_id = _clean_non_empty_string(metadata.get("trace_user_id"))
    header_trace_user_id = _get_header_value(headers, "langfuse_trace_user_id")
    desired_trace_user_id = repository or tenant_id
    if not desired_trace_user_id:
        return

    if metadata_trace_user_id is None or _is_generic_grok_trace_user_id(metadata_trace_user_id):
        metadata["trace_user_id"] = desired_trace_user_id
    if header_trace_user_id is None or _is_generic_grok_trace_user_id(header_trace_user_id):
        headers["langfuse_trace_user_id"] = desired_trace_user_id


def _promote_codex_repository_trace_user_id(
    kwargs: Dict[str, Any],
    metadata: Dict[str, Any],
    headers: Dict[str, Any],
) -> None:
    if not _is_native_codex_passthrough_context(metadata, headers):
        return

    if _is_numeric_identity_placeholder(metadata.get("repository")):
        metadata.pop("repository", None)
    if _is_numeric_identity_placeholder(metadata.get("tenant_id")):
        metadata.pop("tenant_id", None)
        metadata.pop("tenant_id_source", None)
    if _is_numeric_identity_placeholder(metadata.get("trace_user_id")):
        metadata.pop("trace_user_id", None)
    if _is_numeric_identity_placeholder(_get_header_value(headers, "langfuse_trace_user_id")):
        headers.pop("langfuse_trace_user_id", None)

    metadata_trace_user_id = _normalize_repository_identity(metadata.get("trace_user_id"))
    header_trace_user_id = _get_header_value(headers, "langfuse_trace_user_id")
    repository, repository_source = _extract_repository_identity_from_kwargs_with_source(
        kwargs,
        metadata=metadata,
    )
    repository_before_memory_workflow = repository
    repository = _apply_codex_memory_workflow_repository(
        kwargs,
        metadata,
        repository,
    )
    if repository and repository_source:
        if repository != repository_before_memory_workflow:
            repository_source = f"{repository_source}.codex_memory_workflow"
        metadata["repository_source"] = repository_source

    desired_trace_user_id: Optional[str] = None
    if metadata_trace_user_id and not _is_generic_codex_trace_user_id(metadata_trace_user_id):
        desired_trace_user_id = metadata_trace_user_id
    elif (
        repository
        and _is_repository_source_trusted_for_tenant(repository_source)
        and (metadata_trace_user_id is None or _is_generic_codex_trace_user_id(metadata_trace_user_id))
        and (header_trace_user_id is None or _is_generic_codex_trace_user_id(header_trace_user_id))
    ):
        desired_trace_user_id = repository

    if not desired_trace_user_id:
        return

    metadata["trace_user_id"] = desired_trace_user_id
    if header_trace_user_id is None or _is_generic_codex_trace_user_id(header_trace_user_id):
        headers["langfuse_trace_user_id"] = desired_trace_user_id


_HOST_FUNCTION_NAMES = (
    "_extract_agent_name",
    "_is_native_codex_passthrough_context",
    "_is_codex_client_identity",
    "_is_codex_default_agent_context",
    "_is_codex_subagent_context",
    "_is_native_grok_passthrough_context",
    "_promote_grok_repository_trace_identity",
    "_promote_codex_repository_trace_user_id",
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
