"""Tenant-id and agent-id identity extraction for AAWM.

Behavior-preserving Wave A2 extraction from the identity package
``__init__``. Function bodies resolve free names through the identity
host namespace after :func:`install` rebinds ``__globals__`` (record.py
contract), so most module-level imports are intentionally absent here.
"""

# ruff: noqa: F821 - free names resolve via host globals after rebind
from __future__ import annotations


def _extract_claude_trace_agent_name(value: Any) -> Optional[str]:
    trace_name = _clean_non_empty_string(value)
    if not trace_name or not trace_name.startswith("claude-code."):
        return None
    agent_name = _clean_non_empty_string(trace_name.split(".", 1)[1])
    return agent_name


def _extract_claude_trace_user_identity_from_metadata_sources(
    *sources: Tuple[str, Any],
) -> Tuple[Optional[str], Optional[str]]:
    for source_name, raw_source in sources:
        source = _coerce_mapping(raw_source)
        if not source:
            continue

        trace_user_id = _normalize_repository_identity(source.get("trace_user_id"))
        if (
            trace_user_id
            and _clean_non_empty_string(source.get("trace_name"))
            and str(source.get("trace_name")).startswith("claude-code")
        ):
            return trace_user_id, f"{source_name}.trace_user_id"

        nested_source = _coerce_mapping(source.get("metadata"))
        if not nested_source:
            continue
        trace_user_id = _normalize_repository_identity(nested_source.get("trace_user_id"))
        if (
            trace_user_id
            and _clean_non_empty_string(nested_source.get("trace_name"))
            and str(nested_source.get("trace_name")).startswith("claude-code")
        ):
            return trace_user_id, f"{source_name}.metadata.trace_user_id"

    return None, None


def _extract_tenant_identity_from_kwargs(
    kwargs: Dict[str, Any],
    *,
    metadata: Optional[Dict[str, Any]] = None,
    standard_logging_object: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    litellm_params = kwargs.get("litellm_params") or {}
    standard_logging_object = standard_logging_object or kwargs.get("standard_logging_object") or {}
    passthrough_payload = kwargs.get("passthrough_logging_payload") or {}
    proxy_request = _coerce_mapping(litellm_params.get("proxy_server_request"))
    proxy_body = _coerce_mapping(proxy_request.get("body"))
    passthrough_body = _coerce_mapping(passthrough_payload.get("request_body"))

    tenant_id, source = _extract_tenant_identity_from_metadata_sources(
        ("litellm_params.metadata", metadata or litellm_params.get("metadata")),
        ("standard_logging_object.metadata", standard_logging_object.get("metadata")),
        ("kwargs.metadata", kwargs.get("metadata")),
        ("litellm_params.proxy_server_request.body", proxy_body),
        ("litellm_params.proxy_server_request.body.metadata", proxy_body.get("metadata")),
        ("passthrough_logging_payload", passthrough_payload),
        ("passthrough_logging_payload.request_body", passthrough_body),
        ("passthrough_logging_payload.request_body.metadata", passthrough_body.get("metadata")),
        ("standard_logging_object", standard_logging_object),
        ("kwargs", kwargs),
    )
    metadata_mapping = _coerce_mapping(metadata or litellm_params.get("metadata"))
    if tenant_id and _is_codex_passthrough_tenant_extraction_context(
        kwargs,
        metadata=metadata_mapping,
    ):
        trace_user_id = _normalize_repository_identity(metadata_mapping.get("trace_user_id"))
        tenant_source = _clean_non_empty_string(metadata_mapping.get("tenant_id_source"))
        if _is_codex_trace_user_tenant_source(source) or _is_codex_trace_user_tenant_source(tenant_source):
            tenant_id, source = None, None
        elif isinstance(source, str) and source.endswith(".trace_user_id"):
            tenant_id, source = None, None
        elif (
            trace_user_id
            and tenant_id == trace_user_id
            and not _is_repository_source_trusted_for_codex_tenant(metadata_mapping.get("repository_source"))
        ):
            tenant_id, source = None, None
        elif (
            isinstance(source, str)
            and any(source.endswith(marker) for marker in (".tenant_id", ".aawm_tenant_id"))
            and trace_user_id
            and tenant_id == trace_user_id
        ):
            tenant_id, source = None, None
    if tenant_id:
        return tenant_id, source

    headers = _extract_request_headers_from_kwargs(kwargs)
    tenant_id = _normalize_tenant_identity(_get_header_value(headers, *_AAWM_TENANT_ID_HEADER_NAMES))
    if tenant_id:
        return tenant_id, "request_headers"

    tenant_id, source = _extract_claude_trace_user_identity_from_metadata_sources(
        ("litellm_params.metadata", metadata or litellm_params.get("metadata")),
        ("standard_logging_object.metadata", standard_logging_object.get("metadata")),
        ("kwargs.metadata", kwargs.get("metadata")),
        ("litellm_params.proxy_server_request.body", proxy_body),
        ("litellm_params.proxy_server_request.body.metadata", proxy_body.get("metadata")),
        ("passthrough_logging_payload", passthrough_payload),
        ("passthrough_logging_payload.request_body", passthrough_body),
        ("passthrough_logging_payload.request_body.metadata", passthrough_body.get("metadata")),
        ("standard_logging_object", standard_logging_object),
        ("kwargs", kwargs),
    )
    if tenant_id:
        return tenant_id, source

    return None, None


def _extract_tenant_identity_from_langfuse_trace_observation(
    trace: Dict[str, Any],
    observation: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    trace_metadata = trace.get("metadata") if isinstance(trace, dict) else None
    tenant_id, source = _extract_tenant_identity_from_metadata_sources(
        ("observation.metadata", metadata or observation.get("metadata")),
        ("trace.metadata", trace_metadata),
        ("observation", observation),
        ("trace", trace),
    )
    if tenant_id:
        return tenant_id, source
    trace_user_id = _normalize_tenant_identity(trace.get("userId") if isinstance(trace, dict) else None)
    if trace_user_id:
        return trace_user_id, "trace.userId"
    return None, None


def _is_agent_id_like(value: str) -> bool:
    normalized = value.strip()
    if not normalized:
        return False
    return bool(
        _AAWM_AGENT_ID_UUID_RE.fullmatch(normalized)
        or _AAWM_AGENT_ID_HEX_RE.fullmatch(normalized)
        or _AAWM_AGENT_ID_PREFIXED_RE.fullmatch(normalized)
    )


def _normalize_agent_id_identity(
    value: Any,
    *,
    disallowed_values: Optional[Set[str]] = None,
) -> Optional[str]:
    cleaned = _clean_non_empty_string(value)
    if not cleaned:
        return None
    cleaned = cleaned.strip("`'\"")
    normalized = cleaned.lower()
    if normalized in {"none", "null", "unknown", "orchestrator"}:
        return None
    if disallowed_values and normalized in disallowed_values:
        return None
    if not _is_agent_id_like(cleaned):
        return None
    return cleaned


def _extract_agent_id_from_metadata_sources(
    *sources: Tuple[str, Any],
    disallowed_values: Optional[Set[str]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    for source_name, raw_source in sources:
        source = _coerce_mapping(raw_source)
        if not source:
            continue
        for key in _AAWM_AGENT_ID_METADATA_KEYS:
            agent_id = _normalize_agent_id_identity(
                source.get(key),
                disallowed_values=disallowed_values,
            )
            if agent_id:
                return agent_id, f"{source_name}.{key}"

        for nested_key in (
            "metadata",
            "litellm_metadata",
            "request_metadata",
            "user_api_key_metadata",
        ):
            nested_source = _coerce_mapping(source.get(nested_key))
            if not nested_source:
                continue
            for key in _AAWM_AGENT_ID_METADATA_KEYS:
                agent_id = _normalize_agent_id_identity(
                    nested_source.get(key),
                    disallowed_values=disallowed_values,
                )
                if agent_id:
                    return agent_id, f"{source_name}.{nested_key}.{key}"

    return None, None


def _extract_agent_id_from_kwargs(
    kwargs: Dict[str, Any],
    *,
    metadata: Optional[Dict[str, Any]] = None,
    standard_logging_object: Optional[Dict[str, Any]] = None,
    agent_name: Optional[str] = None,
    tenant_id: Optional[str] = None,
    repository: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    litellm_params = kwargs.get("litellm_params") or {}
    standard_logging_object = standard_logging_object or kwargs.get("standard_logging_object") or {}
    passthrough_payload = kwargs.get("passthrough_logging_payload") or {}
    proxy_request = _coerce_mapping(litellm_params.get("proxy_server_request"))
    proxy_body = _coerce_mapping(proxy_request.get("body"))
    passthrough_body = _coerce_mapping(passthrough_payload.get("request_body"))
    disallowed_values = _agent_id_disallowed_values_from_kwargs(
        kwargs,
        metadata=metadata,
        standard_logging_object=standard_logging_object,
        agent_name=agent_name,
        tenant_id=tenant_id,
        repository=repository,
    )

    agent_id, source = _extract_agent_id_from_metadata_sources(
        ("litellm_params.metadata", metadata or litellm_params.get("metadata")),
        ("standard_logging_object.metadata", standard_logging_object.get("metadata")),
        ("kwargs.metadata", kwargs.get("metadata")),
        ("litellm_params.proxy_server_request.body", proxy_body),
        ("litellm_params.proxy_server_request.body.metadata", proxy_body.get("metadata")),
        ("litellm_params.proxy_server_request.body.litellm_metadata", proxy_body.get("litellm_metadata")),
        ("passthrough_logging_payload", passthrough_payload),
        ("passthrough_logging_payload.request_body", passthrough_body),
        ("passthrough_logging_payload.request_body.metadata", passthrough_body.get("metadata")),
        ("passthrough_logging_payload.request_body.litellm_metadata", passthrough_body.get("litellm_metadata")),
        disallowed_values=disallowed_values,
    )
    if agent_id:
        return agent_id, source

    headers = _extract_request_headers_from_kwargs(kwargs)
    for header_name in _AAWM_AGENT_ID_HEADER_NAMES:
        agent_id = _normalize_agent_id_identity(
            _get_header_value(headers, header_name),
            disallowed_values=disallowed_values,
        )
        if agent_id:
            return agent_id, f"request_headers.{header_name}"

    return None, None


def _extract_agent_id_from_langfuse_trace_observation(
    trace: Dict[str, Any],
    observation: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    *,
    agent_name: Optional[str] = None,
    tenant_id: Optional[str] = None,
    repository: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    trace_metadata = trace.get("metadata") if isinstance(trace, dict) else None
    disallowed_values = _agent_id_disallowed_values(
        agent_name,
        tenant_id,
        repository,
        trace.get("sessionId") if isinstance(trace, dict) else None,
        trace.get("session_id") if isinstance(trace, dict) else None,
        trace.get("id") if isinstance(trace, dict) else None,
        observation.get("traceId") if isinstance(observation, dict) else None,
        observation.get("id") if isinstance(observation, dict) else None,
        metadata.get("session_id") if isinstance(metadata, dict) else None,
        metadata.get("trace_id") if isinstance(metadata, dict) else None,
        metadata.get("trace_user_id") if isinstance(metadata, dict) else None,
        metadata.get("repository") if isinstance(metadata, dict) else None,
        metadata.get("tenant_id") if isinstance(metadata, dict) else None,
        metadata.get("agent_name") if isinstance(metadata, dict) else None,
    )
    return _extract_agent_id_from_metadata_sources(
        ("observation.metadata", metadata or observation.get("metadata")),
        ("trace.metadata", trace_metadata),
        disallowed_values=disallowed_values,
    )


_HOST_FUNCTION_NAMES = (
    "_extract_claude_trace_agent_name",
    "_extract_claude_trace_user_identity_from_metadata_sources",
    "_extract_tenant_identity_from_kwargs",
    "_extract_tenant_identity_from_langfuse_trace_observation",
    "_is_agent_id_like",
    "_normalize_agent_id_identity",
    "_extract_agent_id_from_metadata_sources",
    "_extract_agent_id_from_kwargs",
    "_extract_agent_id_from_langfuse_trace_observation",
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
