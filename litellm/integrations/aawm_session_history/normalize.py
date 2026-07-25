"""Record normalization chain: trust/sync/quality/latency/zero-token.

Behavior-preserving Wave A4C extraction from the identity package
``__init__``. Function bodies resolve free names through the identity
host namespace after :func:`install` rebinds ``__globals__`` (record.py
contract), so module-level imports of identity helpers are intentionally
absent here."""

from __future__ import annotations

import json
from datetime import datetime
from types import FunctionType as _FunctionType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from litellm.integrations.aawm_agent_quality_rules import (
        AgentQualityCommand,
    )

    # Host-global function dependencies (resolved via __globals__ at runtime)
    def _agent_id_disallowed_values(*args: Any, **kwargs: Any) -> Any: ...

    def _apply_claude_auto_review_identity_to_record(*args: Any, **kwargs: Any) -> Any: ...

    def _build_session_history_latency_breakdown(*args: Any, **kwargs: Any) -> Any: ...

    def _build_session_runtime_identity(*args: Any, **kwargs: Any) -> Any: ...

    def _clean_non_empty_string(*args: Any, **kwargs: Any) -> Any: ...

    def _coerce_string_dict(*args: Any, **kwargs: Any) -> Any: ...

    def _compute_provider_cache_miss_cost_state(*args: Any, **kwargs: Any) -> Any: ...

    def _content_to_text(*args: Any, **kwargs: Any) -> Any: ...

    def _extract_command_text_from_tool_arguments(*args: Any, **kwargs: Any) -> Any: ...

    def _extract_file_paths_from_tool_arguments(*args: Any, **kwargs: Any) -> Any: ...

    def _extract_first_response_message(*args: Any, **kwargs: Any) -> Any: ...

    def _extract_repository_identity_from_metadata_sources_with_source(*args: Any, **kwargs: Any) -> Any: ...

    def _extract_request_headers_from_kwargs(*args: Any, **kwargs: Any) -> Any: ...

    def _extract_session_host_attribution(*args: Any, **kwargs: Any) -> Any: ...

    def _first_non_empty_string(*args: Any, **kwargs: Any) -> Any: ...

    def _first_non_none(*args: Any, **kwargs: Any) -> Any: ...

    def _is_codex_client_identity(*args: Any, **kwargs: Any) -> Any: ...

    def _is_known_aawm_workspace_repository(*args: Any, **kwargs: Any) -> Any: ...

    def _is_native_codex_passthrough_context(*args: Any, **kwargs: Any) -> Any: ...

    def _is_numeric_identity_placeholder(*args: Any, **kwargs: Any) -> Any: ...

    def _json_safe_rate_limit_value(*args: Any, **kwargs: Any) -> Any: ...

    def _maybe_get(*args: Any, **kwargs: Any) -> Any: ...

    def _metadata_bool(*args: Any, **kwargs: Any) -> Any: ...

    def _nonnegative_float_or_none(*args: Any, **kwargs: Any) -> Any: ...

    def _normalize_agent_id_identity(*args: Any, **kwargs: Any) -> Any: ...

    def _normalize_identity_for_placeholder_check(*args: Any, **kwargs: Any) -> Any: ...

    def _normalize_provider_cache_family(*args: Any, **kwargs: Any) -> Any: ...

    def _normalize_repository_identity(*args: Any, **kwargs: Any) -> Any: ...

    def _normalize_sensitive_config_change_state_on_record(*args: Any, **kwargs: Any) -> Any: ...

    def _normalize_session_history_provider(*args: Any, **kwargs: Any) -> Any: ...

    def _normalize_tenant_identity(*args: Any, **kwargs: Any) -> Any: ...

    def _positive_int_or_none(*args: Any, **kwargs: Any) -> Any: ...

    def _resolve_provider_cache_state(*args: Any, **kwargs: Any) -> Any: ...

    def _safe_float(*args: Any, **kwargs: Any) -> Any: ...

    def _safe_int(*args: Any, **kwargs: Any) -> Any: ...

    def score_agent_quality_context(*args: Any, **kwargs: Any) -> Any: ...

    # Host-global constant dependencies
    _AAWM_TOOL_DEFINITION_SNAPSHOT_METADATA_KEY: str = ""
    _PROMPT_OVERHEAD_TOKEN_FIELDS: Tuple[str, ...] = ()
    _SESSION_HISTORY_AGENT_SCORE_BOOL_FIELDS: Tuple[str, ...] = ()
    _SESSION_HISTORY_AGENT_SCORE_FLOAT_FIELDS: Tuple[str, ...] = ()
    _SESSION_HISTORY_AGENT_SCORE_INT_FIELDS: Tuple[str, ...] = ()
    _SESSION_HISTORY_LATENCY_FIELDS: Tuple[str, ...] = ()
    _SESSION_HISTORY_OUTPUT_CONTRACT_BOOL_FIELDS: Tuple[str, ...] = ()
    _SESSION_HISTORY_OUTPUT_CONTRACT_INT_FIELDS: Tuple[str, ...] = ()
    _SESSION_HISTORY_OUTPUT_CONTRACT_JSON_FIELDS: Tuple[str, ...] = ()
    _SESSION_HISTORY_OUTPUT_CONTRACT_STRING_FIELDS: Tuple[str, ...] = ()

_REQUEST_HEADER_TENANT_LITELLM_REPOSITORY_FRAGMENTS = (
    "harness",
    "validation",
)


_REPOSITORY_SOURCE_CODEX_MEMORY_METADATA_MARKERS = (
    ".metadata.",
    ".litellm_metadata.",
)


_REPOSITORY_SOURCE_GENERAL_METADATA_MARKERS = (
    ".metadata.",
    ".litellm_metadata.",
    ".request_metadata.",
    ".user_api_key_metadata.",
)


_REPOSITORY_SOURCE_TEXT_SUFFIXES = (
    ".text.environment_context.cwd",
    ".text.cwd_tag",
    ".text.agents_instructions",
    ".text.workspace_directories",
)


_GEMINI_CONTROL_PLANE_METHOD_LABELS = {
    "fetchadmincontrols": "google-fetch-admin-controls",
    "listexperiments": "google-list-experiments",
    "loadcodeassist": "google-load-code-assist",
    "retrieveuserquota": "google-retrieve-user-quota",
}


_GEMINI_CONTROL_PLANE_METHOD_NAMES = {
    "fetchadmincontrols": "fetchAdminControls",
    "listexperiments": "listExperiments",
    "loadcodeassist": "loadCodeAssist",
    "retrieveuserquota": "retrieveUserQuota",
}


def _normalize_reasoning_state(record: Dict[str, Any]) -> None:
    reported = _positive_int_or_none(record.get("reasoning_tokens_reported"))
    estimated = _positive_int_or_none(record.get("reasoning_tokens_estimated"))
    source = record.get("reasoning_tokens_source")
    reasoning_present = bool(record.get("reasoning_present") or record.get("thinking_signature_present"))

    record["reasoning_tokens_reported"] = reported
    record["reasoning_tokens_estimated"] = estimated

    if source == "provider_signature_present" and reported is not None:
        record["reasoning_tokens_source"] = "provider_signature_present"
    elif source == "provider_reported" and reported is not None:
        record["reasoning_tokens_source"] = "provider_reported"
    elif estimated is not None:
        record["reasoning_tokens_source"] = "estimated_from_reasoning_text"
    elif reasoning_present:
        record["reasoning_tokens_source"] = "not_available"
    else:
        record["reasoning_tokens_source"] = "not_applicable"


def _row_usage_object_from_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "prompt_tokens": int(record.get("input_tokens") or 0),
        "completion_tokens": int(record.get("output_tokens") or 0),
        "total_tokens": int(record.get("total_tokens") or 0),
        "cache_read_input_tokens": int(record.get("cache_read_input_tokens") or 0),
        "cache_creation_input_tokens": int(record.get("cache_creation_input_tokens") or 0),
    }


def _normalize_provider_cache_state_on_record(record: Dict[str, Any]) -> None:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    provider = _normalize_session_history_provider(
        record.get("provider"),
        str(record.get("model") or ""),
        metadata,
    )
    if provider is not None:
        record["provider"] = provider

    provider_family = _normalize_provider_cache_family(
        provider,
        str(record.get("model") or ""),
        metadata,
    )
    if provider_family is None:
        return

    cache_state = _resolve_provider_cache_state(
        provider=provider,
        model=str(record.get("model") or ""),
        usage_obj=_row_usage_object_from_record(record),
        metadata=metadata,
        request_body=None,
    )
    if cache_state is None:
        return

    current_status = record.get("provider_cache_status")
    if isinstance(current_status, str) and current_status.strip() and cache_state.get("status") == "not_attempted":
        return
    should_override = (
        not isinstance(current_status, str)
        or not current_status.strip()
        or bool(record.get("cache_read_input_tokens") or record.get("cache_creation_input_tokens"))
        or (
            bool(record.get("provider_cache_miss"))
            and (
                record.get("provider_cache_miss_token_count") is None
                or record.get("provider_cache_miss_cost_usd") is None
            )
        )
    )
    if not should_override:
        return

    cache_state = dict(cache_state)
    cache_state.update(
        _compute_provider_cache_miss_cost_state(
            provider_family=provider_family,
            model=str(record.get("model") or ""),
            usage_obj=_row_usage_object_from_record(record),
            cache_state=cache_state,
            metadata=metadata,
            response_cost_usd=_safe_float(record.get("response_cost_usd")),
        )
    )
    record["provider_cache_attempted"] = bool(cache_state.get("attempted"))
    record["provider_cache_status"] = cache_state.get("status")
    record["provider_cache_miss"] = bool(cache_state.get("miss"))
    record["provider_cache_miss_reason"] = cache_state.get("miss_reason")
    record["provider_cache_miss_token_count"] = cache_state.get("miss_token_count")
    record["provider_cache_miss_cost_usd"] = cache_state.get("miss_cost_usd")


def _normalize_session_runtime_identity_on_record(record: Dict[str, Any]) -> None:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    identity = _build_session_runtime_identity(
        metadata=metadata,
        kwargs=None,
        allow_runtime=False,
    )
    for key in (
        "litellm_environment",
        "litellm_version",
        "litellm_fork_version",
        "client_name",
        "client_version",
        "client_user_agent",
        "client_ip",
        "host_name",
    ):
        if not _clean_non_empty_string(record.get(key)):
            record[key] = identity.get(key)

    record_wheel_versions = _coerce_string_dict(record.get("litellm_wheel_versions"))
    metadata_wheel_versions = _coerce_string_dict(identity.get("litellm_wheel_versions"))
    record["litellm_wheel_versions"] = {
        **metadata_wheel_versions,
        **record_wheel_versions,
    }
    host_attribution = _extract_session_host_attribution(metadata)
    for key in ("client_ip", "host_name"):
        if not _clean_non_empty_string(record.get(key)):
            record[key] = host_attribution.get(key)


def _is_harness_tenant_identity(value: Any) -> bool:
    normalized = _normalize_identity_for_placeholder_check(value)
    if not normalized:
        return False
    return any(fragment in normalized for fragment in _REQUEST_HEADER_TENANT_LITELLM_REPOSITORY_FRAGMENTS)


def _normalize_request_header_tenant_repository(value: Any) -> Optional[str]:
    repository = _normalize_repository_identity(value)
    if repository is None:
        return None
    normalized = repository.lower()
    if any(fragment in normalized for fragment in _REQUEST_HEADER_TENANT_LITELLM_REPOSITORY_FRAGMENTS):
        return "litellm"
    if normalized.endswith("-dev") or "tenant" in normalized:
        return None
    return repository


def _normalize_repository_trust_source(value: Any) -> Optional[str]:
    source = _clean_non_empty_string(value)
    if not source:
        return None
    if source.endswith(".codex_memory_workflow"):
        return source[: -len(".codex_memory_workflow")]
    return source


def _repository_source_has_codex_memory_workflow(value: Any) -> bool:
    source = _clean_non_empty_string(value)
    return bool(source and source.endswith(".codex_memory_workflow"))


def _is_repository_source_trusted_common(
    value: Any,
    *,
    allow_general_metadata_markers: bool,
    allow_route_rollup_label: bool,
) -> bool:
    source = _normalize_repository_trust_source(value)
    if not source:
        return False

    if _repository_source_has_codex_memory_workflow(value) and any(
        marker in source for marker in _REPOSITORY_SOURCE_CODEX_MEMORY_METADATA_MARKERS
    ):
        return True
    if source == "tenant_id.request_headers":
        return True
    if source.startswith("request_headers."):
        return True
    if allow_general_metadata_markers and any(
        marker in source for marker in _REPOSITORY_SOURCE_GENERAL_METADATA_MARKERS
    ):
        return True
    if allow_route_rollup_label and source.endswith(".aawm_route_rollup_context.group_header_label"):
        return True
    if "x-codex-turn-metadata" in source and source.endswith(".text.project_path"):
        return True
    return any(source.endswith(marker) for marker in _REPOSITORY_SOURCE_TEXT_SUFFIXES)


def _is_repository_source_trusted_for_tenant(value: Any) -> bool:
    return _is_repository_source_trusted_common(
        value,
        allow_general_metadata_markers=True,
        allow_route_rollup_label=True,
    )


def _is_codex_trace_user_tenant_source(value: Any) -> bool:
    source = _clean_non_empty_string(value)
    if not source:
        return False
    normalized = source.lower()
    return normalized.endswith(".trace_user_id") or normalized == "trace_user_id"


def _is_codex_passthrough_tenant_extraction_context(
    kwargs: Dict[str, Any],
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    litellm_params = kwargs.get("litellm_params") or {}
    metadata = metadata or litellm_params.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    headers = _extract_request_headers_from_kwargs(kwargs)
    return bool(_is_native_codex_passthrough_context(metadata, headers) or _is_codex_client_identity(metadata, headers))


def _is_repository_source_trusted_for_codex_tenant(value: Any) -> bool:
    # Codex trust deliberately omits general metadata markers and the
    # route-rollup label that the general tenant helper accepts.
    return _is_repository_source_trusted_common(
        value,
        allow_general_metadata_markers=False,
        allow_route_rollup_label=False,
    )


def _is_codex_session_history_record(record: Dict[str, Any]) -> bool:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    route_family = _clean_non_empty_string(
        _first_non_none(
            metadata.get("passthrough_route_family"),
            metadata.get("openai_passthrough_route_family"),
        )
    )
    if route_family and route_family.lower() == "codex_responses":
        return True
    client_name = _clean_non_empty_string(_first_non_none(record.get("client_name"), metadata.get("client_name")))
    if client_name and "codex" in client_name.lower():
        return True
    trace_name = _clean_non_empty_string(metadata.get("trace_name"))
    user_agent = _clean_non_empty_string(
        _first_non_none(
            record.get("client_user_agent"),
            metadata.get("client_user_agent"),
            metadata.get("user_agent"),
        )
    )
    return bool(trace_name and trace_name.lower() == "codex" and user_agent and "codex" in user_agent.lower())


def _is_claude_session_history_record(record: Dict[str, Any]) -> bool:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    provider = str(record.get("provider") or "").strip().lower()
    if provider == "anthropic":
        return True
    client_name = _clean_non_empty_string(_first_non_none(record.get("client_name"), metadata.get("client_name")))
    if client_name and "claude" in client_name.lower():
        return True
    trace_name = _clean_non_empty_string(metadata.get("trace_name"))
    return bool(trace_name and "claude" in trace_name.lower())


def _is_claude_project_repository_source(value: Any) -> bool:
    source = _clean_non_empty_string(value)
    return bool(source and source.endswith(".aawm_claude_project"))


def _is_claude_metadata_tenant_source(value: Any) -> bool:
    source = _clean_non_empty_string(value)
    if not source:
        return False
    return source.endswith(".metadata.tenant_id") or source.endswith(".metadata.aawm_tenant_id")


def _claude_project_identity_is_trusted(
    record: Dict[str, Any],
    repository: Optional[str],
    source: Any,
) -> bool:
    if not (repository and _is_claude_session_history_record(record) and _is_claude_project_repository_source(source)):
        return True
    return _is_known_aawm_workspace_repository(repository)


def _codex_repository_source_trusted_for_record(record: Dict[str, Any]) -> bool:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    repository_source = metadata.get("repository_source")
    if _is_codex_session_history_record(record):
        return _is_repository_source_trusted_for_codex_tenant(repository_source)
    return _is_repository_source_trusted_for_tenant(repository_source)


def _clear_untrusted_codex_trace_user_tenant_on_record(
    record: Dict[str, Any],
    tenant_id: str,
) -> bool:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    if not _is_codex_session_history_record(record):
        return False
    if not _is_codex_trace_user_tenant_source(metadata.get("tenant_id_source")):
        trace_user_id = _normalize_repository_identity(metadata.get("trace_user_id"))
        normalized_tenant = _normalize_tenant_identity(tenant_id)
        if not (
            trace_user_id
            and normalized_tenant
            and trace_user_id == normalized_tenant
            and not _codex_tenant_source_trusted_for_record(record)
        ):
            return False

    metadata = dict(metadata)
    metadata["aawm_original_tenant_id"] = tenant_id
    metadata.pop("tenant_id", None)
    metadata["tenant_id_source"] = "trace_user_untrusted"
    metadata["trace_user_tenant_fallback_skipped"] = True
    record["tenant_id"] = None
    record["metadata"] = metadata
    return True


def _mark_codex_trace_user_tenant_skipped(
    record: Dict[str, Any],
    original_tenant_id: Optional[str],
) -> None:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    metadata = dict(metadata)
    if original_tenant_id:
        metadata.setdefault("aawm_original_tenant_id", original_tenant_id)
    metadata.pop("tenant_id", None)
    metadata["tenant_id_source"] = "trace_user_untrusted"
    metadata["trace_user_tenant_fallback_skipped"] = True
    record["tenant_id"] = None
    record["metadata"] = metadata


def _codex_untrusted_repository_reason(metadata: Dict[str, Any]) -> str:
    repository_source = _clean_non_empty_string(metadata.get("repository_source")) or ""
    if ".metadata.repository" in repository_source:
        return "untrusted_metadata_repository_label"
    if ".text." in repository_source or "project_path" in repository_source:
        return "untrusted_prompt_text_repository_candidate"
    return "untrusted_repository_tenant_source"


def _mark_repository_unresolved_metadata(metadata: Dict[str, Any]) -> None:
    metadata["session_history_repository_status"] = "unresolved"
    metadata["session_history_repository_unresolved"] = True
    metadata["session_history_repository_unresolved_reason"] = _codex_untrusted_repository_reason(metadata)


def _session_history_missing_repository_reason(record: Dict[str, Any]) -> str:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    provider = str(record.get("provider") or "").strip().lower()
    client_name = str(metadata.get("client_name") or "").strip().lower()
    if provider == "anthropic" or "claude" in client_name:
        return "no_trusted_claude_project_signal"
    if provider in {"xai", "grok"} or "grok" in str(record.get("model") or "").lower():
        return "no_trusted_grok_project_signal"
    return "no_trusted_repository_signal"


def _mark_missing_repository_unresolved(
    record: Dict[str, Any],
) -> None:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    else:
        metadata = dict(metadata)
    if _metadata_bool(metadata.get("session_history_reporting_excluded")):
        record["metadata"] = metadata
        return
    if metadata.get("session_history_repository_status") == "unresolved":
        record["metadata"] = metadata
        return
    metadata["session_history_repository_status"] = "unresolved"
    metadata["session_history_repository_unresolved"] = True
    metadata["session_history_repository_unresolved_reason"] = _session_history_missing_repository_reason(record)
    record["metadata"] = metadata


def _clear_untrusted_claude_project_repository_on_record(
    record: Dict[str, Any],
    repository: Optional[str],
    repository_source: Any,
) -> Optional[str]:
    if _claude_project_identity_is_trusted(record, repository, repository_source):
        return repository
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    else:
        metadata = dict(metadata)
    if repository:
        metadata["aawm_original_repository"] = repository
    source = _clean_non_empty_string(repository_source)
    if source:
        metadata["repository_source_untrusted"] = source
    metadata.pop("repository", None)
    record["metadata"] = metadata
    return None


def _clear_untrusted_claude_metadata_tenant_on_record(
    record: Dict[str, Any],
    tenant_id: str,
) -> bool:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    if not (
        _is_claude_session_history_record(record)
        and not _is_known_aawm_workspace_repository(tenant_id)
        and _is_claude_metadata_tenant_source(metadata.get("tenant_id_source"))
    ):
        return False
    metadata = dict(metadata)
    metadata["aawm_original_tenant_id"] = tenant_id
    tenant_source = _clean_non_empty_string(metadata.get("tenant_id_source"))
    if tenant_source:
        metadata["tenant_id_source_untrusted"] = tenant_source
    metadata.pop("tenant_id", None)
    record["tenant_id"] = None
    record["metadata"] = metadata
    if _normalize_repository_identity(record.get("repository")) is None:
        _mark_missing_repository_unresolved(record)
    return True


def _clear_repository_unresolved_metadata(metadata: Dict[str, Any]) -> None:
    metadata.pop("session_history_repository_unresolved", None)
    metadata.pop("session_history_repository_unresolved_reason", None)
    if metadata.get("session_history_repository_status") == "unresolved":
        metadata.pop("session_history_repository_status", None)


def _mark_codex_repository_tenant_skipped(
    record: Dict[str, Any],
    original_tenant_id: Optional[str] = None,
) -> None:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    metadata = dict(metadata)
    if original_tenant_id:
        metadata["aawm_original_tenant_id"] = original_tenant_id
    metadata.pop("tenant_id", None)
    metadata["tenant_id_source"] = "repository_untrusted"
    metadata["repository_tenant_fallback_skipped"] = True
    _mark_repository_unresolved_metadata(metadata)
    record["tenant_id"] = None
    record["metadata"] = metadata


def _clear_codex_trace_user_tenant_source_on_record(record: Dict[str, Any]) -> bool:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    if not (
        _is_codex_session_history_record(record)
        and _is_codex_trace_user_tenant_source(metadata.get("tenant_id_source"))
    ):
        return False

    original_tenant_id = _clean_non_empty_string(record.get("tenant_id")) or _clean_non_empty_string(
        metadata.get("tenant_id")
    )
    if original_tenant_id is None:
        original_tenant_id = _normalize_repository_identity(metadata.get("trace_user_id"))
    _mark_codex_trace_user_tenant_skipped(record, original_tenant_id)
    return True


def _clear_untrusted_codex_tenant_on_record(
    record: Dict[str, Any],
    tenant_id: str,
) -> bool:
    if not _is_codex_session_history_record(record):
        return False
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    if _is_codex_trace_user_tenant_source(metadata.get("tenant_id_source")) or (
        _normalize_repository_identity(metadata.get("trace_user_id")) == tenant_id
        and not _codex_tenant_source_trusted_for_record(record)
    ):
        _mark_codex_trace_user_tenant_skipped(record, tenant_id)
        return True
    if _clear_untrusted_codex_trace_user_tenant_on_record(record, tenant_id):
        return True
    if _clear_untrusted_codex_repository_tenant_on_record(record, tenant_id):
        return True
    if not _codex_tenant_source_trusted_for_record(record):
        _mark_codex_repository_tenant_skipped(record, tenant_id)
        return True
    return False


def _codex_tenant_source_trusted_for_record(record: Dict[str, Any]) -> bool:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    tenant_source = metadata.get("tenant_id_source")
    if _is_codex_trace_user_tenant_source(tenant_source):
        return False
    if tenant_source == "repository":
        return _codex_repository_source_trusted_for_record(record)
    if tenant_source in {
        "request_headers",
        "harness_tenant_repository",
        "agent_context_text",
    }:
        return True
    if isinstance(tenant_source, str) and tenant_source.startswith("request_headers."):
        return True
    if isinstance(tenant_source, str) and ".trace_user_id" in tenant_source:
        return False
    if isinstance(tenant_source, str) and any(
        marker in tenant_source
        for marker in (
            ".metadata.tenant_id",
            ".metadata.aawm_tenant_id",
            ".litellm_metadata.tenant_id",
            ".litellm_metadata.aawm_tenant_id",
        )
    ):
        return _codex_repository_source_trusted_for_record(record)
    return tenant_source is None


def _clear_untrusted_codex_repository_tenant_on_record(
    record: Dict[str, Any],
    tenant_id: str,
) -> bool:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    if not (
        _is_codex_session_history_record(record)
        and metadata.get("tenant_id_source") == "repository"
        and not _codex_repository_source_trusted_for_record(record)
    ):
        return False

    metadata = dict(metadata)
    metadata["aawm_original_tenant_id"] = tenant_id
    metadata.pop("tenant_id", None)
    metadata["tenant_id_source"] = "repository_untrusted"
    metadata["repository_tenant_fallback_skipped"] = True
    _mark_repository_unresolved_metadata(metadata)
    record["tenant_id"] = None
    record["metadata"] = metadata
    return True


def _normalize_session_repository_on_record(record: Dict[str, Any]) -> None:
    repository = _normalize_repository_identity(record.get("repository"))
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    repository_source = metadata.get("repository_source")
    if repository is None:
        repository, repository_source = _extract_repository_identity_from_metadata_sources_with_source(
            ("record.metadata", metadata)
        )
        if repository is not None and repository_source:
            metadata = dict(metadata)
            metadata.setdefault("repository_source", repository_source)
            record["metadata"] = metadata
    if repository is None and metadata.get("tenant_id_source") == "request_headers":
        repository = _normalize_request_header_tenant_repository(
            record.get("tenant_id")
        ) or _normalize_request_header_tenant_repository(metadata.get("tenant_id"))
        if repository is not None:
            metadata = dict(metadata)
            metadata.setdefault("repository_source", "tenant_id.request_headers")
            record["metadata"] = metadata
            repository_source = metadata.get("repository_source")
    repository = _clear_untrusted_claude_project_repository_on_record(
        record,
        repository,
        repository_source,
    )
    record["repository"] = repository


def _can_promote_known_codex_repository_to_tenant(
    repository: str,
    metadata: Dict[str, Any],
) -> bool:
    # Bounded relaxation for Codex: generic metadata.repository (or
    # litellm_metadata.repository) may promote to tenant ONLY when the
    # normalized repository label is a known AAWM workspace repo from the
    # conservative built-in allowlist (or AAWM_KNOWN_WORKSPACE_REPOS env).
    # Headers, x-codex-turn-metadata project_path, cwd/workspace text, and
    # other previously trusted sources remain trusted without the name check.
    repo_source = metadata.get("repository_source")
    return _is_known_aawm_workspace_repository(repository) and (
        metadata.get("tenant_id_source") in {"repository_untrusted", "trace_user_untrusted"}
        or metadata.get("trace_user_tenant_fallback_skipped") is True
        or metadata.get("repository_tenant_fallback_skipped") is True
        or (
            isinstance(repo_source, str)
            and (".metadata.repository" in repo_source or "litellm_metadata.repository" in repo_source)
        )
    )


def _normalize_session_tenant_on_record(record: Dict[str, Any]) -> None:
    _clear_codex_trace_user_tenant_source_on_record(record)

    raw_tenant_id = record.get("tenant_id")
    if _is_harness_tenant_identity(raw_tenant_id):
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        else:
            metadata = dict(metadata)

        original_tenant_id = _clean_non_empty_string(raw_tenant_id)
        if original_tenant_id:
            metadata["aawm_original_tenant_id"] = original_tenant_id
        metadata["aawm_harness_tenant_alias"] = True

        repository = _normalize_repository_identity(
            record.get("repository")
        ) or _normalize_request_header_tenant_repository(raw_tenant_id)
        record["tenant_id"] = repository
        if repository is not None:
            metadata["tenant_id"] = repository
            metadata["tenant_id_source"] = "harness_tenant_repository"
        else:
            metadata.pop("tenant_id", None)
            metadata["tenant_id_source"] = "harness_tenant_excluded"
        record["metadata"] = metadata
        return

    tenant_id = _normalize_tenant_identity(record.get("tenant_id"))
    if tenant_id:
        if _clear_untrusted_claude_metadata_tenant_on_record(record, tenant_id):
            return
        _clear_untrusted_codex_tenant_on_record(record, tenant_id)
        tenant_id = _normalize_tenant_identity(record.get("tenant_id"))
        if not tenant_id:
            # Continue into repository fallback. A stale Codex tenant can be
            # rejected while the same row still has a trusted current repo.
            pass
        else:
            record["tenant_id"] = tenant_id
            return

    repository = _normalize_repository_identity(record.get("repository"))
    if repository is None:
        record["tenant_id"] = None
        _mark_missing_repository_unresolved(record)
        return

    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    if not _codex_repository_source_trusted_for_record(record):
        if not _can_promote_known_codex_repository_to_tenant(repository, metadata):
            _mark_codex_repository_tenant_skipped(record)
            return

    record["tenant_id"] = repository
    metadata = dict(metadata)
    metadata["tenant_id"] = repository
    metadata["tenant_id_source"] = "repository"
    metadata.pop("repository_tenant_fallback_skipped", None)
    _clear_repository_unresolved_metadata(metadata)
    record["metadata"] = metadata


def _sync_session_history_record_metadata(record: Dict[str, Any]) -> None:  # noqa: PLR0915
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    else:
        metadata = dict(metadata)

    reasoning_source = record.get("reasoning_tokens_source")
    if isinstance(reasoning_source, str) and reasoning_source.strip():
        metadata["usage_reasoning_tokens_source"] = reasoning_source

    if record.get("reasoning_tokens_reported") is not None:
        metadata["usage_reasoning_tokens_reported"] = record["reasoning_tokens_reported"]
    else:
        metadata.pop("usage_reasoning_tokens_reported", None)

    if record.get("reasoning_tokens_estimated") is not None:
        metadata["usage_reasoning_tokens_estimated"] = record["reasoning_tokens_estimated"]
    else:
        metadata.pop("usage_reasoning_tokens_estimated", None)

    for field in _PROMPT_OVERHEAD_TOKEN_FIELDS:
        metadata[f"usage_{field}"] = int(record.get(field) or 0)

    metadata["usage_invalid_tool_call_count"] = int(record.get("invalid_tool_call_count") or 0)

    metadata["usage_structured_output_attempted"] = bool(record.get("structured_output_attempted"))
    metadata["usage_structured_output_failed"] = bool(record.get("structured_output_failed"))
    for field, metadata_key in (
        ("structured_output_mode", "usage_structured_output_mode"),
        ("structured_output_schema_hash", "usage_structured_output_schema_hash"),
        (
            "structured_output_failure_reason",
            "usage_structured_output_failure_reason",
        ),
    ):
        value = _clean_non_empty_string(record.get(field))
        if value is None:
            metadata.pop(metadata_key, None)
        else:
            metadata[metadata_key] = value

    metadata["is_compact_summary"] = bool(record.get("is_compact_summary"))
    for field in (
        "compact_summary_source",
        "compact_summary_role",
        "compact_summary_id",
    ):
        value = _clean_non_empty_string(record.get(field))
        if value is None:
            metadata.pop(field, None)
        else:
            metadata[field] = value

    for field in _SESSION_HISTORY_AGENT_SCORE_FLOAT_FIELDS:
        float_value = _safe_float(record.get(field))
        metadata_key = f"usage_{field}"
        if float_value is None:
            metadata.pop(metadata_key, None)
        else:
            metadata[metadata_key] = float_value

    for field in _SESSION_HISTORY_AGENT_SCORE_BOOL_FIELDS:
        bool_value = _optional_metadata_bool(record.get(field))
        metadata_key = f"usage_{field}"
        if bool_value is None:
            metadata.pop(metadata_key, None)
        else:
            metadata[metadata_key] = bool_value

    for field in _SESSION_HISTORY_AGENT_SCORE_INT_FIELDS:
        int_value = _safe_int(record.get(field))
        metadata_key = f"usage_{field}"
        if int_value is None:
            metadata.pop(metadata_key, None)
        else:
            metadata[metadata_key] = int_value

    for field in _SESSION_HISTORY_OUTPUT_CONTRACT_STRING_FIELDS:
        value = _clean_non_empty_string(record.get(field))
        metadata_key = f"usage_{field}"
        if value is None:
            metadata.pop(metadata_key, None)
        else:
            metadata[metadata_key] = value

    for field in _SESSION_HISTORY_OUTPUT_CONTRACT_BOOL_FIELDS:
        bool_value = _optional_metadata_bool(record.get(field))
        metadata_key = f"usage_{field}"
        if bool_value is None:
            metadata.pop(metadata_key, None)
        else:
            metadata[metadata_key] = bool_value

    for field in _SESSION_HISTORY_OUTPUT_CONTRACT_INT_FIELDS:
        int_value = _safe_int(record.get(field))
        metadata_key = f"usage_{field}"
        if int_value is None:
            metadata.pop(metadata_key, None)
        else:
            metadata[metadata_key] = int_value

    for field in _SESSION_HISTORY_OUTPUT_CONTRACT_JSON_FIELDS:
        value = record.get(field)
        metadata_key = f"usage_{field}"
        if value in (None, [], {}):
            metadata.pop(metadata_key, None)
        else:
            metadata[metadata_key] = _json_safe_rate_limit_value(value)

    agent_score_reasons = _normalize_agent_score_reasons(record.get("agent_score_reasons"))
    if agent_score_reasons:
        metadata["usage_agent_score_reasons"] = agent_score_reasons
    else:
        metadata.pop("usage_agent_score_reasons", None)

    provider_family = _normalize_provider_cache_family(
        record.get("provider"),
        str(record.get("model") or ""),
        metadata,
    )
    cache_status = record.get("provider_cache_status")
    if provider_family is not None and isinstance(cache_status, str) and cache_status.strip():
        cache_values: Dict[str, Any] = {
            "provider_cache_attempted": bool(record.get("provider_cache_attempted")),
            "provider_cache_status": cache_status,
            "provider_cache_miss": bool(record.get("provider_cache_miss")),
            "provider_cache_miss_reason": record.get("provider_cache_miss_reason"),
            "provider_cache_miss_token_count": record.get("provider_cache_miss_token_count"),
            "provider_cache_miss_cost_usd": record.get("provider_cache_miss_cost_usd"),
        }
        for suffix, value in cache_values.items():
            generic_key = f"usage_{suffix}"
            provider_key = f"{provider_family}_{suffix}"
            if value is None or value == "":
                metadata.pop(generic_key, None)
                metadata.pop(provider_key, None)
            else:
                metadata[generic_key] = value
                metadata[provider_key] = value

    for key in (
        "litellm_environment",
        "litellm_version",
        "litellm_fork_version",
        "client_name",
        "client_version",
        "client_user_agent",
        "client_ip",
        "host_name",
    ):
        value = _clean_non_empty_string(record.get(key))
        if value is not None:
            metadata[key] = value

    wheel_versions = _coerce_string_dict(record.get("litellm_wheel_versions"))
    if wheel_versions:
        metadata["litellm_wheel_versions"] = wheel_versions

    repository = _normalize_repository_identity(record.get("repository"))
    if repository is not None:
        metadata["repository"] = repository
    else:
        metadata.pop("repository", None)

    tenant_id = _normalize_tenant_identity(record.get("tenant_id"))
    if tenant_id is not None:
        metadata["tenant_id"] = tenant_id
    else:
        metadata.pop("tenant_id", None)
        if metadata.get("trace_user_tenant_fallback_skipped") is True:
            metadata.setdefault("tenant_id_source", "trace_user_untrusted")
        elif metadata.get("repository_tenant_fallback_skipped") is True:
            metadata.setdefault("tenant_id_source", "repository_untrusted")
        else:
            metadata.pop("tenant_id_source", None)

    if _is_numeric_identity_placeholder(metadata.get("trace_user_id")):
        metadata.pop("trace_user_id", None)

    record["metadata"] = metadata


def _normalize_prompt_overhead_state_on_record(record: Dict[str, Any]) -> None:
    for field in _PROMPT_OVERHEAD_TOKEN_FIELDS:
        value = _safe_int(record.get(field))
        record[field] = value if value is not None else 0


def _normalize_invalid_tool_call_state_on_record(record: Dict[str, Any]) -> None:
    value = _safe_int(record.get("invalid_tool_call_count"))
    record["invalid_tool_call_count"] = value if value is not None and value > 0 else 0


def _normalize_structured_output_state_on_record(record: Dict[str, Any]) -> None:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    attempted_value = record.get("structured_output_attempted")
    failed_value = record.get("structured_output_failed")
    attempted = (
        _metadata_bool(attempted_value)
        if attempted_value is not None
        else _metadata_bool(metadata.get("usage_structured_output_attempted"))
    )
    failed = (
        _metadata_bool(failed_value)
        if failed_value is not None
        else _metadata_bool(metadata.get("usage_structured_output_failed"))
    )
    if failed:
        attempted = True

    record["structured_output_attempted"] = attempted
    record["structured_output_failed"] = failed
    record["structured_output_mode"] = _first_non_empty_string(
        record.get("structured_output_mode"),
        metadata.get("usage_structured_output_mode"),
        metadata.get("structured_output_mode"),
    )
    record["structured_output_schema_hash"] = _first_non_empty_string(
        record.get("structured_output_schema_hash"),
        metadata.get("usage_structured_output_schema_hash"),
        metadata.get("structured_output_schema_hash"),
    )
    record["structured_output_failure_reason"] = _first_non_empty_string(
        record.get("structured_output_failure_reason"),
        metadata.get("usage_structured_output_failure_reason"),
        metadata.get("structured_output_failure_reason"),
    )
    if not failed:
        record["structured_output_failure_reason"] = None


def _normalize_compact_summary_state_on_record(record: Dict[str, Any]) -> None:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    is_compact = (
        _optional_metadata_bool(record.get("is_compact_summary"))
        if record.get("is_compact_summary") is not None
        else _optional_metadata_bool(metadata.get("is_compact_summary"))
    )
    record["is_compact_summary"] = bool(is_compact)
    record["compact_summary_source"] = _first_non_empty_string(
        record.get("compact_summary_source"),
        metadata.get("compact_summary_source"),
    )
    record["compact_summary_role"] = _first_non_empty_string(
        record.get("compact_summary_role"),
        metadata.get("compact_summary_role"),
    )
    record["compact_summary_id"] = _first_non_empty_string(
        record.get("compact_summary_id"),
        metadata.get("compact_summary_id"),
    )

    if record["is_compact_summary"]:
        record["compact_summary_role"] = record["compact_summary_role"] or "event"


def _optional_metadata_bool(value: Any) -> Optional[bool]:
    if value is None or value == "":
        return None
    return _metadata_bool(value)


def _normalize_agent_score_reasons(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {}
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def _append_agent_quality_text(
    *,
    role: Optional[str],
    content: Any,
    user_texts: List[str],
    assistant_texts: List[str],
    tool_result_texts: List[str],
) -> None:
    text = _content_to_text(content).strip()
    if not text:
        return
    role_lower = str(role or "").lower()
    if role_lower in {"assistant", "model"}:
        assistant_texts.append(text)
    elif role_lower in {"tool", "function"}:
        tool_result_texts.append(text)
    else:
        user_texts.append(text)


def _append_agent_quality_command_from_arguments(
    *,
    commands: List[AgentQualityCommand],
    name: str,
    arguments: Any,
) -> None:
    command_text = _extract_command_text_from_tool_arguments(arguments)
    if not command_text:
        return
    commands.append(
        AgentQualityCommand(
            name=name,
            command=command_text,
            affected_paths=tuple(_extract_file_paths_from_tool_arguments(arguments)),
        )
    )


def _append_agent_quality_commands_from_message(
    *,
    message: Dict[str, Any],
    commands: List[AgentQualityCommand],
) -> None:
    content = message.get("content")
    content_blocks = content if isinstance(content, list) else [content]
    for block in content_blocks:
        if not isinstance(block, dict):
            continue
        block_type = str(block.get("type") or "").lower()
        if block_type not in {"tool_use", "function_call", "custom_tool_call"}:
            continue
        arguments = block.get("input")
        if arguments is None:
            arguments = block.get("arguments")
        _append_agent_quality_command_from_arguments(
            commands=commands,
            name=str(block.get("name") or block_type or "tool"),
            arguments=arguments,
        )

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if isinstance(function, dict):
                _append_agent_quality_command_from_arguments(
                    commands=commands,
                    name=str(function.get("name") or tool_call.get("type") or "tool"),
                    arguments=function.get("arguments"),
                )
                continue
            _append_agent_quality_command_from_arguments(
                commands=commands,
                name=str(tool_call.get("name") or tool_call.get("type") or "tool"),
                arguments=tool_call.get("arguments") or tool_call.get("input"),
            )


def _collect_agent_quality_context_from_request_body(
    request_body: Any,
) -> Tuple[List[str], List[str], List[str], List[AgentQualityCommand]]:
    user_texts: List[str] = []
    assistant_texts: List[str] = []
    tool_result_texts: List[str] = []
    commands: List[AgentQualityCommand] = []
    if not isinstance(request_body, dict):
        return user_texts, assistant_texts, tool_result_texts, commands

    messages = request_body.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "")
            content = message.get("content")
            if role.lower() == "user" and isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        tool_result_texts.append(_content_to_text(block.get("content")))
            if role.lower() in {"assistant", "model"}:
                _append_agent_quality_commands_from_message(
                    message=message,
                    commands=commands,
                )
            _append_agent_quality_text(
                role=role,
                content=content,
                user_texts=user_texts,
                assistant_texts=assistant_texts,
                tool_result_texts=tool_result_texts,
            )

    input_items = request_body.get("input")
    if isinstance(input_items, list):
        for item in input_items:
            if not isinstance(item, dict):
                user_texts.append(_content_to_text(item))
                continue
            item_type = str(item.get("type") or "").lower()
            role = str(item.get("role") or "")
            if item_type in {"message", ""} or role:
                _append_agent_quality_text(
                    role=role or "user",
                    content=item.get("content"),
                    user_texts=user_texts,
                    assistant_texts=assistant_texts,
                    tool_result_texts=tool_result_texts,
                )
                continue
            if item_type in {"function_call_output", "tool_result"}:
                tool_result_texts.append(_content_to_text(item.get("output")))
                continue
            if item_type in {"function_call", "tool_use", "custom_tool_call"}:
                command_text = _extract_command_text_from_tool_arguments(item.get("arguments") or item.get("input"))
                if command_text:
                    commands.append(
                        AgentQualityCommand(
                            name=str(item.get("name") or item_type),
                            command=command_text,
                            affected_paths=tuple(
                                _extract_file_paths_from_tool_arguments(item.get("arguments") or item.get("input"))
                            ),
                        )
                    )

    return user_texts, assistant_texts, tool_result_texts, commands


def _collect_agent_quality_response_texts(result: Any) -> List[str]:
    assistant_texts: List[str] = []
    message = _extract_first_response_message(result)
    if message is not None:
        text = _content_to_text(_maybe_get(message, "content")).strip()
        if text:
            assistant_texts.append(text)

    output_items = _maybe_get(result, "output")
    if isinstance(output_items, list):
        for item in output_items:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "")
            if item.get("type") == "message" or role == "assistant":
                text = _content_to_text(item.get("content")).strip()
                if text:
                    assistant_texts.append(text)
    return assistant_texts


def _agent_quality_commands_from_tool_activity(
    tool_activity: List[Dict[str, Any]],
) -> List[AgentQualityCommand]:
    commands: List[AgentQualityCommand] = []
    for item in tool_activity:
        if not isinstance(item, dict):
            continue
        command_text = item.get("command_text")
        if not isinstance(command_text, str) or not command_text.strip():
            continue
        affected_paths = tuple(
            value
            for value in (list(item.get("file_paths_modified") or []) + list(item.get("file_paths_read") or []))
            if isinstance(value, str)
        )
        commands.append(
            AgentQualityCommand(
                name=str(item.get("tool_name") or ""),
                command=command_text,
                affected_paths=affected_paths,
            )
        )
    return commands


def _apply_runtime_agent_quality_scores(
    *,
    record: Dict[str, Any],
    request_body: Any,
    result: Any,
    tool_activity: List[Dict[str, Any]],
) -> None:
    user_texts, assistant_texts, tool_result_texts, commands = _collect_agent_quality_context_from_request_body(
        request_body
    )
    assistant_texts.extend(_collect_agent_quality_response_texts(result))
    commands.extend(_agent_quality_commands_from_tool_activity(tool_activity))

    start_time = record.get("start_time")
    end_time = record.get("end_time")
    elapsed_ms: Optional[float] = None
    if isinstance(start_time, datetime) and isinstance(end_time, datetime):
        elapsed_ms = max(0.0, (end_time - start_time).total_seconds() * 1000)

    task_progress = bool(
        record.get("output_tokens")
        or record.get("tool_call_count")
        or record.get("file_modified_count")
        or record.get("git_commit_count")
    )
    result_scores = score_agent_quality_context(
        user_texts=user_texts,
        assistant_texts=assistant_texts,
        tool_result_texts=tool_result_texts,
        commands=commands,
        input_tokens=_safe_int(record.get("input_tokens")) or 0,
        output_tokens=_safe_int(record.get("output_tokens")) or 0,
        elapsed_ms=elapsed_ms,
        task_progress=task_progress,
    )
    for field, value in result_scores.fields.items():
        if record.get(field) is None:
            record[field] = value

    reasons = _normalize_agent_score_reasons(record.get("agent_score_reasons"))
    for key, value in result_scores.reasons.items():
        if value:
            reasons[key] = value
    record["agent_score_reasons"] = reasons


def _normalize_agent_score_state_on_record(record: Dict[str, Any]) -> None:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    for field in _SESSION_HISTORY_AGENT_SCORE_FLOAT_FIELDS:
        record[field] = _first_non_none(
            _safe_float(record.get(field)),
            _safe_float(metadata.get(f"usage_{field}")),
            _safe_float(metadata.get(field)),
        )

    for field in _SESSION_HISTORY_AGENT_SCORE_BOOL_FIELDS:
        record[field] = _first_non_none(
            _optional_metadata_bool(record.get(field)),
            _optional_metadata_bool(metadata.get(f"usage_{field}")),
            _optional_metadata_bool(metadata.get(field)),
        )

    for field in _SESSION_HISTORY_AGENT_SCORE_INT_FIELDS:
        value = _first_non_none(
            _safe_int(record.get(field)),
            _safe_int(metadata.get(f"usage_{field}")),
            _safe_int(metadata.get(field)),
        )
        record[field] = value if value is not None and value >= 0 else None

    for field in _SESSION_HISTORY_OUTPUT_CONTRACT_STRING_FIELDS:
        record[field] = _first_non_empty_string(
            record.get(field),
            metadata.get(f"usage_{field}"),
            metadata.get(field),
        )

    for field in _SESSION_HISTORY_OUTPUT_CONTRACT_BOOL_FIELDS:
        record[field] = _first_non_none(
            _optional_metadata_bool(record.get(field)),
            _optional_metadata_bool(metadata.get(f"usage_{field}")),
            _optional_metadata_bool(metadata.get(field)),
        )

    for field in _SESSION_HISTORY_OUTPUT_CONTRACT_INT_FIELDS:
        value = _first_non_none(
            _safe_int(record.get(field)),
            _safe_int(metadata.get(f"usage_{field}")),
            _safe_int(metadata.get(field)),
        )
        record[field] = value if value is not None and value >= 0 else None

    for field in _SESSION_HISTORY_OUTPUT_CONTRACT_JSON_FIELDS:
        record[field] = _first_non_none(
            record.get(field),
            metadata.get(f"usage_{field}"),
            metadata.get(field),
        )

    metadata_reasons = _normalize_agent_score_reasons(
        _first_non_none(
            metadata.get("usage_agent_score_reasons"),
            metadata.get("agent_score_reasons"),
        )
    )
    record_reasons = _normalize_agent_score_reasons(record.get("agent_score_reasons"))
    record["agent_score_reasons"] = {
        **metadata_reasons,
        **record_reasons,
    }


def _normalize_session_latency_state_on_record(record: Dict[str, Any]) -> None:
    derived_latency = _build_session_history_latency_breakdown(
        metadata=record.get("metadata"),
        start_time=record.get("start_time"),
        end_time=record.get("end_time"),
    )
    for field in _SESSION_HISTORY_LATENCY_FIELDS:
        explicit_value = _nonnegative_float_or_none(record.get(field))
        record[field] = explicit_value if explicit_value is not None else derived_latency.get(field)


def _extract_gemini_control_plane_method_from_record(
    record: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Optional[str]:
    candidates = (
        record.get("call_type"),
        record.get("model"),
        metadata.get("user_api_key_request_route"),
        metadata.get("passthrough_route_family"),
        metadata.get("aawm_local_route"),
        metadata.get("aawm_local_endpoint"),
    )
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        candidate_lower = candidate.lower()
        for method_lower in _GEMINI_CONTROL_PLANE_METHOD_LABELS:
            if method_lower in candidate_lower:
                return method_lower
    return None


def _session_history_record_provider_usage_token_total(record: Dict[str, Any]) -> int:
    total = 0
    for field in (
        "input_tokens",
        "output_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
        "reasoning_tokens_reported",
    ):
        value = _safe_int(record.get(field))
        if value is not None and value > 0:
            total += value
    return total


def _classify_zero_token_session_history_record(record: Dict[str, Any]) -> None:
    if _session_history_record_provider_usage_token_total(record) > 0:
        return

    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    else:
        metadata = dict(metadata)

    provider = str(record.get("provider") or "").strip().lower()
    zero_token_class: Optional[str] = None
    zero_token_reason: Optional[str] = None

    gemini_control_plane_method = _extract_gemini_control_plane_method_from_record(
        record,
        metadata,
    )
    has_gemini_quota_payload = isinstance(metadata.get("google_retrieve_user_quota"), dict)
    if provider in {"gemini", "google"} and (
        metadata.get("aawm_rate_limit_observation_only") is True
        or has_gemini_quota_payload
        or gemini_control_plane_method is not None
    ):
        zero_token_class = "non_usage_rate_limit_observation"
        zero_token_reason = "gemini_control_plane_rate_limit_payload"
        if gemini_control_plane_method is not None:
            metadata.setdefault(
                "gemini_control_plane_method",
                _GEMINI_CONTROL_PLANE_METHOD_NAMES[gemini_control_plane_method],
            )
            metadata["gemini_control_plane_excluded"] = True
            model = _clean_non_empty_string(record.get("model"))
            if model is None or model.lower() in {"unknown", "null", "none"}:
                record["model"] = _GEMINI_CONTROL_PLANE_METHOD_LABELS[gemini_control_plane_method]
    elif (
        provider == "gemini"
        and metadata.get("codex_adapter_output_shape") == "openai_responses"
        and _safe_int(metadata.get("aawm_stream_chunk_count")) is not None
    ):
        zero_token_class = "empty_provider_response_no_usage"
        zero_token_reason = "gemini_code_assist_adapter_empty_response"
    elif metadata.get("source_status") == "failure":
        zero_token_class = "failed_observation_no_usage"
        zero_token_reason = "langfuse_observation_failed_without_usage"
    elif (
        provider in {"xai", "grok"}
        and str(record.get("model") or "").strip().lower() == "unknown"
        and str(metadata.get("passthrough_route_family") or "").strip().lower() == "grok_cli_chat_proxy"
    ):
        inferred_model = _first_non_empty_string(
            metadata.get("grok_model_override"),
            metadata.get("model_group"),
            record.get("model_group"),
        )
        if inferred_model is None or inferred_model.lower() in {
            "unknown",
            "null",
            "none",
        }:
            inferred_model = "grok-build"
            metadata.setdefault("grok_side_channel_model_defaulted", True)
            metadata.setdefault(
                "grok_side_channel_model_default_reason",
                "grok_cli_side_channel_without_request_model",
            )

        record["model"] = inferred_model
        if _clean_non_empty_string(record.get("model_group")) is None:
            record["model_group"] = inferred_model
        metadata.setdefault("model_group", inferred_model)
        zero_token_class = "grok_cli_side_channel_no_usage"
        zero_token_reason = "grok_side_channel_without_model_usage"
        metadata["session_history_reporting_excluded"] = True
        metadata["session_history_reporting_exclusion_reason"] = zero_token_reason
        metadata["grok_side_channel_excluded"] = True

    if zero_token_class is not None:
        metadata.setdefault("session_history_usage_record", False)
        metadata.setdefault("session_history_zero_token_class", zero_token_class)
        metadata.setdefault("d1_140_zero_token_class", zero_token_class)
        if zero_token_reason is not None:
            metadata.setdefault("d1_140_zero_token_reason", zero_token_reason)

    record["metadata"] = metadata


def _normalize_session_history_record(record: Dict[str, Any]) -> Dict[str, Any]:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    record["provider_response_id"] = _first_non_empty_string(
        record.get("provider_response_id"),
        metadata.get("provider_response_id"),
        metadata.get("response_id"),
    )
    _normalize_agent_id_on_record(record)
    _normalize_inbound_model_alias_on_record(record)
    _normalize_reasoning_state(record)
    _normalize_provider_cache_state_on_record(record)
    _normalize_invalid_tool_call_state_on_record(record)
    _normalize_structured_output_state_on_record(record)
    _normalize_compact_summary_state_on_record(record)
    _normalize_reporting_exclusion_state_on_record(record)
    _normalize_agent_score_state_on_record(record)
    _normalize_prompt_overhead_state_on_record(record)
    _normalize_session_runtime_identity_on_record(record)
    _apply_claude_auto_review_identity_to_record(record)
    _normalize_session_repository_on_record(record)
    _normalize_session_tenant_on_record(record)
    _normalize_session_latency_state_on_record(record)
    _normalize_sensitive_config_change_state_on_record(record)
    _extract_inline_tool_definition_snapshot_from_metadata(record)
    _classify_zero_token_session_history_record(record)
    _sync_session_history_record_metadata(record)
    return record


def _normalize_agent_id_on_record(record: Dict[str, Any]) -> None:
    metadata = dict(record.get("metadata") or {}) if isinstance(record.get("metadata"), dict) else {}
    record["metadata"] = metadata
    disallowed_values = _agent_id_disallowed_values(
        record.get("session_id"),
        record.get("trace_id"),
        record.get("litellm_call_id"),
        record.get("agent_name"),
        record.get("tenant_id"),
        record.get("repository"),
        metadata.get("session_id"),
        metadata.get("trace_id"),
        metadata.get("trace_user_id"),
        metadata.get("agent_name"),
        metadata.get("tenant_id"),
        metadata.get("repository"),
    )
    agent_id = _normalize_agent_id_identity(
        _first_non_empty_string(record.get("agent_id"), metadata.get("agent_id")),
        disallowed_values=disallowed_values,
    )
    record["agent_id"] = agent_id
    if agent_id:
        metadata["agent_id"] = agent_id
    else:
        metadata.pop("agent_id", None)
        metadata.pop("agent_id_source", None)


def _normalize_inbound_model_alias_on_record(record: Dict[str, Any]) -> None:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    record["inbound_model_alias"] = _first_non_empty_string(
        record.get("inbound_model_alias"),
        metadata.get("model_alias_label"),
        metadata.get("requested_model_alias"),
        metadata.get("codex_auto_agent_alias"),
        metadata.get("anthropic_auto_agent_alias"),
        metadata.get("aawm_auto_agent_alias"),
        record.get("model"),
    )


def _extract_inline_tool_definition_snapshot_from_metadata(
    record: Dict[str, Any],
) -> None:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        return

    snapshot = metadata.pop(_AAWM_TOOL_DEFINITION_SNAPSHOT_METADATA_KEY, None)
    if isinstance(snapshot, list) and snapshot:
        record.setdefault(_AAWM_TOOL_DEFINITION_SNAPSHOT_METADATA_KEY, snapshot)
    record["metadata"] = metadata


def _normalize_reporting_exclusion_state_on_record(record: Dict[str, Any]) -> None:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    else:
        metadata = dict(metadata)

    call_type = str(record.get("call_type") or "").strip().lower()
    model = str(record.get("model") or "").strip().lower()
    source = str(metadata.get("source") or "").strip().lower()

    if call_type == "codex_transcript" or source == "codex_transcript":
        metadata["session_history_usage_record"] = False
        metadata["session_history_reporting_excluded"] = True
        metadata["session_history_reporting_exclusion_reason"] = "synthetic_codex_transcript"

    if model == "unknown":
        metadata["session_history_model_unresolved"] = True
        metadata["session_history_model_reporting_excluded"] = True
        metadata.setdefault(
            "session_history_model_unresolved_reason",
            "missing_source_model_evidence",
        )

    record["metadata"] = metadata



_HOST_FUNCTION_NAMES = (
    "_normalize_reasoning_state",
    "_row_usage_object_from_record",
    "_normalize_provider_cache_state_on_record",
    "_normalize_session_runtime_identity_on_record",
    "_is_harness_tenant_identity",
    "_normalize_request_header_tenant_repository",
    "_normalize_repository_trust_source",
    "_repository_source_has_codex_memory_workflow",
    "_is_repository_source_trusted_common",
    "_is_repository_source_trusted_for_tenant",
    "_is_codex_trace_user_tenant_source",
    "_is_codex_passthrough_tenant_extraction_context",
    "_is_repository_source_trusted_for_codex_tenant",
    "_is_codex_session_history_record",
    "_is_claude_session_history_record",
    "_is_claude_project_repository_source",
    "_is_claude_metadata_tenant_source",
    "_claude_project_identity_is_trusted",
    "_codex_repository_source_trusted_for_record",
    "_clear_untrusted_codex_trace_user_tenant_on_record",
    "_mark_codex_trace_user_tenant_skipped",
    "_codex_untrusted_repository_reason",
    "_mark_repository_unresolved_metadata",
    "_session_history_missing_repository_reason",
    "_mark_missing_repository_unresolved",
    "_clear_untrusted_claude_project_repository_on_record",
    "_clear_untrusted_claude_metadata_tenant_on_record",
    "_clear_repository_unresolved_metadata",
    "_mark_codex_repository_tenant_skipped",
    "_clear_codex_trace_user_tenant_source_on_record",
    "_clear_untrusted_codex_tenant_on_record",
    "_codex_tenant_source_trusted_for_record",
    "_clear_untrusted_codex_repository_tenant_on_record",
    "_normalize_session_repository_on_record",
    "_can_promote_known_codex_repository_to_tenant",
    "_normalize_session_tenant_on_record",
    "_sync_session_history_record_metadata",
    "_normalize_prompt_overhead_state_on_record",
    "_normalize_invalid_tool_call_state_on_record",
    "_normalize_structured_output_state_on_record",
    "_normalize_compact_summary_state_on_record",
    "_optional_metadata_bool",
    "_normalize_agent_score_reasons",
    "_append_agent_quality_text",
    "_append_agent_quality_command_from_arguments",
    "_append_agent_quality_commands_from_message",
    "_collect_agent_quality_context_from_request_body",
    "_collect_agent_quality_response_texts",
    "_agent_quality_commands_from_tool_activity",
    "_apply_runtime_agent_quality_scores",
    "_normalize_agent_score_state_on_record",
    "_normalize_session_latency_state_on_record",
    "_extract_gemini_control_plane_method_from_record",
    "_session_history_record_provider_usage_token_total",
    "_classify_zero_token_session_history_record",
    "_normalize_session_history_record",
    "_normalize_agent_id_on_record",
    "_normalize_inbound_model_alias_on_record",
    "_extract_inline_tool_definition_snapshot_from_metadata",
    "_normalize_reporting_exclusion_state_on_record",
)


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


def _rebind_installable_callable(value, host_globals):
    if isinstance(value, _FunctionType):
        return _rebind_to_host_globals(value, host_globals)
    return value


def install(host_globals):
    """Publish this module's helpers onto the identity host namespace.

    Plain functions are rebound so their ``__globals__`` is the identity
    package dict (record.py contract) -- free-name lookups then resolve
    through the identity namespace and monkeypatches on it stay effective.
    """
    mod = globals()
    for _name in _HOST_FUNCTION_NAMES:
        _original = mod[_name]
        _installed = _rebind_installable_callable(_original, host_globals)
        mod[_name] = _installed
        host_globals[_name] = _installed
