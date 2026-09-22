"""OpenCode Zen provider runtime extracted from the passthrough god module.

Host-owned utilities are supplied through :func:`configure_runtime`.  The
shared integrator can use :func:`install` to publish the same function objects
into the host module while retaining live monkeypatch lookups.
"""

from __future__ import annotations

import hashlib
import asyncio
import errno
import json
import os
import stat
from collections import OrderedDict
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Never, NoReturn, Optional

import httpx
from starlette.requests import Request
from starlette.responses import StreamingResponse

from litellm.llms.anthropic.experimental_pass_through.providers.opencode_zen import (
    constants as _constants,
)
from litellm.llms.anthropic.experimental_pass_through.providers.opencode_zen import (
    normalization as _normalization,
)
from litellm.proxy.pass_through_endpoints.providers.common import (
    _raise_opencode_zen_auto_agent_candidate_unavailable as _common_raise_opencode_zen_unavailable,
)


from ...aawm_alias_routing.failure_vocabulary import ZenFailure

Payload = dict[str, Any]


@dataclass(frozen=True)
class Runtime:
    """Callbacks owned by the passthrough host."""

    get_secret_str: Callable[[str], Optional[str]]
    assemble_headers: Callable[..., dict[str, str]]
    normalize_endpoint_for_target: Callable[[str, str], str]
    join_url_paths: Callable[[httpx.URL, str, str], str]
    extract_exception_status_code: Callable[[Exception], Optional[int]]
    extract_exception_detail: Callable[[Exception], Any]
    merge_metadata: Callable[..., Payload]
    add_route_family_logging_metadata: Callable[[Payload, str], Payload]
    build_langfuse_span_descriptor: Callable[..., Payload]
    normalization_runtime_factory: Callable[[], _normalization.Runtime]
    is_openai_responses_endpoint: Callable[[str], bool]
    has_anthropic_responses_adapter_endpoint: Callable[[str], bool]
    get_anthropic_adapter_model_candidates: Callable[[Payload], list[str]]
    load_local_api_key: Optional[Callable[[], Awaitable[str]]] = None
    raise_candidate_unavailable: Optional[Callable[[Exception], Any]] = None
    load_candidate_api_key: Optional[Callable[..., Awaitable[str]]] = None


_runtime: Optional[Runtime] = None

# Zen file credentials only. OpenCode Go keeps its own uncached read so the
# two families cannot reuse each other's keys or invalidation state.
_ZEN_AUTH_MAX_BYTES = 1_048_576
_ZEN_AUTH_READ_ATTEMPTS = 3
_ZEN_AUTH_CACHE_MAX_ENTRIES = 8
_ZenAuthGeneration = tuple[int, int, int, int, int]


@dataclass(frozen=True)
class _ZenAuthCacheEntry:
    """One Zen file generation. The key stays out of repr."""

    generation: _ZenAuthGeneration
    api_key: str = field(repr=False, compare=False)


@dataclass(frozen=True)
class _ZenAuthFlight:
    generation: _ZenAuthGeneration
    task: "asyncio.Task[str]"


_zen_auth_cache: "OrderedDict[str, _ZenAuthCacheEntry]" = OrderedDict()
_zen_auth_flights: dict[str, _ZenAuthFlight] = {}
_zen_auth_lock: Optional[asyncio.Lock] = None
_zen_auth_lock_loop_id: Optional[int] = None


_HOST_FUNCTION_NAMES = (
    "_get_opencode_zen_target_base",
    "_get_opencode_go_target_base",
    "_get_opencode_zen_auth_file_path",
    "_load_local_opencode_zen_api_key",
    "_load_opencode_go_api_key",
    "_load_opencode_zen_api_key_for_candidate",
    "_build_opencode_zen_headers",
    "_add_opencode_zen_logging_metadata",
    "_get_anthropic_opencode_zen_normalization_runtime",
    "_get_opencode_zen_responses_tool_name",
    "_ordered_unique_str_values",
    "_strip_opencode_zen_unsupported_responses_tools",
    "_opencode_zen_chat_message_role",
    "_opencode_zen_chat_tool_call_id",
    "_opencode_zen_chat_message_tool_call_ids",
    "_opencode_zen_chat_message_tool_result_id",
    "_collect_opencode_zen_following_tool_block",
    "_sanitize_opencode_zen_completion_messages_for_chat_completion",
    "_opencode_zen_responses_sse_event",
    "_opencode_zen_response_payload_for_stream",
    "_opencode_zen_message_item_for_stream",
    "_opencode_zen_completed_response_for_stream",
    "_normalize_opencode_zen_responses_stream_for_codex",
    "_build_codex_opencode_zen_streaming_response",
    "_join_opencode_zen_passthrough_url",
)


def configure_runtime(runtime: Runtime) -> None:
    """Configure callbacks without importing the passthrough host module."""

    global _runtime
    _runtime = runtime
    _get_anthropic_opencode_zen_normalization_runtime.cache_clear()


def _require_runtime() -> Runtime:
    if _runtime is None:
        raise RuntimeError("OpenCode Zen runtime callbacks have not been configured")
    return _runtime


def install(host_globals: dict[str, Any]) -> None:
    """Configure live host lookups and publish same-object facades."""

    def _host(name: str) -> Any:
        return host_globals[name]

    def _normalization_runtime_factory() -> _normalization.Runtime:
        from litellm.responses.litellm_completion_transformation.transformation import (
            LiteLLMCompletionResponsesConfig,
        )

        return _normalization.Runtime(
            clean_secret_string=lambda value: _host("_clean_secret_string")(
                value if isinstance(value, str) else None
            ),
            merge_metadata=lambda *args, **kwargs: _host("_merge_litellm_metadata")(
                *args, **kwargs
            ),
            add_logging_metadata=lambda *args, **kwargs: _host(
                "_add_opencode_zen_logging_metadata"
            )(*args, **kwargs),
            build_span=lambda *args, **kwargs: _host("_build_langfuse_span_descriptor")(
                *args, **kwargs
            ),
            transform_responses_api_request_to_chat_completion_request=(
                LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request
            ),
            async_responses_api_session_handler=(
                LiteLLMCompletionResponsesConfig.async_responses_api_session_handler
            ),
            iterate_responses_sse_events=lambda iterator: _host(
                "_iterate_responses_sse_events"
            )(iterator),
            coerce_namespace_to_mapping=lambda value: _host(
                "_coerce_namespace_to_mapping"
            )(value),
            responses_output_item_has_meaningful_content=lambda item: _host(
                "_responses_output_item_has_meaningful_content"
            )(item),
            streaming_response_factory=_host("StreamingResponse"),
        )

    configure_runtime(
        Runtime(
            get_secret_str=lambda name: _host("get_secret_str")(name),
            assemble_headers=lambda **kwargs: _host(
                "BaseOpenAIPassThroughHandler"
            )._assemble_headers(**kwargs),
            normalize_endpoint_for_target=lambda endpoint, base_target_url: _host(
                "BaseOpenAIPassThroughHandler"
            )._normalize_endpoint_for_target(
                endpoint=endpoint,
                base_target_url=base_target_url,
            ),
            join_url_paths=lambda base_url, path, provider: _host(
                "BaseOpenAIPassThroughHandler"
            )._join_url_paths(base_url, path, provider),
            extract_exception_status_code=lambda exc: _host(
                "_extract_adapter_exception_status_code"
            )(exc),
            extract_exception_detail=lambda exc: _host(
                "_extract_adapter_exception_detail"
            )(exc),
            merge_metadata=lambda *args, **kwargs: _host("_merge_litellm_metadata")(
                *args, **kwargs
            ),
            add_route_family_logging_metadata=lambda body, route_family: _host(
                "_add_route_family_logging_metadata"
            )(body, route_family),
            build_langfuse_span_descriptor=lambda *args, **kwargs: _host(
                "_build_langfuse_span_descriptor"
            )(*args, **kwargs),
            normalization_runtime_factory=_normalization_runtime_factory,
            is_openai_responses_endpoint=lambda endpoint: _host(
                "_is_openai_responses_endpoint"
            )(endpoint),
            has_anthropic_responses_adapter_endpoint=lambda endpoint: _host(
                "_has_anthropic_responses_adapter_endpoint"
            )(endpoint),
            get_anthropic_adapter_model_candidates=lambda body: _host(
                "_get_anthropic_adapter_model_candidates"
            )(body),
            load_local_api_key=lambda: _host("_load_local_opencode_zen_api_key")(),
            raise_candidate_unavailable=lambda exc: _host(
                "_raise_opencode_zen_auto_agent_candidate_unavailable"
            )(exc),
            load_candidate_api_key=lambda **kwargs: _host(
                "_load_opencode_zen_api_key_for_candidate"
            )(**kwargs),
        )
    )
    for name in _HOST_FUNCTION_NAMES:
        host_globals[name] = globals()[name]


def _clean_secret_string(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'"}:
        cleaned = cleaned[1:-1].strip()
    return cleaned or None


def _get_first_secret_value(secret_names: tuple[str, ...]) -> Optional[str]:
    runtime = _require_runtime()
    for secret_name in secret_names:
        value = _clean_secret_string(runtime.get_secret_str(secret_name))
        if value:
            return value
    return None


def _get_opencode_zen_target_base() -> str:
    runtime = _require_runtime()
    cleaned = (
        _clean_secret_string(runtime.get_secret_str("OPENCODE_ZEN_API_BASE"))
        or _clean_secret_string(runtime.get_secret_str("AAWM_OPENCODE_ZEN_API_BASE"))
        or _clean_secret_string(os.getenv("OPENCODE_ZEN_API_BASE"))
        or _clean_secret_string(os.getenv("AAWM_OPENCODE_ZEN_API_BASE"))
        or _constants._OPENCODE_ZEN_DEFAULT_BASE_URL
    ).rstrip("/")
    if cleaned.endswith("/v1"):
        return cleaned[: -len("/v1")]
    return cleaned


def _get_opencode_go_target_base() -> str:
    runtime = _require_runtime()
    cleaned = (
        _clean_secret_string(runtime.get_secret_str("OPENCODE_GO_API_BASE"))
        or _clean_secret_string(runtime.get_secret_str("AAWM_OPENCODE_GO_API_BASE"))
        or _clean_secret_string(os.getenv("OPENCODE_GO_API_BASE"))
        or _clean_secret_string(os.getenv("AAWM_OPENCODE_GO_API_BASE"))
        or _constants._OPENCODE_GO_DEFAULT_BASE_URL
    ).rstrip("/")
    if cleaned.endswith("/v1"):
        return cleaned[: -len("/v1")]
    return cleaned


def _get_opencode_zen_auth_file_path() -> Optional[Path]:
    """Resolve the OpenCode Zen auth file path.

    The first nonempty auth-file environment variable is authoritative.
    If it is set but the path is missing or not a regular file, raise
    immediately without consulting later variables or HOME defaults.
    HOME-relative defaults are used only when no auth-file variable is
    configured.
    """
    for env_name in _constants._OPENCODE_ZEN_AUTH_FILE_ENV_VARS:
        value = _clean_secret_string(os.getenv(env_name))
        if value:
            candidate = Path(value).expanduser()
            if not candidate.is_file():
                raise ValueError(
                    f"OpenCode Zen auth file configured via {env_name} "
                    "is missing or not a regular file."
                )
            return candidate

    for candidate_str in _constants._OPENCODE_ZEN_DEFAULT_AUTH_PATHS:
        candidate = Path(candidate_str).expanduser()
        if candidate.is_file():
            return candidate
    return None


def _select_opencode_zen_provider_auth(auth_data: Any) -> Any:
    """OC-010: Zen selects only the ``opencode`` auth entry.

    Never fall back to ``opencode-go``. Presence of a Go entry is used only
    to choose a deterministic Zen-ineligible error; the Go secret is not
    read, copied, or returned.
    """
    if not isinstance(auth_data, dict):
        return None
    return auth_data.get("opencode")


def _select_opencode_go_provider_auth(auth_data: Any) -> Any:
    """OC-019: Go selects only the ``opencode-go`` auth entry.

    Never fall back to ``opencode``. A missing entry is returned as None
    so the caller fails closed. The Zen/general entry is not read, copied,
    or returned.
    """
    if not isinstance(auth_data, dict) or "opencode-go" not in auth_data:
        return None
    return auth_data.get("opencode-go")


def _get_opencode_go_auth_file_path() -> Optional[Path]:
    """Resolve the auth file used for OpenCode Go entry selection.

    A configured Go auth-file variable is authoritative. Otherwise the
    shared OpenCode auth file may be read, and only its ``opencode-go``
    entry is eligible.
    """
    for env_name in _constants._OPENCODE_GO_AUTH_FILE_ENV_VARS:
        value = _clean_secret_string(os.getenv(env_name))
        if value:
            candidate = Path(value).expanduser()
            if not candidate.is_file():
                raise ValueError(
                    f"OpenCode Go auth file configured via {env_name} "
                    "is missing or not a regular file."
                )
            return candidate
    try:
        return _get_opencode_zen_auth_file_path()
    except ValueError as exc:
        message = str(exc)
        zen_prefix = "OpenCode Zen auth file"
        if message.startswith(zen_prefix):
            message = "OpenCode Go auth file" + message[len(zen_prefix) :]
        raise ValueError(message) from None


def _opencode_go_credential_fingerprint(api_key: str) -> str:
    """Non-reversible id for the selected Go credential.

    The cache namespace and target family are part of the material, so the
    same secret text under the Zen family would not share this fingerprint.
    The raw key is not returned.
    """
    namespace = _constants._OPENCODE_GO_CREDENTIAL_CACHE_NAMESPACE
    target_family = _constants._OPENCODE_GO_TARGET_FAMILY
    return hashlib.sha256(
        f"{namespace}\0{target_family}\0{api_key}".encode("utf-8")
    ).hexdigest()


def _bind_opencode_go_credential_identity(api_key: str) -> str:
    """Return the Go key only when its identity stays on the Go family."""
    fingerprint = _opencode_go_credential_fingerprint(api_key)
    if (
        _constants._OPENCODE_GO_CREDENTIAL_CACHE_NAMESPACE
        != _constants._OPENCODE_GO_CREDENTIAL_FAMILY
        or _constants._OPENCODE_GO_TARGET_FAMILY
        != _constants._OPENCODE_GO_CREDENTIAL_FAMILY
        or len(fingerprint) != 64
    ):
        raise ValueError(
            "OpenCode Go credential fingerprint, target family, and "
            "cache namespace must stay on the Go credential family."
        )
    return api_key


def _raise_invalid_opencode_go_auth_file(
    *,
    source_label: str,
    auth_data: Any = None,
) -> NoReturn:
    has_go_entry = isinstance(auth_data, dict) and "opencode-go" in auth_data
    has_general_entry = isinstance(auth_data, dict) and "opencode" in auth_data
    if has_general_entry and not has_go_entry:
        raise ValueError(
            f"OpenCode Go auth file configured via {source_label} "
            "contains only an OpenCode ('opencode') credential, "
            "which is not valid for OpenCode Go. OpenCode Go "
            "requires provider 'opencode-go' with API-key auth."
        )
    raise ValueError(
        f"OpenCode Go auth file configured via {source_label} "
        "must contain provider 'opencode-go' with API-key auth."
    )


def _raise_invalid_opencode_auth_file(
    *,
    source_label: str,
    auth_data: Any,
    source_family: str,
) -> NoReturn:
    normalized_family = str(source_family or "").strip().casefold()
    if normalized_family == _constants._OPENCODE_GO_CREDENTIAL_FAMILY:
        _raise_invalid_opencode_go_auth_file(
            source_label=source_label,
            auth_data=auth_data,
        )
    zen_entry_present = isinstance(auth_data, dict) and "opencode" in auth_data
    has_go_entry = isinstance(auth_data, dict) and "opencode-go" in auth_data
    if has_go_entry and not zen_entry_present:
        raise ValueError(
            f"OpenCode Zen auth file configured via {source_label} "
            "contains only an OpenCode Go ('opencode-go') credential, "
            "which is not valid for OpenCode Zen. OpenCode Zen "
            "requires provider 'opencode' with API-key auth."
        )
    raise ValueError(
        f"OpenCode Zen auth file configured via {source_label} "
        "must contain provider 'opencode' with API-key auth; "
        "OpenCode Go ('opencode-go') credentials are not valid "
        "for OpenCode Zen."
    )


def _configured_opencode_auth_source_label() -> str:
    """Name the first configured auth-file variable, never its path."""

    for env_name in _constants._OPENCODE_ZEN_AUTH_FILE_ENV_VARS:
        if _clean_secret_string(os.getenv(env_name)):
            return env_name
    return "default"


def _missing_opencode_auth_file_error() -> FileNotFoundError:
    return FileNotFoundError(
        "OpenCode Zen auth file not found. Expected "
        "'~/.local/share/opencode/auth.json' or set "
        "'LITELLM_OPENCODE_AUTH_FILE'."
    )


def _require_opencode_auth_path() -> Path:
    auth_path = _get_opencode_zen_auth_file_path()
    if auth_path is None:
        raise _missing_opencode_auth_file_error()
    return auth_path


def _zen_auth_file_error(source_label: str, reason: str) -> ValueError:
    return ValueError(f"OpenCode Zen auth file configured via {source_label} {reason}")


def _zen_auth_cache_key(path: Path) -> str:
    """Cache by the authoritative path without following a symlink."""

    return os.path.normcase(os.path.abspath(os.fspath(path)))


def _zen_auth_generation(stat_result: os.stat_result) -> _ZenAuthGeneration:
    return (
        int(stat_result.st_dev),
        int(stat_result.st_ino),
        int(stat_result.st_mtime_ns),
        int(stat_result.st_ctime_ns),
        int(stat_result.st_size),
    )


def _zen_auth_read_fingerprint(
    stat_result: os.stat_result,
) -> tuple[int, int, int, int]:
    # A replaced inode can change ctime/link metadata while an open
    # descriptor still exposes the previous bytes.
    return (
        int(stat_result.st_dev),
        int(stat_result.st_ino),
        int(stat_result.st_mtime_ns),
        int(stat_result.st_size),
    )


def _open_zen_auth_descriptor(path: Path) -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise OSError(errno.EINVAL, "secure open is unavailable")
    flags = os.O_RDONLY | nofollow
    cloexec = getattr(os, "O_CLOEXEC", None)
    if cloexec is not None:
        flags |= cloexec
    return os.open(os.fspath(path), flags)


def _map_zen_auth_open_error(exc: OSError, source_label: str) -> ValueError:
    if exc.errno in (errno.ELOOP, errno.EMLINK):
        return _zen_auth_file_error(
            source_label,
            "is missing or not a regular file.",
        )
    return _zen_auth_file_error(source_label, "is not readable.")


def _read_descriptor_bounded(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    limit = _ZEN_AUTH_MAX_BYTES
    while total <= limit:
        try:
            chunk = os.read(descriptor, min(8192, limit + 1 - total))
        except OSError:
            raise OSError(errno.EIO, "auth file read failed") from None
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    if total > limit:
        raise ValueError("auth file exceeds the maximum size")
    return b"".join(chunks)


def _zen_auth_text_differs(path_text: str, descriptor_text: str) -> bool:
    if path_text == descriptor_text:
        return False
    normalized_path = path_text.replace("\r\n", "\n").replace("\r", "\n")
    normalized_descriptor = descriptor_text.replace("\r\n", "\n").replace("\r", "\n")
    return normalized_path != normalized_descriptor


def _api_key_from_opencode_auth_text(
    raw_text: str,
    *,
    source_label: str,
    source_family: str,
) -> str:
    try:
        auth_data = json.loads(raw_text)
    except Exception:
        raise ValueError(
            f"OpenCode Zen auth file configured via {source_label} "
            "does not contain valid JSON."
        ) from None

    normalized_family = str(source_family or "").strip().casefold()
    if normalized_family == _constants._OPENCODE_GO_CREDENTIAL_FAMILY:
        provider_auth = _select_opencode_go_provider_auth(auth_data)
    else:
        provider_auth = _select_opencode_zen_provider_auth(auth_data)
    if not isinstance(provider_auth, dict):
        _raise_invalid_opencode_auth_file(
            source_label=source_label,
            auth_data=auth_data,
            source_family=normalized_family,
        )
    api_key = _clean_secret_string(provider_auth.get("key"))
    auth_type = _clean_secret_string(provider_auth.get("type"))
    if auth_type not in {None, "api"}:
        raise ValueError(
            f"OpenCode Zen auth file configured via {source_label} "
            "must contain provider 'opencode' with API-key auth type."
        )
    if api_key is None:
        _raise_invalid_opencode_auth_file(
            source_label=source_label,
            auth_data=auth_data,
            source_family=normalized_family,
        )
    assert api_key is not None
    return api_key


def _stat_zen_auth_generation_sync(
    path: Path,
    source_label: str,
) -> _ZenAuthGeneration:
    try:
        descriptor = _open_zen_auth_descriptor(path)
    except OSError as exc:
        raise _map_zen_auth_open_error(exc, source_label) from None
    try:
        try:
            metadata = os.fstat(descriptor)
        except OSError:
            raise _zen_auth_file_error(
                source_label,
                "is not readable.",
            ) from None
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
    if not stat.S_ISREG(metadata.st_mode):
        raise _zen_auth_file_error(
            source_label,
            "is missing or not a regular file.",
        )
    if metadata.st_size > _ZEN_AUTH_MAX_BYTES:
        raise _zen_auth_file_error(
            source_label,
            "exceeds the maximum auth-file size.",
        )
    return _zen_auth_generation(metadata)


def _read_one_zen_auth_attempt_sync(
    path: Path,
    source_label: str,
) -> Optional[tuple[str, _ZenAuthGeneration]]:
    """Read one generation. Return None when the file changes mid-read."""

    try:
        descriptor = _open_zen_auth_descriptor(path)
    except OSError as exc:
        raise _map_zen_auth_open_error(exc, source_label) from None
    try:
        try:
            before = os.fstat(descriptor)
        except OSError:
            raise _zen_auth_file_error(
                source_label,
                "is not readable.",
            ) from None
        if not stat.S_ISREG(before.st_mode):
            raise _zen_auth_file_error(
                source_label,
                "is missing or not a regular file.",
            )
        if before.st_size > _ZEN_AUTH_MAX_BYTES:
            raise _zen_auth_file_error(
                source_label,
                "exceeds the maximum auth-file size.",
            )
        try:
            # Path.read_text stays on this path so an unreadable replacement
            # fails closed with the existing sanitized error.
            path_text = path.read_text(encoding="utf-8")
        except Exception:
            raise _zen_auth_file_error(
                source_label,
                "is not readable.",
            ) from None
        try:
            descriptor_text = _read_descriptor_bounded(descriptor).decode("utf-8")
        except (OSError, UnicodeDecodeError):
            raise _zen_auth_file_error(
                source_label,
                "is not readable.",
            ) from None
        except ValueError:
            raise _zen_auth_file_error(
                source_label,
                "exceeds the maximum auth-file size.",
            ) from None
        try:
            after = os.fstat(descriptor)
        except OSError:
            raise _zen_auth_file_error(
                source_label,
                "is not readable.",
            ) from None
        if _zen_auth_read_fingerprint(before) != _zen_auth_read_fingerprint(
            after
        ) or _zen_auth_text_differs(path_text, descriptor_text):
            return None
        api_key = _api_key_from_opencode_auth_text(
            descriptor_text,
            source_label=source_label,
            source_family=_constants._OPENCODE_ZEN_CREDENTIAL_FAMILY,
        )
        return api_key, _zen_auth_generation(after)
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass


def _read_stable_zen_auth_api_key_sync(
    path: Path,
    source_label: str,
) -> tuple[str, _ZenAuthGeneration]:
    for _attempt in range(_ZEN_AUTH_READ_ATTEMPTS):
        loaded = _read_one_zen_auth_attempt_sync(path, source_label)
        if loaded is not None:
            return loaded
    raise _zen_auth_file_error(
        source_label,
        "changed while it was read.",
    )


def _resolve_zen_auth_load_inputs_sync() -> tuple[Path, str]:
    return (
        _require_opencode_auth_path(),
        _configured_opencode_auth_source_label(),
    )


def _get_zen_auth_lock() -> asyncio.Lock:
    global _zen_auth_lock, _zen_auth_lock_loop_id

    loop_id = id(asyncio.get_running_loop())
    if _zen_auth_lock is None or _zen_auth_lock_loop_id != loop_id:
        _zen_auth_lock = asyncio.Lock()
        _zen_auth_lock_loop_id = loop_id
        _zen_auth_flights.clear()
    return _zen_auth_lock


def _remember_zen_auth_unlocked(
    path_key: str,
    generation: _ZenAuthGeneration,
    api_key: str,
) -> None:
    _zen_auth_cache[path_key] = _ZenAuthCacheEntry(
        generation=generation,
        api_key=api_key,
    )
    _zen_auth_cache.move_to_end(path_key)
    while len(_zen_auth_cache) > _ZEN_AUTH_CACHE_MAX_ENTRIES:
        _zen_auth_cache.popitem(last=False)


def _finish_zen_auth_flight(path_key: str, task: "asyncio.Task[str]") -> None:
    flight = _zen_auth_flights.get(path_key)
    if flight is not None and flight.task is task:
        _zen_auth_flights.pop(path_key, None)
    if task.cancelled():
        return
    task.exception()


async def _drop_zen_auth_cache_if_owner(path_key: str) -> None:
    task = asyncio.current_task()
    async with _get_zen_auth_lock():
        flight = _zen_auth_flights.get(path_key)
        if flight is not None and flight.task is task:
            _zen_auth_cache.pop(path_key, None)


async def _publish_zen_auth_if_current(
    path: Path,
    source_label: str,
    path_key: str,
    generation: _ZenAuthGeneration,
    api_key: str,
) -> None:
    task = asyncio.current_task()
    async with _get_zen_auth_lock():
        flight = _zen_auth_flights.get(path_key)
        if flight is None or flight.task is not task:
            return
        try:
            current = await asyncio.to_thread(
                _stat_zen_auth_generation_sync,
                path,
                source_label,
            )
        except Exception:
            _zen_auth_cache.pop(path_key, None)
            return
        if current != generation:
            _zen_auth_cache.pop(path_key, None)
            return
        _remember_zen_auth_unlocked(path_key, generation, api_key)


async def _run_zen_auth_flight(
    path: Path,
    source_label: str,
    path_key: str,
) -> str:
    try:
        api_key, generation = await asyncio.to_thread(
            _read_stable_zen_auth_api_key_sync,
            path,
            source_label,
        )
    except BaseException:
        await _drop_zen_auth_cache_if_owner(path_key)
        raise
    await _publish_zen_auth_if_current(
        path,
        source_label,
        path_key,
        generation,
        api_key,
    )
    return api_key


async def _load_cached_zen_file_api_key() -> str:
    path, source_label = await asyncio.to_thread(_resolve_zen_auth_load_inputs_sync)
    path_key = _zen_auth_cache_key(path)
    async with _get_zen_auth_lock():
        try:
            generation = await asyncio.to_thread(
                _stat_zen_auth_generation_sync,
                path,
                source_label,
            )
        except Exception:
            _zen_auth_cache.pop(path_key, None)
            raise
        cached = _zen_auth_cache.get(path_key)
        if cached is not None and cached.generation == generation:
            _zen_auth_cache.move_to_end(path_key)
            return cached.api_key
        _zen_auth_cache.pop(path_key, None)
        flight = _zen_auth_flights.get(path_key)
        if flight is None or flight.generation != generation or flight.task.done():
            task = asyncio.create_task(
                _run_zen_auth_flight(path, source_label, path_key)
            )
            flight = _ZenAuthFlight(generation=generation, task=task)
            _zen_auth_flights[path_key] = flight
            task.add_done_callback(
                lambda done, key=path_key: _finish_zen_auth_flight(key, done)
            )
        task = flight.task
    return await asyncio.shield(task)


async def _load_local_opencode_auth_api_key(*, source_family: str) -> str:
    normalized_family = str(source_family or "").strip().casefold()
    if normalized_family == _constants._OPENCODE_GO_CREDENTIAL_FAMILY:
        return await _load_opencode_go_api_key()

    explicit_key = _get_first_secret_value(_constants._OPENCODE_ZEN_API_KEY_ENV_VARS)
    if explicit_key is not None:
        return explicit_key
    return await _load_cached_zen_file_api_key()


async def _load_local_opencode_zen_api_key() -> str:
    return await _load_local_opencode_auth_api_key(
        source_family=_constants._OPENCODE_ZEN_CREDENTIAL_FAMILY,
    )


async def _load_opencode_go_api_key() -> str:
    """Load an OpenCode Go API key from Go-specific sources only.

    Explicit keys come from Go environment names. A shared auth file may
    be read, but only ``auth_data["opencode-go"]`` is selected. The
    Zen/general ``auth_data["opencode"]`` entry is never read, copied, or
    removed. A missing or malformed Go entry fails closed. Credential
    fingerprint, target family, and cache namespace stay on ``opencode_go``.
    """
    explicit_key = _get_first_secret_value(
        _constants._OPENCODE_GO_API_KEY_ENV_VARS
    )
    if explicit_key is not None:
        return _bind_opencode_go_credential_identity(explicit_key)

    configured_source: Optional[str] = None
    for env_name in (
        *_constants._OPENCODE_GO_AUTH_FILE_ENV_VARS,
        *_constants._OPENCODE_ZEN_AUTH_FILE_ENV_VARS,
    ):
        if _clean_secret_string(os.getenv(env_name)):
            configured_source = env_name
            break

    auth_path = _get_opencode_go_auth_file_path()
    if auth_path is None:
        raise FileNotFoundError(
            "OpenCode Go auth file not found. Expected "
            "'~/.local/share/opencode/auth.json' or set "
            "'LITELLM_OPENCODE_GO_AUTH_FILE'."
        )

    source_label = configured_source or "default"

    try:
        raw_text = auth_path.read_text(encoding="utf-8")
    except Exception:
        raise ValueError(
            f"OpenCode Go auth file configured via {source_label} "
            "is not readable."
        ) from None

    try:
        auth_data = json.loads(raw_text)
    except Exception:
        raise ValueError(
            f"OpenCode Go auth file configured via {source_label} "
            "does not contain valid JSON."
        ) from None

    provider_auth = _select_opencode_go_provider_auth(auth_data)
    if not isinstance(provider_auth, dict):
        _raise_invalid_opencode_go_auth_file(
            source_label=source_label,
            auth_data=auth_data,
        )
    raw_key = provider_auth.get("key")
    raw_type = provider_auth.get("type")
    if (raw_key is not None and not isinstance(raw_key, str)) or (
        raw_type is not None and not isinstance(raw_type, str)
    ):
        _raise_invalid_opencode_go_auth_file(
            source_label=source_label,
            auth_data=auth_data,
        )
    api_key = _clean_secret_string(raw_key if isinstance(raw_key, str) else None)
    auth_type = _clean_secret_string(
        raw_type if isinstance(raw_type, str) else None
    )
    if auth_type not in {None, "api"}:
        raise ValueError(
            f"OpenCode Go auth file configured via {source_label} "
            "must contain provider 'opencode-go' with API-key auth type."
        )
    if api_key is None:
        _raise_invalid_opencode_go_auth_file(
            source_label=source_label,
            auth_data=auth_data,
        )
    assert api_key is not None
    return _bind_opencode_go_credential_identity(api_key)


async def _load_local_opencode_go_api_key() -> str:
    return await _load_opencode_go_api_key()


async def _load_opencode_zen_api_key_for_candidate(
    *,
    use_alias_candidate_probe: bool = False,
    source_family: str = _constants._OPENCODE_ZEN_CREDENTIAL_FAMILY,
) -> str:
    runtime = _require_runtime()
    normalized_family = str(source_family or "").strip().casefold()
    try:
        if normalized_family == _constants._OPENCODE_GO_CREDENTIAL_FAMILY:
            return await _load_local_opencode_go_api_key()
        load_api_key = runtime.load_local_api_key
        if load_api_key is not None:
            return await load_api_key()
        return await _load_local_opencode_zen_api_key()
    except (FileNotFoundError, ValueError) as exc:
        if use_alias_candidate_probe:
            if runtime.raise_candidate_unavailable is not None:
                runtime.raise_candidate_unavailable(exc)
            _common_raise_opencode_zen_unavailable(exc)
        raise


async def _build_opencode_zen_headers(
    request: Request,
    *,
    use_alias_candidate_probe: bool = False,
) -> dict[str, str]:
    runtime = _require_runtime()
    load_candidate = runtime.load_candidate_api_key
    if load_candidate is not None:
        api_key = await load_candidate(
            use_alias_candidate_probe=use_alias_candidate_probe,
        )
    else:
        api_key = await _load_opencode_zen_api_key_for_candidate(
            use_alias_candidate_probe=use_alias_candidate_probe,
        )
    return runtime.assemble_headers(
        api_key=api_key,
        request=request,
    )


def _add_opencode_zen_logging_metadata(
    request_body: Payload,
    *,
    route_family: str,
    tag_prefix: str,
    requested_model: Any,
    adapter_model: Optional[str] = None,
    input_shape: Optional[str] = None,
    output_shape: Optional[str] = None,
    client_name: Optional[str] = None,
) -> Payload:
    runtime = _require_runtime()
    extra_fields: Payload = {
        "opencode_zen": True,
        "opencode_zen_requested_model": requested_model,
    }
    if client_name is not None:
        extra_fields["client_name"] = client_name
    if adapter_model is not None:
        extra_fields["opencode_zen_adapter_model"] = adapter_model
    if input_shape is not None:
        extra_fields["codex_adapter_input_shape"] = input_shape
    if output_shape is not None:
        extra_fields["codex_adapter_output_shape"] = output_shape

    tags = [tag_prefix, "opencode-zen"]
    if adapter_model is not None:
        tags.append(f"opencode-zen-model:{adapter_model}")

    return runtime.merge_metadata(
        runtime.add_route_family_logging_metadata(request_body, route_family),
        tags_to_add=tags,
        extra_fields=extra_fields,
    )


@lru_cache(maxsize=1)
def _get_anthropic_opencode_zen_normalization_runtime() -> _normalization.Runtime:
    return _require_runtime().normalization_runtime_factory()


def _get_opencode_zen_responses_tool_name(tool: Any) -> Optional[str]:
    return _normalization.get_responses_tool_name(
        _get_anthropic_opencode_zen_normalization_runtime(),
        tool,
    )


def _ordered_unique_str_values(
    values: list[Optional[str]],
) -> list[str]:
    unique_values: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value:
            continue
        if value not in unique_values:
            unique_values.append(value)
    return unique_values


def _strip_opencode_zen_unsupported_responses_tools(
    request_body: Payload,
) -> Payload:
    return _normalization.strip_unsupported_responses_tools(
        _get_anthropic_opencode_zen_normalization_runtime(),
        request_body,
    )


def _opencode_zen_chat_message_role(message: Any) -> Optional[str]:
    return _normalization.chat_message_role(message)


def _opencode_zen_chat_tool_call_id(tool_call: Any) -> Optional[str]:
    return _normalization.chat_tool_call_id(tool_call)


def _opencode_zen_chat_message_tool_call_ids(message: Any) -> list[str]:
    return _normalization.chat_message_tool_call_ids(message)


def _opencode_zen_chat_message_tool_result_id(
    message: Any,
) -> Optional[str]:
    return _normalization.chat_message_tool_result_id(message)


def _collect_opencode_zen_following_tool_block(
    messages: list[Any],
    start_index: int,
) -> tuple[list[Any], list[Optional[str]], int]:
    return _normalization.collect_following_tool_block(messages, start_index)


def _sanitize_opencode_zen_completion_messages_for_chat_completion(
    completion_kwargs: Payload,
) -> tuple[Payload, Payload]:
    return _normalization.sanitize_completion_messages_for_chat_completion(
        completion_kwargs
    )


def _opencode_zen_responses_sse_event(
    event_type: str,
    payload: Payload,
) -> str:
    return _normalization.responses_sse_event(event_type, payload)


def _opencode_zen_response_payload_for_stream(
    *,
    response_id: str,
    model: str,
    status: str,
    output: Optional[list[Payload]] = None,
    usage: Optional[Payload] = None,
) -> Payload:
    return _normalization.response_payload_for_stream(
        response_id=response_id,
        model=model,
        status=status,
        output=output,
        usage=usage,
    )


def _opencode_zen_message_item_for_stream(
    *,
    message_id: str,
    status: str,
    output_text: str = "",
) -> Payload:
    return _normalization.message_item_for_stream(
        message_id=message_id,
        status=status,
        output_text=output_text,
    )


def _opencode_zen_completed_response_for_stream(
    *,
    response_event: Payload,
    response_id: str,
    model: str,
    message_id: Optional[str],
    output_text: str,
) -> Payload:
    return _normalization.completed_response_for_stream(
        _get_anthropic_opencode_zen_normalization_runtime(),
        response_event=response_event,
        response_id=response_id,
        model=model,
        message_id=message_id,
        output_text=output_text,
    )


async def _normalize_opencode_zen_responses_stream_for_codex(
    response: Any,
    *,
    adapter_model: str,
) -> AsyncIterator[str]:
    async for chunk in _normalization.normalize_responses_stream_for_codex(
        _get_anthropic_opencode_zen_normalization_runtime(),
        response,
        adapter_model=adapter_model,
    ):
        yield chunk


def _build_codex_opencode_zen_streaming_response(
    response: Any,
    *,
    adapter_model: str,
) -> StreamingResponse:
    return _normalization.build_codex_streaming_response(
        _get_anthropic_opencode_zen_normalization_runtime(),
        response,
        adapter_model=adapter_model,
    )


def _join_opencode_zen_passthrough_url(
    base_target_url: str,
    endpoint: str,
) -> str:
    runtime = _require_runtime()
    normalized_endpoint = runtime.normalize_endpoint_for_target(
        endpoint,
        base_target_url,
    )
    return str(
        runtime.join_url_paths(
            httpx.URL(base_target_url),
            normalized_endpoint,
            _constants._OPENCODE_ZEN_PROVIDER,
        )
    )


def _extract_opencode_zen_failure(
    exc: Exception,
    *,
    use_alias_candidate_probe: bool = False,
    model: Optional[str] = None,
    route_family: Optional[str] = None,
) -> ZenFailure:
    """Use the same typed policy for direct HTTP/SSE and alias failures."""
    from ...aawm_alias_routing.error_signals import classify_opencode_zen_failure
    from ...aawm_alias_routing.policy import CODEX_AUTO_AGENT_OPENCODE_PROVIDER

    candidate = {
        "provider": CODEX_AUTO_AGENT_OPENCODE_PROVIDER,
        "route_family": route_family or "codex_opencode_zen_adapter",
        "model": model,
    }
    return classify_opencode_zen_failure(
        exc,
        candidate=candidate,
        route="alias" if use_alias_candidate_probe else "direct",
        attempted_provider_call=getattr(exc, "attempted_provider_call", True)
        is not False,
    )


def _raise_opencode_zen_failure(
    exc: Exception,
    *,
    use_alias_candidate_probe: bool = False,
    model: Optional[str] = None,
    route_family: Optional[str] = None,
) -> Never:
    """Translate without leaking provider bodies or replacing origin with 429."""
    from litellm.proxy._types import ProxyException

    failure = _extract_opencode_zen_failure(
        exc,
        use_alias_candidate_probe=use_alias_candidate_probe,
        model=model,
        route_family=route_family,
    )
    retry_after = failure.retry_after_seconds
    headers = {"Retry-After": str(retry_after)} if retry_after is not None else None
    mapped = ProxyException(
        message=failure.public_detail,
        type="rate_limit_error"
        if failure.public_status_code == 429
        else "upstream_error",
        param="model",
        code=failure.public_status_code,
        headers=headers,
    )
    mapped.status_code = failure.public_status_code
    mapped._aawm_zen_failure = failure
    mapped._aawm_provider_returned = failure.origin == "upstream"
    mapped.attempted_provider_call = (
        getattr(exc, "attempted_provider_call", True) is not False
    )
    mapped.detail = {
        "error": {
            "message": failure.public_detail,
            "code": failure.error_class or "provider_terminal_error",
        }
    }
    raise mapped from exc
