"""xAI / Grok OAuth credential helpers for LiteLLM request paths.

Production token accessors are **read-only**. In-process refresh is intentionally
unwired: the provider-status sidecar scripts (`scripts/xai_oauth_refresh.py`,
`scripts/grok_oidc_refresh.py`) own refresh and exclusive flock writes via the
shared ``litellm.secret_managers.credential_file_lock`` helper (RR-040 #1/#3).
This module keeps private-mode credential writes for Hermes migration only.
"""

from __future__ import annotations

import asyncio
import errno
import hashlib
import json
import os
import stat
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Tuple

import httpx  # noqa: F401  # harness patch surface; refresh path removed (RR-040)

from litellm.constants import XAI_API_BASE
from litellm.llms.xai import route_descriptors as _xai_route_descriptors
from litellm.llms.xai.route_descriptors import (
    get_grok_native_route_descriptor,
    get_oa_xai_route_descriptor,
    resolve_oa_xai_route_descriptor,
)
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.secret_managers.grok_oidc_auth_path import (
    resolve_grok_oidc_auth_path,
)
from litellm.secret_managers.main import get_secret_str
from litellm.secret_managers.xai_oauth_credentials import (
    DEFAULT_XAI_OAUTH_AUTH_FILE,
    DEFAULT_XAI_OAUTH_SCOPE,
    XaiOAuthCredentialResolution,
    credential_account_identity,
    credential_access_token,
    credential_expires_at,
    credential_identity,
    evaluate_xai_oauth_credential_lifecycle,
    resolve_xai_oauth_credentials,
    select_xai_oauth_credential_record,
)

OA_XAI_PROVIDER_PREFIX = _xai_route_descriptors.OA_XAI_PROVIDER_PREFIX
XAI_OAUTH_ROUTE_FAMILY = _xai_route_descriptors.XAI_OAUTH_ROUTE_FAMILY
XAI_OAUTH_CREDENTIAL_FAMILY = _xai_route_descriptors.XAI_OAUTH_CREDENTIAL_FAMILY
XAI_GROK_SUBSCRIPTION_QUOTA_FAMILY = "xai_grok_subscription"
GROK_NATIVE_OAUTH_ROUTE_FAMILY = (
    _xai_route_descriptors.GROK_NATIVE_OAUTH_ROUTE_FAMILY
)
GROK_NATIVE_OAUTH_CREDENTIAL_FAMILY = (
    _xai_route_descriptors.GROK_NATIVE_OAUTH_CREDENTIAL_FAMILY
)
GROK_NATIVE_OAUTH_CLIENT_NAME = "grok-build"

_DEFAULT_XAI_OAUTH_SCOPE = DEFAULT_XAI_OAUTH_SCOPE
_DEFAULT_XAI_OAUTH_TOKEN_ENDPOINT = "https://auth.x.ai/oauth2/token"
_DEFAULT_REFRESH_BUFFER_SECONDS = 300
_DEFAULT_HERMES_XAI_OAUTH_PROVIDER_ID = "xai-oauth"
_DEFAULT_HERMES_AUTH_PATH = "~/.hermes/auth.json"
_DEFAULT_LITELLM_XAI_OAUTH_AUTH_PATH = DEFAULT_XAI_OAUTH_AUTH_FILE

_XAI_RESPONSES_PREVIOUS_RESPONSE_ID_DECODED_METADATA = {
    "xai_responses_previous_response_id_decoded": True,
    "tags": ["xai-responses-previous-response-id-decoded"],
}

_XAI_UNSUPPORTED_INPUT_ITEM_REMOVED_TAG = "codex-unsupported-input-item-removed"
_XAI_UNSUPPORTED_INPUT_ITEM_TYPES = frozenset(
    {"reasoning", "tool_search_call", "tool_search_output"}
)

_XAI_CREDENTIAL_MAX_BYTES = 1_048_576
_GROK_NATIVE_CREDENTIAL_MAX_BYTES = 1_048_576


@dataclass(frozen=True)
class XaiOAuthCredentialSnapshot:
    """Immutable, validated managed xAI request snapshot."""

    credential_family: str
    auth_file: Path = field(repr=False, compare=False)
    scope: str
    access_token: str = field(repr=False, compare=False)
    generation: str
    account_identity: Optional[str]
    expires_at: datetime
    generation_metadata: Tuple[int, int, int, int, int] = field(
        repr=False,
        compare=False,
    )


_xai_snapshot_cache: Dict[
    Tuple[str, str, str],
    XaiOAuthCredentialSnapshot,
] = {}
_xai_snapshot_locks: Dict[str, asyncio.Lock] = {}
_xai_snapshot_flights: Dict[
    Tuple[str, str, str],
    "asyncio.Task[XaiOAuthCredentialSnapshot]",
] = {}
_XAI_MANAGED_SNAPSHOT_FAMILY = "xai_oauth"


@dataclass(frozen=True)
class GrokNativeOAuthCredentialSnapshot:
    """Immutable, validated native OIDC request snapshot."""

    auth_file: Path = field(repr=False, compare=False)
    scope: str
    access_token: str = field(repr=False, compare=False)
    expires_at: datetime
    generation: str
    file_fingerprint: Tuple[int, int, int, int, int] = field(
        repr=False,
        compare=False,
    )


_grok_native_snapshot_cache: Dict[
    Tuple[str, str],
    GrokNativeOAuthCredentialSnapshot,
] = {}
_grok_native_snapshot_locks: Dict[str, asyncio.Lock] = {}
_grok_native_snapshot_flights: Dict[
    Tuple[str, str],
    "asyncio.Task[GrokNativeOAuthCredentialSnapshot]",
] = {}
_refresh_locks: Dict[str, asyncio.Lock] = {}


def _write_private_file_text(path: Path, content: str, *, mode: int = 0o600) -> None:
    """Create/write path with restrictive mode at creation time (no umask window)."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = os.open(str(path), flags, mode)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    try:
        os.chmod(path, mode)
    except OSError:
        pass


def is_oa_xai_model(model: Any) -> bool:
    return get_oa_xai_route_descriptor(model) is not None


def normalize_grok_native_oauth_model(model: Any) -> Optional[str]:
    descriptor = get_grok_native_route_descriptor(model)
    return descriptor.public_model if descriptor is not None else None


def is_grok_native_oauth_model(model: Any) -> bool:
    return get_grok_native_route_descriptor(model) is not None


def resolve_oa_xai_upstream_model(model: str) -> str:
    return resolve_oa_xai_route_descriptor(model).upstream_model


def build_oa_xai_metadata(public_model: str, upstream_model: str) -> Dict[str, Any]:
    descriptor = resolve_oa_xai_route_descriptor(public_model)
    if upstream_model != descriptor.upstream_model:
        raise ValueError(
            "xAI OAuth upstream model does not match the authoritative route "
            f"descriptor for {public_model}."
        )
    return {
        "auth_mode": descriptor.auth_mode,
        "credential_family": descriptor.credential_family,
        "passthrough_route_family": descriptor.route_family,
        "route_family": descriptor.route_family,
        "xai_oauth_managed": True,
        "xai_oauth_public_model": descriptor.public_model,
        "xai_oauth_upstream_model": descriptor.upstream_model,
        "xai_quota_family": XAI_GROK_SUBSCRIPTION_QUOTA_FAMILY,
        "shared_quota_family": XAI_GROK_SUBSCRIPTION_QUOTA_FAMILY,
        "grok_subscription_quota_shared": True,
        "model_group": public_model,
        "tags": [
            "route:xai_oauth_api",
            "auth:xai_oauth",
            "provider:xai",
            "quota:xai_grok_subscription",
        ],
    }


def build_grok_native_oauth_metadata(public_model: str) -> Dict[str, Any]:
    descriptor = get_grok_native_route_descriptor(public_model)
    if descriptor is None:
        raise ValueError(f"Unsupported Grok native OIDC model: {public_model}")
    return {
        "auth_mode": descriptor.auth_mode,
        "credential_family": descriptor.credential_family,
        "client_name": GROK_NATIVE_OAUTH_CLIENT_NAME,
        "grok_cli_chat_proxy": True,
        "grok_model_override": descriptor.upstream_model,
        "grok_native_oauth_managed": True,
        "model_group": descriptor.public_model,
        "passthrough_route_family": descriptor.route_family,
        "route_family": descriptor.route_family,
        "shared_quota_family": XAI_GROK_SUBSCRIPTION_QUOTA_FAMILY,
        "xai_cli_chat_proxy": True,
        "xai_quota_family": XAI_GROK_SUBSCRIPTION_QUOTA_FAMILY,
        "tags": [
            "grok-build",
            "route:grok_cli_chat_proxy",
            "auth:grok_oidc",
            "provider:xai",
            "quota:xai_grok_subscription",
            f"grok-model:{descriptor.public_model}",
        ],
    }


async def prepare_oa_xai_request(
    data: Dict[str, Any],
    *,
    snapshot_out: Optional[MutableMapping[str, Any]] = None,
) -> bool:
    public_model = data.get("model")
    if not is_oa_xai_model(public_model):
        return False

    upstream_model = resolve_oa_xai_upstream_model(public_model)
    data["model"] = upstream_model
    data["api_base"] = get_secret_str("LITELLM_XAI_OAUTH_API_BASE") or XAI_API_BASE
    snapshot, credential_resolution = await _get_xai_oauth_snapshot_with_resolution()
    if snapshot_out is not None:
        snapshot_out["snapshot"] = snapshot
    data["api_key"] = snapshot.access_token
    data["custom_llm_provider"] = "xai"
    decoded_previous_response_id = _decode_previous_response_id_in_place(data)
    removed_input_items = _drop_xai_unsupported_input_items_in_place(data)

    existing_litellm_metadata = data.get("litellm_metadata")
    litellm_metadata = (
        dict(existing_litellm_metadata)
        if isinstance(existing_litellm_metadata, dict)
        else {}
    )
    data["litellm_metadata"] = litellm_metadata

    _merge_metadata(
        litellm_metadata,
        build_oa_xai_metadata(public_model, upstream_model),
        authoritative=True,
    )
    _merge_metadata(
        litellm_metadata,
        {
            "credential_identity": credential_resolution.credential_identity,
            "auth_file_source": credential_resolution.auth_file_source,
            "scope_source": credential_resolution.scope_source,
        },
        authoritative=True,
    )
    if decoded_previous_response_id:
        _merge_metadata(
            litellm_metadata,
            _XAI_RESPONSES_PREVIOUS_RESPONSE_ID_DECODED_METADATA,
            authoritative=True,
        )
    if removed_input_items:
        _merge_metadata(
            litellm_metadata,
            _build_xai_unsupported_input_removed_metadata(removed_input_items),
            authoritative=True,
        )

    return True


def _decode_previous_response_id_in_place(data: Dict[str, Any]) -> bool:
    previous_response_id = data.get("previous_response_id")
    if not isinstance(previous_response_id, str) or not previous_response_id:
        return False

    decoded = ResponsesAPIRequestUtils.decode_previous_response_id_to_original_previous_response_id(
        previous_response_id
    )
    if decoded == previous_response_id:
        return False

    data["previous_response_id"] = decoded
    return True


def _drop_xai_unsupported_input_items_in_place(
    data: Dict[str, Any],
) -> list[Dict[str, Any]]:
    input_items = data.get("input")
    if not isinstance(input_items, list):
        return []

    updated_input_items: list[Any] = []
    removed_items: list[Dict[str, Any]] = []
    for index, item in enumerate(input_items):
        if not isinstance(item, dict):
            updated_input_items.append(item)
            continue

        item_type = str(item.get("type") or "").lower()
        if item_type in _XAI_UNSUPPORTED_INPUT_ITEM_TYPES:
            removed_item: Dict[str, Any] = {"type": item_type, "index": index}
            if item_type == "reasoning" and isinstance(
                item.get("encrypted_content"), str
            ):
                removed_item["encrypted_content"] = True
            removed_items.append(removed_item)
            continue

        updated_input_items.append(item)

    if removed_items:
        data["input"] = updated_input_items
    return removed_items


def _build_xai_unsupported_input_removed_metadata(
    removed_items: list[Dict[str, Any]],
) -> Dict[str, Any]:
    removed_item_types = sorted(
        {
            item_type
            for item in removed_items
            if isinstance((item_type := item.get("type")), str) and item_type
        }
    )
    return {
        "codex_unsupported_input_item_removed_count": len(removed_items),
        "codex_unsupported_input_item_types_removed": removed_item_types,
        "codex_unsupported_input_items_removed": removed_items,
        "tags": [
            _XAI_UNSUPPORTED_INPUT_ITEM_REMOVED_TAG,
            *(
                f"codex-unsupported-input-item:{item_type}"
                for item_type in removed_item_types
            ),
        ],
    }


def _merge_metadata(
    target: Dict[str, Any],
    incoming: Dict[str, Any],
    *,
    authoritative: bool = False,
) -> None:
    incoming_tags = incoming.get("tags")
    existing_tags = target.get("tags")
    merged_tags: list[str] = []
    for tag_list in (existing_tags, incoming_tags):
        if not isinstance(tag_list, list):
            continue
        for tag in tag_list:
            if isinstance(tag, str) and tag not in merged_tags:
                merged_tags.append(tag)

    for key, value in incoming.items():
        if key == "tags":
            continue
        if authoritative:
            target[key] = value
        else:
            target.setdefault(key, value)
    if merged_tags:
        target["tags"] = merged_tags


def _xai_stat_fingerprint(
    stat_result: os.stat_result,
) -> Tuple[int, int, int, int, int]:
    return (
        int(stat_result.st_dev),
        int(stat_result.st_ino),
        int(stat_result.st_mtime_ns),
        int(stat_result.st_ctime_ns),
        int(stat_result.st_size),
    )


def _xai_stat_read_fingerprint(
    stat_result: os.stat_result,
) -> Tuple[int, int, int, int]:
    """Return descriptor-stable metadata for one complete read."""

    # Unlinking an open old inode changes its ctime/link metadata without
    # changing the bytes visible through the descriptor.
    return (
        int(stat_result.st_dev),
        int(stat_result.st_ino),
        int(stat_result.st_mtime_ns),
        int(stat_result.st_size),
    )


def _open_xai_credential_file(path: Path) -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise ValueError("xAI OAuth credential file cannot be opened safely.")
    flags = os.O_RDONLY | nofollow
    cloexec = getattr(os, "O_CLOEXEC", None)
    if cloexec is not None:
        flags |= cloexec
    try:
        return os.open(os.fspath(path), flags)
    except OSError as exc:
        if exc.errno == errno.ENOENT:
            raise FileNotFoundError from None
        if exc.errno in (errno.ELOOP, errno.EMLINK):
            raise ValueError("xAI OAuth credential file must not be a symlink.")
        raise ValueError("xAI OAuth credential file is unreadable.") from None


def _read_xai_credential_bytes(
    file_descriptor: int,
    *,
    max_bytes: int = _XAI_CREDENTIAL_MAX_BYTES,
) -> bytes:
    chunks: list[bytes] = []
    bytes_read = 0
    while bytes_read <= max_bytes:
        try:
            chunk = os.read(
                file_descriptor,
                min(8192, max_bytes + 1 - bytes_read),
            )
        except OSError:
            raise ValueError("xAI OAuth credential file could not be read.") from None
        if not chunk:
            break
        chunks.append(chunk)
        bytes_read += len(chunk)
    if bytes_read > max_bytes:
        raise ValueError("xAI OAuth credential file is too large.")
    return b"".join(chunks)


def _read_xai_credential_payload_secure(
    path: Path,
) -> Tuple[Dict[str, Any], os.stat_result]:
    """Read one complete managed credential generation from an open descriptor."""

    file_descriptor = _open_xai_credential_file(path)
    try:
        try:
            before = os.fstat(file_descriptor)
        except OSError:
            raise ValueError("xAI OAuth credential metadata is unreadable.") from None
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("xAI OAuth credential path is not a regular file.")
        if before.st_size > _XAI_CREDENTIAL_MAX_BYTES:
            raise ValueError("xAI OAuth credential file is too large.")
        raw_bytes = _read_xai_credential_bytes(file_descriptor)
        try:
            after = os.fstat(file_descriptor)
        except OSError:
            raise ValueError("xAI OAuth credential metadata is unreadable.") from None
    finally:
        try:
            os.close(file_descriptor)
        except OSError:
            pass

    if _xai_stat_read_fingerprint(before) != _xai_stat_read_fingerprint(after):
        raise ValueError("xAI OAuth credential changed while it was read.")
    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise ValueError("xAI OAuth credential file is not valid JSON.") from None
    if not isinstance(payload, dict):
        raise ValueError("xAI OAuth credential file must contain a JSON object.")
    return payload, after


def _stat_xai_credential_file(path: Path) -> os.stat_result:
    file_descriptor = _open_xai_credential_file(path)
    try:
        try:
            result = os.fstat(file_descriptor)
        except OSError:
            raise ValueError("xAI OAuth credential metadata is unreadable.") from None
    finally:
        try:
            os.close(file_descriptor)
        except OSError:
            pass
    if not stat.S_ISREG(result.st_mode):
        raise ValueError("xAI OAuth credential path is not a regular file.")
    if result.st_size > _XAI_CREDENTIAL_MAX_BYTES:
        raise ValueError("xAI OAuth credential file is too large.")
    return result


def _xai_snapshot_key(
    *,
    credential_family: str,
    credential_path: Path,
    scope: str,
) -> Tuple[str, str, str]:
    # Path resolution is performed in the input-resolution worker. Avoid
    # filesystem access here so request callers stay off blocking operations.
    return (credential_family, os.fspath(credential_path), scope)


def _xai_snapshot_is_route_usable(
    snapshot: XaiOAuthCredentialSnapshot,
) -> bool:
    return datetime.now(timezone.utc) < snapshot.expires_at - timedelta(
        seconds=_refresh_buffer_seconds()
    )


def _xai_oauth_credential_file_not_found_error(
    *,
    credential_path: Path,
) -> ValueError:
    return ValueError(
        f"xAI OAuth credential file not found at {credential_path}. Run the "
        "provider-status sidecar xAI OAuth refresh or reseed/relogin the "
        "managed xAI OAuth credential."
    )


def _start_xai_snapshot_flight(
    *,
    cache_key: Tuple[str, str, str],
    credential_path: Path,
    scope: str,
) -> "asyncio.Task[XaiOAuthCredentialSnapshot]":
    flight = asyncio.create_task(
        _run_xai_snapshot_flight(
            cache_key=cache_key,
            credential_path=credential_path,
            scope=scope,
        )
    )
    _xai_snapshot_flights[cache_key] = flight
    flight.add_done_callback(
        lambda completed: _finish_xai_snapshot_flight(
            cache_key,
            completed,
        )
    )
    return flight


async def _reload_xai_oauth_snapshot_after_flight(
    *,
    snapshot: XaiOAuthCredentialSnapshot,
    cache_key: Tuple[str, str, str],
    lock: asyncio.Lock,
    credential_path: Path,
    scope: str,
) -> XaiOAuthCredentialSnapshot:
    # A forced caller may have joined a flight that started before the file
    # was atomically replaced. Re-stat after that flight and perform at most
    # one follow-up load when its publication is no longer current.
    try:
        current_stat = await asyncio.to_thread(
            _stat_xai_credential_file,
            credential_path,
        )
    except FileNotFoundError as exc:
        raise _xai_oauth_credential_file_not_found_error(
            credential_path=credential_path,
        ) from exc
    if snapshot.generation_metadata == _xai_stat_fingerprint(current_stat):
        return snapshot

    async with lock:
        flight = _xai_snapshot_flights.get(cache_key)
        if flight is not None and flight.done():
            _xai_snapshot_flights.pop(cache_key, None)
            flight = None
        if flight is None:
            try:
                current_stat = await asyncio.to_thread(
                    _stat_xai_credential_file,
                    credential_path,
                )
            except FileNotFoundError as exc:
                raise _xai_oauth_credential_file_not_found_error(
                    credential_path=credential_path,
                ) from exc

            current_fingerprint = _xai_stat_fingerprint(current_stat)
            cached = _xai_snapshot_cache.get(cache_key)
            if (
                cached is not None
                and cached.generation_metadata == current_fingerprint
                and _xai_snapshot_is_route_usable(cached)
            ):
                return cached
            if cached is not None:
                _xai_snapshot_cache.pop(cache_key, None)
            flight = _start_xai_snapshot_flight(
                cache_key=cache_key,
                credential_path=credential_path,
                scope=scope,
            )

    try:
        return await asyncio.shield(flight)
    except FileNotFoundError as exc:
        raise _xai_oauth_credential_file_not_found_error(
            credential_path=credential_path,
        ) from exc


def _load_xai_oauth_snapshot_sync(
    *,
    credential_path: Path,
    scope: str,
) -> XaiOAuthCredentialSnapshot:
    payload, stat_result = _read_xai_credential_payload_secure(credential_path)
    credential = _select_credential_record(payload, scope)
    token = _credential_access_token(credential)
    if not token:
        raise _xai_oauth_refresh_required_error(missing_token=True)

    lifecycle = evaluate_xai_oauth_credential_lifecycle(
        credential,
        route_safety_buffer_seconds=_refresh_buffer_seconds(),
        refresh_min_seconds=_refresh_buffer_seconds(),
    )
    expires_at = credential_expires_at(credential)
    if (
        expires_at is None
        or not bool(lifecycle["route_usable"])
        or datetime.now(timezone.utc)
        >= expires_at - timedelta(seconds=_refresh_buffer_seconds())
    ):
        raise _xai_oauth_refresh_required_error(missing_token=False)

    generation = credential_identity(
        credential_path,
        credential,
        scope=scope,
        stat_result=stat_result,
    )
    if generation is None:
        raise ValueError("xAI OAuth credential generation is unavailable.")
    return XaiOAuthCredentialSnapshot(
        credential_family=_XAI_MANAGED_SNAPSHOT_FAMILY,
        auth_file=credential_path,
        scope=scope,
        access_token=token,
        generation=generation,
        account_identity=credential_account_identity(
            credential,
            scope=scope,
        ),
        expires_at=expires_at,
        generation_metadata=_xai_stat_fingerprint(stat_result),
    )


async def _get_xai_oauth_snapshot_for_path(
    *,
    credential_path: Path,
    scope: str,
    force_reload: bool = False,
) -> XaiOAuthCredentialSnapshot:
    cache_key = _xai_snapshot_key(
        credential_family=_XAI_MANAGED_SNAPSHOT_FAMILY,
        credential_path=credential_path,
        scope=scope,
    )
    lock_key = "\x1f".join(cache_key)
    try:
        observed_stat = await asyncio.to_thread(
            _stat_xai_credential_file,
            credential_path,
        )
    except FileNotFoundError as exc:
        raise _xai_oauth_credential_file_not_found_error(
            credential_path=credential_path,
        ) from exc

    observed_fingerprint = _xai_stat_fingerprint(observed_stat)
    cached = _xai_snapshot_cache.get(cache_key)
    if (
        not force_reload
        and cached is not None
        and cached.generation_metadata == observed_fingerprint
        and _xai_snapshot_is_route_usable(cached)
    ):
        return cached
    if cached is not None and (
        cached.generation_metadata != observed_fingerprint
        or not _xai_snapshot_is_route_usable(cached)
    ):
        _xai_snapshot_cache.pop(cache_key, None)

    lock = _xai_snapshot_locks.setdefault(lock_key, asyncio.Lock())
    async with lock:
        flight = _xai_snapshot_flights.get(cache_key)
        if flight is None:
            try:
                observed_stat = await asyncio.to_thread(
                    _stat_xai_credential_file,
                    credential_path,
                )
            except FileNotFoundError as exc:
                raise _xai_oauth_credential_file_not_found_error(
                    credential_path=credential_path,
                ) from exc

            observed_fingerprint = _xai_stat_fingerprint(observed_stat)
            cached = _xai_snapshot_cache.get(cache_key)
            if (
                not force_reload
                and cached is not None
                and cached.generation_metadata == observed_fingerprint
                and _xai_snapshot_is_route_usable(cached)
            ):
                return cached
            if cached is not None and (
                cached.generation_metadata != observed_fingerprint
                or not _xai_snapshot_is_route_usable(cached)
            ):
                _xai_snapshot_cache.pop(cache_key, None)

            flight = _start_xai_snapshot_flight(
                cache_key=cache_key,
                credential_path=credential_path,
                scope=scope,
            )

    try:
        snapshot = await asyncio.shield(flight)
    except FileNotFoundError as exc:
        raise _xai_oauth_credential_file_not_found_error(
            credential_path=credential_path,
        ) from exc
    if not force_reload:
        return snapshot
    return await _reload_xai_oauth_snapshot_after_flight(
        snapshot=snapshot,
        cache_key=cache_key,
        lock=lock,
        credential_path=credential_path,
        scope=scope,
    )


async def _run_xai_snapshot_flight(
    *,
    cache_key: Tuple[str, str, str],
    credential_path: Path,
    scope: str,
) -> XaiOAuthCredentialSnapshot:
    try:
        snapshot = await asyncio.to_thread(
            _load_xai_oauth_snapshot_sync,
            credential_path=credential_path,
            scope=scope,
        )
    except BaseException:
        _xai_snapshot_cache.pop(cache_key, None)
        raise
    _xai_snapshot_cache[cache_key] = snapshot
    return snapshot


def _finish_xai_snapshot_flight(
    cache_key: Tuple[str, str, str],
    flight: "asyncio.Task[XaiOAuthCredentialSnapshot]",
) -> None:
    if _xai_snapshot_flights.get(cache_key) is flight:
        _xai_snapshot_flights.pop(cache_key, None)
    if not flight.cancelled():
        flight.exception()


def _resolve_xai_oauth_snapshot_inputs_sync() -> XaiOAuthCredentialResolution:
    resolution = resolve_xai_oauth_credentials(value_getter=get_secret_str)
    if resolution.auth_file_source == "default":
        raise ValueError(
            "xAI OAuth-managed models require an explicit managed auth-file "
            "configuration (AAWM_XAI_OAUTH_AUTH_FILE or "
            "LITELLM_XAI_OAUTH_AUTH_FILE) pointing at the sidecar-maintained "
            "xAI OAuth credential file. Run the provider-status sidecar xAI "
            "OAuth refresh or reseed/relogin the managed credential before "
            "calling oa_xai/*."
        )
    return resolution


async def _get_xai_oauth_snapshot_with_resolution(
) -> Tuple[XaiOAuthCredentialSnapshot, XaiOAuthCredentialResolution]:
    resolution = await asyncio.to_thread(_resolve_xai_oauth_snapshot_inputs_sync)
    snapshot = await _get_xai_oauth_snapshot_for_path(
        credential_path=resolution.canonical_auth_file,
        scope=resolution.scope,
    )
    return snapshot, resolution


async def get_xai_oauth_snapshot() -> XaiOAuthCredentialSnapshot:
    """Return one immutable managed OAuth snapshot without blocking the loop."""

    snapshot, _resolution = await _get_xai_oauth_snapshot_with_resolution()
    return snapshot


async def get_xai_oauth_access_token() -> str:
    snapshot = await get_xai_oauth_snapshot()
    return snapshot.access_token


async def _get_xai_oauth_access_token_with_resolution(
) -> Tuple[str, XaiOAuthCredentialResolution]:
    """Read one managed credential snapshot and return its binding."""

    snapshot, resolution = await _get_xai_oauth_snapshot_with_resolution()
    return snapshot.access_token, resolution


async def get_grok_native_oauth_access_token() -> str:
    return (await get_grok_native_oauth_snapshot()).access_token


async def get_grok_native_oauth_snapshot() -> GrokNativeOAuthCredentialSnapshot:
    """Return one validated native OIDC snapshot without blocking the loop."""

    credential_path, scope = await asyncio.to_thread(
        _resolve_grok_native_snapshot_inputs_sync
    )
    return await _get_grok_native_oauth_snapshot_for_path(
        credential_path=credential_path,
        scope=scope,
    )


def _resolve_grok_native_snapshot_inputs_sync() -> Tuple[Path, str]:
    credential_path = default_grok_xai_oauth_auth_path()
    scope = (
        get_secret_str("LITELLM_XAI_GROK_OAUTH_SCOPE")
        or get_secret_str("LITELLM_XAI_OAUTH_SCOPE")
        or _DEFAULT_XAI_OAUTH_SCOPE
    )
    # Keep path normalization, including getcwd() for relative overrides, off
    # the request event loop.
    return Path(os.path.abspath(os.fspath(credential_path))), scope


def _grok_native_stat_fingerprint(
    stat_result: os.stat_result,
) -> Tuple[int, int, int, int, int]:
    return (
        int(stat_result.st_dev),
        int(stat_result.st_ino),
        int(stat_result.st_mtime_ns),
        int(stat_result.st_ctime_ns),
        int(stat_result.st_size),
    )


def _open_grok_native_credential_file(path: Path) -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise ValueError("Grok OIDC credential file cannot be opened safely.")
    flags = os.O_RDONLY | nofollow
    cloexec = getattr(os, "O_CLOEXEC", None)
    if cloexec is not None:
        flags |= cloexec
    try:
        return os.open(os.fspath(path), flags)
    except OSError as exc:
        if exc.errno == errno.ENOENT:
            raise FileNotFoundError from None
        if exc.errno in (errno.ELOOP, errno.EMLINK):
            raise ValueError("Grok OIDC credential file must not be a symlink.")
        raise ValueError("Grok OIDC credential file is unreadable.") from None


def _read_grok_native_credential_bytes(
    file_descriptor: int,
    *,
    max_bytes: int = _GROK_NATIVE_CREDENTIAL_MAX_BYTES,
) -> bytes:
    chunks: list[bytes] = []
    bytes_read = 0
    while bytes_read <= max_bytes:
        try:
            chunk = os.read(
                file_descriptor,
                min(8192, max_bytes + 1 - bytes_read),
            )
        except OSError:
            raise ValueError("Grok OIDC credential file could not be read.") from None
        if not chunk:
            break
        chunks.append(chunk)
        bytes_read += len(chunk)
    if bytes_read > max_bytes:
        raise ValueError("Grok OIDC credential file is too large.")
    return b"".join(chunks)


def _read_grok_native_credential_payload_secure(
    path: Path,
) -> Tuple[Dict[str, Any], os.stat_result, bytes]:
    file_descriptor = _open_grok_native_credential_file(path)
    try:
        try:
            before = os.fstat(file_descriptor)
        except OSError:
            raise ValueError("Grok OIDC credential metadata is unreadable.") from None
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("Grok OIDC credential path is not a regular file.")
        if before.st_size > _GROK_NATIVE_CREDENTIAL_MAX_BYTES:
            raise ValueError("Grok OIDC credential file is too large.")
        raw_bytes = _read_grok_native_credential_bytes(file_descriptor)
        try:
            after = os.fstat(file_descriptor)
        except OSError:
            raise ValueError("Grok OIDC credential metadata is unreadable.") from None
    finally:
        try:
            os.close(file_descriptor)
        except OSError:
            pass

    if _grok_native_stat_fingerprint(before) != _grok_native_stat_fingerprint(
        after
    ):
        raise ValueError("Grok OIDC credential changed while it was read.")
    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise ValueError("Grok OIDC credential file is not valid JSON.") from None
    if not isinstance(payload, dict):
        raise ValueError("Grok OIDC credential file must contain a JSON object.")
    return payload, after, raw_bytes


def _stat_grok_native_credential_file(path: Path) -> os.stat_result:
    file_descriptor = _open_grok_native_credential_file(path)
    try:
        try:
            result = os.fstat(file_descriptor)
        except OSError:
            raise ValueError("Grok OIDC credential metadata is unreadable.") from None
    finally:
        try:
            os.close(file_descriptor)
        except OSError:
            pass
    if not stat.S_ISREG(result.st_mode):
        raise ValueError("Grok OIDC credential path is not a regular file.")
    if result.st_size > _GROK_NATIVE_CREDENTIAL_MAX_BYTES:
        raise ValueError("Grok OIDC credential file is too large.")
    return result


def _grok_native_snapshot_key(
    credential_path: Path,
    scope: str,
) -> Tuple[str, str]:
    # The path is normalized by _resolve_grok_native_snapshot_inputs_sync().
    # Do not resolve through the filesystem: the descriptor read below rejects
    # a final symlink and fstat binds the opened file.
    return (os.fspath(credential_path), scope)


def _grok_native_snapshot_is_usable(
    snapshot: GrokNativeOAuthCredentialSnapshot,
) -> bool:
    return datetime.now(timezone.utc) < snapshot.expires_at - timedelta(
        seconds=_refresh_buffer_seconds()
    )


def _load_grok_native_snapshot_sync(
    *,
    credential_path: Path,
    scope: str,
) -> GrokNativeOAuthCredentialSnapshot:
    payload, stat_result, raw_bytes = _read_grok_native_credential_payload_secure(
        credential_path
    )
    credential = _select_credential_record(payload, scope)
    token = _credential_access_token(credential)
    if not token:
        raise _grok_native_oauth_refresh_required_error(missing_token=True)
    expires_at = _parse_expires_at(credential.get("expires_at"))
    if (
        expires_at is None
        or datetime.now(timezone.utc)
        >= expires_at - timedelta(seconds=_refresh_buffer_seconds())
    ):
        raise _grok_native_oauth_refresh_required_error(missing_token=False)
    return GrokNativeOAuthCredentialSnapshot(
        auth_file=credential_path,
        scope=scope,
        access_token=token,
        expires_at=expires_at,
        generation=f"sha256:{hashlib.sha256(raw_bytes).hexdigest()}",
        file_fingerprint=_grok_native_stat_fingerprint(stat_result),
    )


async def _get_grok_native_oauth_snapshot_for_path(
    *,
    credential_path: Path,
    scope: str,
) -> GrokNativeOAuthCredentialSnapshot:
    cache_key = _grok_native_snapshot_key(credential_path, scope)
    lock_key = "\x1f".join(cache_key)
    try:
        observed_stat = await asyncio.to_thread(
            _stat_grok_native_credential_file,
            credential_path,
        )
    except FileNotFoundError as exc:
        raise ValueError(
            f"Grok OIDC credential file not found at {credential_path}. "
            "Run the health/provider-status sidecar Grok OIDC refresh or "
            "relogin with the Grok CLI before Grok native traffic can proceed."
        ) from exc

    observed_fingerprint = _grok_native_stat_fingerprint(observed_stat)
    cached = _grok_native_snapshot_cache.get(cache_key)
    if (
        cached is not None
        and cached.file_fingerprint == observed_fingerprint
        and _grok_native_snapshot_is_usable(cached)
    ):
        return cached
    if cached is not None:
        _grok_native_snapshot_cache.pop(cache_key, None)

    lock = _grok_native_snapshot_locks.setdefault(lock_key, asyncio.Lock())
    async with lock:
        flight = _grok_native_snapshot_flights.get(cache_key)
        if flight is None:
            try:
                observed_stat = await asyncio.to_thread(
                    _stat_grok_native_credential_file,
                    credential_path,
                )
            except FileNotFoundError as exc:
                raise ValueError(
                    f"Grok OIDC credential file not found at {credential_path}. "
                    "Run the health/provider-status sidecar Grok OIDC refresh or "
                    "relogin with the Grok CLI before Grok native traffic can proceed."
                ) from exc
            observed_fingerprint = _grok_native_stat_fingerprint(observed_stat)
            cached = _grok_native_snapshot_cache.get(cache_key)
            if (
                cached is not None
                and cached.file_fingerprint == observed_fingerprint
                and _grok_native_snapshot_is_usable(cached)
            ):
                return cached
            if cached is not None:
                _grok_native_snapshot_cache.pop(cache_key, None)
            flight = asyncio.create_task(
                _run_grok_native_snapshot_flight(
                    cache_key=cache_key,
                    credential_path=credential_path,
                    scope=scope,
                )
            )
            _grok_native_snapshot_flights[cache_key] = flight
            flight.add_done_callback(
                lambda completed: _finish_grok_native_snapshot_flight(
                    cache_key,
                    completed,
                )
            )

    try:
        return await asyncio.shield(flight)
    except FileNotFoundError as exc:
        raise ValueError(
            f"Grok OIDC credential file not found at {credential_path}. "
            "Run the health/provider-status sidecar Grok OIDC refresh or "
            "relogin with the Grok CLI before Grok native traffic can proceed."
        ) from exc


async def _run_grok_native_snapshot_flight(
    *,
    cache_key: Tuple[str, str],
    credential_path: Path,
    scope: str,
) -> GrokNativeOAuthCredentialSnapshot:
    try:
        snapshot = await asyncio.to_thread(
            _load_grok_native_snapshot_sync,
            credential_path=credential_path,
            scope=scope,
        )
    except BaseException:
        _grok_native_snapshot_cache.pop(cache_key, None)
        raise
    _grok_native_snapshot_cache[cache_key] = snapshot
    return snapshot


def _finish_grok_native_snapshot_flight(
    cache_key: Tuple[str, str],
    flight: "asyncio.Task[GrokNativeOAuthCredentialSnapshot]",
) -> None:
    if _grok_native_snapshot_flights.get(cache_key) is flight:
        _grok_native_snapshot_flights.pop(cache_key, None)
    if not flight.cancelled():
        flight.exception()


def _grok_native_oauth_refresh_required_error(*, missing_token: bool) -> ValueError:
    if missing_token:
        message = (
            "Grok OIDC credential does not contain an access token. "
            "Run the health/provider-status sidecar Grok OIDC refresh or "
            "relogin with the Grok CLI before Grok native traffic can proceed."
        )
    else:
        message = (
            "Grok OIDC credential is missing, expired, or near expiry. "
            "Run the health/provider-status sidecar Grok OIDC refresh or "
            "relogin with the Grok CLI before Grok native traffic can proceed."
        )
    return ValueError(message)


def _xai_oauth_refresh_required_error(*, missing_token: bool) -> ValueError:
    if missing_token:
        message = (
            "Managed xAI OAuth credential does not contain an access token. "
            "Run the provider-status sidecar xAI OAuth refresh or reseed/relogin "
            "the managed credential before calling oa_xai/*."
        )
    else:
        message = (
            "Managed xAI OAuth credential is missing, expired, or near expiry. "
            "Run the provider-status sidecar xAI OAuth refresh before calling "
            "oa_xai/*. Reseed or relogin the managed credential if the "
            "sidecar cannot refresh it."
        )
    return ValueError(message)


def _read_grok_native_credential_payload(credential_path: Path) -> Dict[str, Any]:
    try:
        payload, _stat_result, _raw_bytes = (
            _read_grok_native_credential_payload_secure(credential_path)
        )
    except FileNotFoundError as exc:
        raise ValueError(
            f"Grok OIDC credential file not found at {credential_path}. "
            "Run the health/provider-status sidecar Grok OIDC refresh or "
            "relogin with the Grok CLI before Grok native traffic can proceed."
        ) from exc
    return payload


def _get_grok_native_oauth_access_token_read_only(
    *,
    credential_path: Path,
    scope: str,
) -> str:
    raw_payload = _read_grok_native_credential_payload(credential_path)
    credential = _select_credential_record(raw_payload, scope)
    token = _credential_access_token(credential)
    if token and not _credential_needs_refresh(credential):
        return token
    if not token:
        raise _grok_native_oauth_refresh_required_error(missing_token=True)
    raise _grok_native_oauth_refresh_required_error(missing_token=False)


def _get_xai_oauth_access_token_read_only(
    *,
    credential_path: Path,
    scope: str,
) -> str:
    raw_payload = _read_credential_payload(credential_path)
    credential = _select_credential_record(raw_payload, scope)
    token = _credential_access_token(credential)
    if token and not _credential_needs_refresh(credential):
        return token
    if not token:
        raise _xai_oauth_refresh_required_error(missing_token=True)
    raise _xai_oauth_refresh_required_error(missing_token=False)


def default_litellm_xai_oauth_auth_path() -> Path:
    configured = get_secret_str("LITELLM_XAI_OAUTH_MIGRATED_AUTH_FILE")
    if isinstance(configured, str) and configured.strip():
        return Path(configured.strip()).expanduser()
    return Path(_DEFAULT_LITELLM_XAI_OAUTH_AUTH_PATH).expanduser()


def default_grok_xai_oauth_auth_path() -> Path:
    return resolve_grok_oidc_auth_path(
        value_getter=get_secret_str,
    ).path


def migrate_hermes_xai_oauth_credential(
    *,
    hermes_auth_file: Optional[Path] = None,
    target_auth_file: Optional[Path] = None,
    scope: Optional[str] = None,
    overwrite: bool = False,
) -> Path:
    source_path = (
        hermes_auth_file
        or Path(
            get_secret_str("LITELLM_XAI_OAUTH_HERMES_AUTH_FILE")
            or _DEFAULT_HERMES_AUTH_PATH
        )
    ).expanduser()
    target_path = (
        target_auth_file or default_litellm_xai_oauth_auth_path()
    ).expanduser()
    _validate_xai_oauth_migration_target(source_path, target_path)

    if target_path.exists() and not overwrite:
        raise FileExistsError(
            f"xAI OAuth target credential already exists at {target_path}. "
            "Pass overwrite=True or choose a different LiteLLM-owned path."
        )

    hermes_payload = _read_json_object(
        source_path,
        description="Hermes auth file",
    )
    credential_scope = (
        scope or get_secret_str("LITELLM_XAI_OAUTH_SCOPE") or _DEFAULT_XAI_OAUTH_SCOPE
    )
    credential = _build_litellm_xai_oauth_record_from_hermes(
        hermes_payload,
        scope=credential_scope,
    )
    _write_credential_payload(target_path, {credential_scope: credential})
    return target_path


def _validate_xai_oauth_migration_target(source_path: Path, target_path: Path) -> None:
    resolved_source_parent = source_path.expanduser().resolve().parent
    resolved_target = target_path.expanduser().resolve()
    if any(part == ".hermes" for part in resolved_target.parts):
        raise ValueError(
            "xAI OAuth migration target must be outside the user's .hermes directory."
        )
    if resolved_target == source_path.expanduser().resolve():
        raise ValueError("xAI OAuth migration target cannot be the Hermes source file.")
    try:
        resolved_target.relative_to(resolved_source_parent)
    except ValueError:
        return
    raise ValueError(
        "xAI OAuth migration target must be outside the Hermes auth directory."
    )


def _read_json_object(path: Path, *, description: str) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise ValueError(f"{description} not found at {path}.") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{description} at {path} is not valid JSON.") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"{description} must contain a JSON object.")
    return payload


def _build_litellm_xai_oauth_record_from_hermes(
    payload: Dict[str, Any],
    *,
    scope: str,
) -> Dict[str, Any]:
    provider = _extract_hermes_xai_oauth_provider(payload)
    provider_tokens = provider.get("tokens") if isinstance(provider, dict) else None
    credential = (
        _record_from_hermes_provider_tokens(provider, provider_tokens, scope=scope)
        if isinstance(provider_tokens, dict)
        else None
    )
    if credential is None:
        credential = _record_from_hermes_credential_pool(payload, scope=scope)
    if credential is None:
        raise ValueError(
            "Hermes auth file does not contain a usable xai-oauth credential."
        )
    return credential


def _extract_hermes_xai_oauth_provider(payload: Dict[str, Any]) -> Dict[str, Any]:
    providers = payload.get("providers")
    if isinstance(providers, dict):
        provider = providers.get(_DEFAULT_HERMES_XAI_OAUTH_PROVIDER_ID)
        if isinstance(provider, dict):
            return provider
    return {}


def _record_from_hermes_provider_tokens(
    provider: Dict[str, Any],
    tokens: Dict[str, Any],
    *,
    scope: str,
) -> Optional[Dict[str, Any]]:
    access_token = _clean_oauth_string(tokens.get("access_token"))
    refresh_token = _clean_oauth_string(tokens.get("refresh_token"))
    if not access_token and not refresh_token:
        return None

    record = _base_xai_oauth_record(scope)
    _copy_oauth_token_fields(record, tokens)

    discovery = provider.get("discovery")
    if isinstance(discovery, dict):
        token_endpoint = _clean_oauth_string(discovery.get("token_endpoint"))
        if token_endpoint:
            record["token_endpoint"] = token_endpoint

    last_refresh = _parse_expires_at(provider.get("last_refresh"))
    expires_in = tokens.get("expires_in")
    if last_refresh is not None and isinstance(expires_in, (int, float)):
        expires_at = last_refresh + timedelta(seconds=float(expires_in))
        record["expires_at"] = expires_at.isoformat().replace("+00:00", "Z")

    auth_mode = _clean_oauth_string(provider.get("auth_mode"))
    if auth_mode:
        record["source_auth_mode"] = auth_mode
    redirect_uri = _clean_oauth_string(provider.get("redirect_uri"))
    if redirect_uri:
        record["redirect_uri"] = redirect_uri
    record["source"] = "hermes.providers.xai-oauth"
    return record


def _record_from_hermes_credential_pool(
    payload: Dict[str, Any],
    *,
    scope: str,
) -> Optional[Dict[str, Any]]:
    credential_pool = payload.get("credential_pool")
    if not isinstance(credential_pool, dict):
        return None
    pool = credential_pool.get(_DEFAULT_HERMES_XAI_OAUTH_PROVIDER_ID)
    if not isinstance(pool, list):
        return None

    for item in pool:
        if not isinstance(item, dict):
            continue
        access_token = _clean_oauth_string(item.get("access_token"))
        refresh_token = _clean_oauth_string(item.get("refresh_token"))
        if not access_token and not refresh_token:
            continue
        record = _base_xai_oauth_record(scope)
        _copy_oauth_token_fields(record, item)
        base_url = _clean_oauth_string(item.get("base_url"))
        if base_url:
            record["source_base_url"] = base_url
        last_refresh = _parse_expires_at(item.get("last_refresh"))
        if last_refresh is not None:
            record["source_last_refresh"] = last_refresh.isoformat().replace(
                "+00:00",
                "Z",
            )
        source = _clean_oauth_string(item.get("source"))
        record["source"] = source or "hermes.credential_pool.xai-oauth"
        return record
    return None


def _base_xai_oauth_record(scope: str) -> Dict[str, Any]:
    client_id = _clean_oauth_string(
        scope.rsplit("::", 1)[-1] if "::" in scope else None
    )
    record: Dict[str, Any] = {
        "token_endpoint": _DEFAULT_XAI_OAUTH_TOKEN_ENDPOINT,
    }
    if client_id:
        record["oidc_client_id"] = client_id
    return record


def _copy_oauth_token_fields(record: Dict[str, Any], source: Dict[str, Any]) -> None:
    access_token = _clean_oauth_string(source.get("access_token"))
    if access_token:
        record["key"] = access_token
        record["access_token"] = access_token
    for key in ("refresh_token", "id_token", "token_type"):
        value = _clean_oauth_string(source.get(key))
        if value:
            record[key] = value


def _clean_oauth_string(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _read_credential_payload(credential_path: Path) -> Dict[str, Any]:
    try:
        with credential_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise ValueError(
            f"xAI OAuth credential file not found at {credential_path}. Run the "
            "provider-status sidecar xAI OAuth refresh or reseed/relogin the "
            "managed xAI OAuth credential."
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"xAI OAuth credential file at {credential_path} is not valid JSON."
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError("xAI OAuth credential file must contain a JSON object.")
    return payload


def _select_credential_record(
    payload: Dict[str, Any],
    scope: str,
) -> MutableMapping[str, Any]:
    return select_xai_oauth_credential_record(
        payload,
        scope,
        provider_label="xAI OAuth",
    )


def _credential_access_token(credential: Mapping[str, Any]) -> Optional[str]:
    return credential_access_token(credential)


def _credential_needs_refresh(credential: Mapping[str, Any]) -> bool:
    """Return True when the credential should not be used as-is.

    Missing or unparseable ``expires_at`` fails safe toward refresh (not
    permanently fresh). Production accessors are read-only and raise a sidecar
    refresh-required error in that case rather than minting a new token here.
    """
    buffer_seconds = _refresh_buffer_seconds()
    lifecycle = evaluate_xai_oauth_credential_lifecycle(
        credential,
        route_safety_buffer_seconds=buffer_seconds,
        refresh_min_seconds=buffer_seconds,
    )
    return not bool(lifecycle["route_usable"])


def _parse_expires_at(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return _datetime_from_epoch_numeric(float(value))
    if isinstance(value, str) and value.strip():
        normalized = value.strip()
        try:
            return _datetime_from_epoch_numeric(float(normalized))
        except ValueError:
            pass
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return None


def _datetime_from_epoch_numeric(raw_value: float) -> datetime:
    # Grok auth files may persist expiry as epoch seconds or milliseconds.
    if raw_value >= 1_000_000_000_000:
        raw_value = raw_value / 1000.0
    return datetime.fromtimestamp(raw_value, tz=timezone.utc)


def _refresh_buffer_seconds() -> int:
    raw_value = get_secret_str("LITELLM_XAI_OAUTH_REFRESH_BUFFER_SECONDS")
    if raw_value is None:
        return _DEFAULT_REFRESH_BUFFER_SECONDS
    try:
        return max(0, int(raw_value))
    except ValueError:
        return _DEFAULT_REFRESH_BUFFER_SECONDS


# NOTE (RR-040 #4): In-process locked refresh was intentionally removed.
# Production paths use the read-only accessors above; sidecars own refresh.


def _write_credential_payload(credential_path: Path, payload: Dict[str, Any]) -> None:
    credential_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = credential_path.with_name(f".{credential_path.name}.{os.getpid()}.tmp")
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    _write_private_file_text(tmp_path, content, mode=0o600)
    os.replace(tmp_path, credential_path)
