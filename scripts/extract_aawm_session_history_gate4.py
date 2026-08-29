"""Bounded Gate 4 extractor for live AAWM session-history state.

This file is intentionally standalone: it uses only the standard library and
resolves runtime objects from modules already present in ``sys.modules``.  It
is suitable for decoding to a source string and executing with
``PyRun_SimpleString`` in the live process; it has no CLI or import side
effects.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import queue
import stat
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


_DISPOSITION = "extracted_unvalidated"
_FORMAT_NAME = b"aawm-session-history-gate4-spool-format-v1"
_RESERVED_NAMES = frozenset(
    {
        ".",
        "..",
        "inflight.jsonl",
        "manifest.json",
        "queued.jsonl",
        "sha256sums.txt",
    }
)
_IDENTITY_FIELDS = ("litellm_call_id", "id", "session_id")
_RECORD_SETS = ("queued", "inflight")
_RUNTIME_MODULE = "litellm.integrations.aawm_session_history.runtime"
_SPOOL_MODULE = "litellm.integrations.aawm_session_history.spool"
_IDENTITY_MODULE = "litellm.integrations.aawm_agent_identity"
_STATE_NAMES = (
    "_aawm_session_history_queue",
    "_aawm_session_history_worker_inflight_records",
    "_aawm_session_history_shutdown_lock",
    "_aawm_session_history_shutdown_in_progress",
)
_WORKER_STATE_NAME = "_aawm_session_history_worker"
_SET_STATE_NAME = "_set_state"
_ENCODER_NAME = "_encode_session_history_spool_value"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _module(name: str) -> Optional[Any]:
    return sys.modules.get(name)


def _same_authoritative_state(
    runtime: Any,
    identity: Optional[Any],
    names: Iterable[str],
) -> None:
    for name in names:
        runtime_value = getattr(runtime, name)
        if identity is not None and hasattr(identity, name):
            if getattr(identity, name) is not runtime_value:
                raise RuntimeError(f"ambiguous session_history state: {name}")


def _resolve_runtime_module() -> Any:
    runtime = _module(_RUNTIME_MODULE)
    if runtime is None:
        raise RuntimeError("session_history runtime module is not loaded")
    return runtime


def _resolve_shutdown_lock_and_state() -> Tuple[Any, Any, Any]:
    runtime = _resolve_runtime_module()
    shutdown_lock = runtime._aawm_session_history_shutdown_lock
    set_state = runtime._set_state
    if not callable(shutdown_lock.acquire) or not callable(shutdown_lock.release):
        raise RuntimeError("session_history shutdown lock is invalid")
    if not callable(set_state):
        raise RuntimeError("session_history _set_state is invalid")
    return runtime, shutdown_lock, set_state


def _resolve_live_state() -> Tuple[Any, Any, Any, Any, Any, Any]:
    runtime = _resolve_runtime_module()

    # Worker bindings re-exported by the identity compatibility package can
    # lag runtime after a worker replacement. Runtime owns the manifest-only
    # worker value; queue, in-flight, shutdown, and _set_state must still match.
    identity = _module(_IDENTITY_MODULE)
    _same_authoritative_state(runtime, identity, (*_STATE_NAMES, _SET_STATE_NAME))

    queue_obj = runtime._aawm_session_history_queue
    inflight = runtime._aawm_session_history_worker_inflight_records
    shutdown_lock = runtime._aawm_session_history_shutdown_lock
    set_state = runtime._set_state

    if not isinstance(queue_obj, queue.Queue):
        raise RuntimeError("session_history queue is not queue.Queue")
    if not hasattr(queue_obj, "mutex"):
        raise RuntimeError("session_history queue mutex is unavailable")
    if not callable(shutdown_lock.acquire) or not callable(shutdown_lock.release):
        raise RuntimeError("session_history shutdown lock is invalid")
    if not callable(set_state):
        raise RuntimeError("session_history _set_state is invalid")
    if not isinstance(runtime._aawm_session_history_shutdown_in_progress, bool):
        raise RuntimeError("session_history shutdown mode is invalid")
    if not isinstance(inflight, list):
        raise RuntimeError("session_history in-flight batch is not a list")

    return (
        runtime,
        queue_obj,
        inflight,
        shutdown_lock,
        set_state,
        getattr(runtime, _WORKER_STATE_NAME),
    )


def _resolve_authoritative_state_after_shutdown_lock(
    expected_runtime: Any,
) -> Tuple[Any, Any, Any]:
    live_state = _resolve_live_state()
    queue_obj = live_state[1]
    inflight = live_state[2]
    worker = live_state[5]
    if _module(_RUNTIME_MODULE) is not expected_runtime:
        raise RuntimeError("session_history runtime changed during shutdown lock")
    if _module(_RUNTIME_MODULE)._aawm_session_history_queue is not queue_obj:
        raise RuntimeError("session_history queue changed during shutdown lock")
    return queue_obj, inflight, worker


def _resolve_live_encoder() -> Any:
    spool = _module(_SPOOL_MODULE)
    if spool is None:
        raise RuntimeError("session_history spool module is not loaded")
    if not hasattr(spool, _ENCODER_NAME):
        raise RuntimeError("session_history spool encoder is unavailable")
    encoder = getattr(spool, _ENCODER_NAME)
    if not callable(encoder):
        raise RuntimeError("session_history spool encoder is invalid")

    for module_name in (
        "litellm.integrations.aawm_session_history",
        "litellm.integrations.aawm_session_history.writer",
        _IDENTITY_MODULE,
    ):
        alternate = _module(module_name)
        if alternate is not None and hasattr(alternate, _ENCODER_NAME):
            if getattr(alternate, _ENCODER_NAME) is not encoder:
                raise RuntimeError("ambiguous session_history spool encoder")
    return encoder


def _resolve_active_spool_dir() -> str:
    spool = _module(_SPOOL_MODULE)
    if spool is None:
        raise RuntimeError("session_history spool module is not loaded")
    getter = getattr(spool, "_get_session_history_spool_dir", None)
    if not callable(getter):
        raise RuntimeError("session_history spool directory resolver is unavailable")

    for module_name in (
        "litellm.integrations.aawm_session_history",
        _IDENTITY_MODULE,
    ):
        alternate = _module(module_name)
        if alternate is not None and hasattr(alternate, "_get_session_history_spool_dir"):
            if getattr(alternate, "_get_session_history_spool_dir") is not getter:
                raise RuntimeError("ambiguous session_history spool directory resolver")

    spool_dir = getter()
    if not isinstance(spool_dir, str) or not spool_dir:
        raise RuntimeError("session_history spool directory is invalid")
    return os.path.realpath(spool_dir)


def _reject_symlink_components(path: str) -> None:
    current = os.path.abspath(os.sep)
    for part in os.path.abspath(path).split(os.sep)[1:]:
        current = os.path.join(current, part)
        if os.path.islink(current):
            raise RuntimeError(f"output path contains symlink component: {current}")


def _reject_active_spool_output(output_dir: str) -> None:
    spool_dir = _resolve_active_spool_dir()
    output_real_path = os.path.realpath(output_dir)
    if os.path.commonpath((output_real_path, spool_dir)) == spool_dir:
        raise RuntimeError(f"output directory is inside active session_history spool: {spool_dir}")


def _validate_arguments(output_dir: str, lock_timeout_seconds: float, extractor_sha256: str) -> str:
    if not isinstance(output_dir, str) or not output_dir:
        raise ValueError("output_dir must be a non-empty string")
    if isinstance(lock_timeout_seconds, bool) or not isinstance(lock_timeout_seconds, (int, float)):
        raise ValueError("lock_timeout_seconds must be a finite number")
    if not math.isfinite(lock_timeout_seconds) or lock_timeout_seconds < 0:
        raise ValueError("lock_timeout_seconds must be finite and non-negative")
    if not isinstance(extractor_sha256, str) or len(extractor_sha256) != 64:
        raise ValueError("extractor_sha256 must be a SHA-256 hex digest")
    try:
        int(extractor_sha256, 16)
    except ValueError as exc:
        raise ValueError("extractor_sha256 must be a SHA-256 hex digest") from exc
    if extractor_sha256 != extractor_sha256.lower():
        raise ValueError("extractor_sha256 must use lowercase hexadecimal")

    normalized_output = os.path.abspath(output_dir)
    if os.path.basename(normalized_output) in _RESERVED_NAMES:
        raise ValueError("output_dir uses a reserved artifact name")
    _reject_symlink_components(normalized_output)
    if os.path.lexists(normalized_output):
        raise FileExistsError(f"output directory already exists: {normalized_output}")
    _reject_active_spool_output(normalized_output)
    return normalized_output


def _identity_occurrences(
    records: Mapping[str, List[Dict[str, Any]]],
    encoder: Any,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
    present: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []

    for set_name in _RECORD_SETS:
        for index, record in enumerate(records[set_name]):
            record_has_identity = False
            for field in _IDENTITY_FIELDS:
                value = record.get(field)
                if value is None or (isinstance(value, str) and not value):
                    continue
                encoded_value = encoder(value)
                canonical_value = json.dumps(
                    encoded_value,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                record_has_identity = True
                occurrence = {"set": set_name, "index": index}
                present.append({**occurrence, "field": field, "value": encoded_value})
                group_key = (field, canonical_value)
                group = grouped.setdefault(
                    group_key,
                    {"field": field, "value": encoded_value, "occurrences": {}},
                )
                group["occurrences"].setdefault(set_name, []).append(index)
            if not record_has_identity:
                missing.append({"set": set_name, "index": index})

    within: List[Dict[str, Any]] = []
    across: List[Dict[str, Any]] = []
    for group in grouped.values():
        occurrences = group["occurrences"]
        queued_indices = occurrences.get("queued", [])
        inflight_indices = occurrences.get("inflight", [])
        if len(queued_indices) > 1 or len(inflight_indices) > 1:
            within.append({**group, "occurrences": occurrences})
        if queued_indices and inflight_indices:
            across.append({**group, "occurrences": occurrences})

    return present, missing, {"within": within, "across": across}


def _copy_live_records(
    queue_obj: queue.Queue,
    inflight: List[Dict[str, Any]],
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, int], int]:
    raw_queued = list(queue_obj.queue)
    queued_copied = [copy.deepcopy(record) for record in raw_queued if record is not None]
    inflight_copied = [copy.deepcopy(record) for record in inflight if record is not None]
    sentinels_excluded = sum(record is None for record in raw_queued)
    sentinels_excluded += sum(record is None for record in inflight)

    for set_name, copied_records in (
        ("queued", queued_copied),
        ("inflight", inflight_copied),
    ):
        for record in copied_records:
            if not isinstance(record, dict):
                raise RuntimeError(f"session_history {set_name} record is not a dictionary")

    counts = {
        "queued": len(queued_copied),
        "inflight": len(inflight_copied),
        "sentinels_excluded": sentinels_excluded,
    }
    queue_counts = {
        "size_before_copy": len(raw_queued),
        "size_after_copy": len(queue_obj.queue),
        "maxsize": queue_obj.maxsize,
        "unfinished_tasks": queue_obj.unfinished_tasks,
    }
    return {"queued": queued_copied, "inflight": inflight_copied}, counts, queue_counts


def _worker_snapshot(worker: Optional[threading.Thread]) -> Dict[str, Any]:
    if worker is None:
        return {"present": False, "name": None, "ident": None, "is_alive": None}
    return {
        "present": True,
        "name": worker.name,
        "ident": worker.ident,
        "is_alive": worker.is_alive(),
    }


def _artifact_hash_and_size(path: str) -> Tuple[str, int]:
    digest = hashlib.sha256()
    size_bytes = 0
    with open(path, "rb") as artifact_file:
        while True:
            chunk = artifact_file.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size_bytes += len(chunk)
    return digest.hexdigest(), size_bytes


def _json_bytes(payload: Mapping[str, Any], encoder: Any) -> bytes:
    encoded = encoder(payload)
    text = json.dumps(
        encoded,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return (text + "\n").encode("utf-8")


def _spool_jsonl(
    records: List[Dict[str, Any]],
    *,
    reason: str,
    extracted_at: datetime,
    encoder: Any,
) -> bytes:
    lines: List[bytes] = []
    metadata = {
        "type": "metadata",
        "format_version": 1,
        "spooled_at": extracted_at,
        "reason": reason,
        "retry_count": None,
        "record_count": len(records),
    }
    lines.append(_json_bytes(metadata, encoder))
    for index, record in enumerate(records):
        line = {
            "type": "record",
            "index": index,
            "record": record,
        }
        lines.append(_json_bytes(line, encoder))
    return b"".join(lines)


def _open_flags() -> int:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is not None:
        flags |= no_follow
    binary = getattr(os, "O_BINARY", None)
    if binary is not None:
        flags |= binary
    return flags


def _verify_regular_file(path: str) -> None:
    artifact_stat = os.lstat(path)
    if stat.S_ISLNK(artifact_stat.st_mode) or not stat.S_ISREG(artifact_stat.st_mode):
        raise RuntimeError(f"artifact is not a regular file: {path}")
    if stat.S_IMODE(artifact_stat.st_mode) != 0o600:
        raise RuntimeError(f"artifact mode is not 0600: {path}")


def _write_atomic(directory: str, filename: str, payload: bytes) -> None:
    final_path = os.path.join(directory, filename)
    temporary_path = os.path.join(
        directory,
        f".{filename}.{time.time_ns()}.{threading.get_ident()}.tmp",
    )
    descriptor: Optional[int] = None
    try:
        descriptor = os.open(temporary_path, _open_flags(), 0o600)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as artifact_file:
            descriptor = None
            artifact_file.write(payload)
            artifact_file.flush()
            os.fsync(artifact_file.fileno())
        try:
            os.rename(temporary_path, final_path)
        except BaseException:
            raise
        else:
            _fsync_directory(directory)
        _verify_regular_file(final_path)
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise


def _fsync_directory(directory: str) -> None:
    flags = os.O_RDONLY
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if directory_flag is not None:
        flags |= directory_flag
    binary = getattr(os, "O_BINARY", None)
    if binary is not None:
        flags |= binary
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _create_private_directory(output_dir: str) -> None:
    os.mkdir(output_dir, 0o700)
    directory_stat = os.lstat(output_dir)
    if stat.S_ISLNK(directory_stat.st_mode) or not stat.S_ISDIR(directory_stat.st_mode):
        raise RuntimeError(f"output path is not a real directory: {output_dir}")
    os.chmod(output_dir, 0o700)
    directory_stat = os.lstat(output_dir)
    if stat.S_IMODE(directory_stat.st_mode) != 0o700:
        raise RuntimeError(f"output directory mode is not 0700: {output_dir}")


def extract_gate4(
    output_dir: str,
    *,
    lock_timeout_seconds: float,
    extractor_sha256: str,
) -> dict:
    """Copy live session-history records without claiming or clearing them."""

    extraction_started_at = _utc_now()
    normalized_output = _validate_arguments(
        output_dir,
        lock_timeout_seconds,
        extractor_sha256,
    )
    runtime = _resolve_runtime_module()
    encoder = _resolve_live_encoder()
    runtime, shutdown_lock, set_state = _resolve_shutdown_lock_and_state()

    shutdown_lock_acquired = False
    queue_mutex_acquired = False
    try:
        if not shutdown_lock.acquire(True, lock_timeout_seconds):
            raise TimeoutError("timed out acquiring session_history shutdown lock")
        shutdown_lock_acquired = True
        set_state("_aawm_session_history_shutdown_in_progress", True)

        queue_obj, inflight, worker = _resolve_authoritative_state_after_shutdown_lock(runtime)

        if not queue_obj.mutex.acquire(True, lock_timeout_seconds):
            raise TimeoutError("timed out acquiring session_history queue mutex")
        queue_mutex_acquired = True
        records, counts, queue_counts = _copy_live_records(queue_obj, inflight)
        worker_state = _worker_snapshot(worker)
    finally:
        if queue_mutex_acquired:
            queue_obj.mutex.release()
        if shutdown_lock_acquired:
            shutdown_lock.release()

    extraction_completed_at = _utc_now()
    identities, missing_identities, duplicate_identities = _identity_occurrences(
        records,
        encoder,
    )

    _create_private_directory(normalized_output)
    queued_path = os.path.join(normalized_output, "queued.jsonl")
    inflight_path = os.path.join(normalized_output, "inflight.jsonl")
    manifest_path = os.path.join(normalized_output, "manifest.json")
    checksum_path = os.path.join(normalized_output, "sha256sums.txt")

    queued_payload = _spool_jsonl(
        records["queued"],
        reason="gate4_extraction_queued",
        extracted_at=extraction_completed_at,
        encoder=encoder,
    )
    inflight_payload = _spool_jsonl(
        records["inflight"],
        reason="gate4_extraction_inflight",
        extracted_at=extraction_completed_at,
        encoder=encoder,
    )
    _write_atomic(normalized_output, "queued.jsonl", queued_payload)
    _write_atomic(normalized_output, "inflight.jsonl", inflight_payload)

    artifact_metadata: Dict[str, Dict[str, Any]] = {}
    for artifact_name, artifact_path in (
        ("queued.jsonl", queued_path),
        ("inflight.jsonl", inflight_path),
    ):
        artifact_hash, artifact_size = _artifact_hash_and_size(artifact_path)
        artifact_metadata[artifact_name] = {
            "sha256": artifact_hash,
            "size_bytes": artifact_size,
        }

    manifest = {
        "type": "aawm_session_history_gate4_manifest",
        "format_version": 1,
        "spool_format_version": 1,
        "disposition": _DISPOSITION,
        "extraction_started_at": extraction_started_at,
        "extraction_completed_at": extraction_completed_at,
        "counts": {
            **counts,
            "identities": len(identities),
            "missing_identities": len(missing_identities),
        },
        "queue": queue_counts,
        "worker": worker_state,
        "identities": {
            "fields": list(_IDENTITY_FIELDS),
            "present": identities,
            "missing": missing_identities,
            "duplicates": duplicate_identities,
        },
        "artifacts": artifact_metadata,
        "hashes": {
            "format_sha256": hashlib.sha256(_FORMAT_NAME).hexdigest(),
            "extractor_sha256": extractor_sha256,
        },
    }
    manifest_payload = _json_bytes(manifest, encoder)
    _write_atomic(normalized_output, "manifest.json", manifest_payload)
    manifest_hash, _ = _artifact_hash_and_size(manifest_path)

    checksums = b"".join(
        [
            (f"{artifact_metadata['queued.jsonl']['sha256']}  queued.jsonl\n").encode("utf-8"),
            (f"{artifact_metadata['inflight.jsonl']['sha256']}  inflight.jsonl\n").encode("utf-8"),
            f"{manifest_hash}  manifest.json\n".encode("utf-8"),
        ]
    )
    _write_atomic(normalized_output, "sha256sums.txt", checksums)

    return {
        "disposition": _DISPOSITION,
        "output_dir": normalized_output,
        "queued_count": counts["queued"],
        "inflight_count": counts["inflight"],
        "artifacts": {
            "queued": queued_path,
            "inflight": inflight_path,
            "manifest": manifest_path,
            "sha256sums": checksum_path,
        },
        "manifest": {
            "sha256": manifest_hash,
            "size_bytes": len(manifest_payload),
        },
    }
