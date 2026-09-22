#!/usr/bin/env python3
"""Host-side publisher for the ZCode native header contract.

Request serving must not import this module and must not SSH. Request
serving reads the local descriptor on each call. File identity makes an
atomic replace visible without a process restart and without SSH. Host
maintenance compares a locally visible AppImage SHA-256 with
source.appimage_sha256. A mismatch stops and asks for a new static
inspection. It does not extract headers or invent a client version. The
host check is daily or when the package changes. That cadence is not a
request-path poll. Egress changes only after the descriptor is atomically
replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

_DEFAULT_DESCRIPTOR = Path(".analysis/zai/20260922/zcode-native-header-contract.json")
_SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")
_SECRET_FIELDS = frozenset({"privateCipher", "apiKeySecret", "apiKey"})
_HASH_CHUNK_BYTES = 1024 * 1024
_PROFILE_NAMES = ("app_server", "cli")


class ContractError(Exception):
    """Descriptor check failed without echoing secrets or header values."""


def _write_line(message: str, *, error: bool = False) -> None:
    stream = sys.stderr if error else sys.stdout
    stream.write(f"{message}\n")


def _default_descriptor() -> Path:
    current = Path.cwd().resolve()
    for directory in (current, *current.parents):
        candidate = directory / _DEFAULT_DESCRIPTOR
        if candidate.is_file():
            return candidate
    raise ContractError(f"descriptor absent: {_DEFAULT_DESCRIPTOR.as_posix()}")


def _read_bytes(path: Path, label: str) -> bytes:
    if not path.is_file():
        raise ContractError(f"{label} absent: {path}")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ContractError(f"{label} unreadable: {path}") from exc


def _parse_object(payload: bytes) -> dict[str, object]:
    try:
        parsed: object = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ContractError("descriptor is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ContractError("descriptor must be a JSON object")
    return parsed


def _object_field(
    document: dict[str, object], key: str, label: str
) -> dict[str, object]:
    value = document.get(key)
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a JSON object")
    return value


def _nonempty_string(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or value.strip() == ""
        or "\n" in value
        or "\r" in value
    ):
        raise ContractError(f"{label} must be a non-empty string")
    return value


def _validate_string_map(value: object, label: str) -> None:
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a string-to-string object")
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise ContractError(f"{label} must be a string-to-string object")


def _validate_profiles(document: dict[str, object]) -> None:
    profiles = _object_field(document, "profiles", "profiles")
    for name in _PROFILE_NAMES:
        profile = profiles.get(name)
        if not isinstance(profile, dict):
            raise ContractError(f"profiles.{name} must be a JSON object")
        _validate_string_map(profile.get("headers"), f"profiles.{name}.headers")


def _validate_current_value(entry: object) -> None:
    # Null is the contract for captured request fields. A string is accepted
    # only when it is one of that header's allowed_values, which is how the
    # static session-type default is recorded.
    if not isinstance(entry, dict) or "current_value" not in entry:
        raise ContractError("request_attribution.headers current_value must be null")
    current = entry["current_value"]
    if current is None:
        return
    allowed = entry.get("allowed_values")
    if (
        isinstance(current, str)
        and isinstance(allowed, list)
        and all(isinstance(item, str) for item in allowed)
        and current in allowed
    ):
        return
    raise ContractError("request_attribution.headers current_value must be null")


def _validate_attribution(document: dict[str, object]) -> None:
    attribution = _object_field(document, "request_attribution", "request_attribution")
    headers = attribution.get("headers")
    if not isinstance(headers, dict):
        raise ContractError("request_attribution.headers must be a JSON object")
    for entry in headers.values():
        _validate_current_value(entry)


def _validate_source(document: dict[str, object]) -> None:
    source = _object_field(document, "source", "source")
    _nonempty_string(
        source.get("runtime_header_version"), "source.runtime_header_version"
    )
    digest = source.get("appimage_sha256")
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        raise ContractError(
            "source.appimage_sha256 must be 64 lowercase hex characters"
        )


def _validate_contract(document: dict[str, object]) -> None:
    if document.get("schema_version") != 2:
        raise ContractError("schema_version must be 2")
    if document.get("client") != "zcode":
        raise ContractError("client must be zcode")
    _validate_source(document)
    _nonempty_string(document.get("captured_at"), "captured_at")
    _validate_profiles(document)
    _validate_attribution(document)


def _has_nonempty_secret_field(node: object) -> bool:
    if isinstance(node, dict):
        for key, value in node.items():
            if key in _SECRET_FIELDS and isinstance(value, str) and value != "":
                return True
            if _has_nonempty_secret_field(value):
                return True
        return False
    if isinstance(node, list):
        return any(_has_nonempty_secret_field(item) for item in node)
    return False


def _summary_line(document: dict[str, object]) -> str:
    source = _object_field(document, "source", "source")
    version = _nonempty_string(
        source.get("runtime_header_version"), "source.runtime_header_version"
    )
    digest = source.get("appimage_sha256")
    captured = _nonempty_string(document.get("captured_at"), "captured_at")
    if not isinstance(digest, str):
        raise ContractError(
            "source.appimage_sha256 must be 64 lowercase hex characters"
        )
    return f"runtime_header_version={version} appimage_sha256={digest} captured_at={captured}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(_HASH_CHUNK_BYTES)
                if chunk == b"":
                    break
                digest.update(chunk)
    except OSError as exc:
        raise ContractError(f"appimage unreadable: {path}") from exc
    return digest.hexdigest()


def _compare_appimage(path: Path, document: dict[str, object]) -> int:
    if not path.is_file():
        raise ContractError(f"appimage absent: {path}")
    source = _object_field(document, "source", "source")
    expected = source.get("appimage_sha256")
    version = _nonempty_string(
        source.get("runtime_header_version"), "source.runtime_header_version"
    )
    if not isinstance(expected, str):
        raise ContractError(
            "source.appimage_sha256 must be 64 lowercase hex characters"
        )
    if _sha256_file(path) != expected:
        _write_line("A new static inspection is required before egress changes.")
        return 2
    _write_line(f"unchanged runtime_header_version={version}")
    return 0


def _atomic_replace(destination: Path, payload: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=".zcode-header-contract.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temp_path = Path(temp_name)
    replaced = False
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), 0o644)
        os.replace(temp_path, destination)
        replaced = True
        os.chmod(destination, 0o644)
    finally:
        if not replaced:
            try:
                temp_path.unlink()
            except OSError:
                pass


def _publish(source: Path, destination: Path) -> None:
    payload = _read_bytes(source, "publish source")
    document = _parse_object(payload)
    _validate_contract(document)
    if _has_nonempty_secret_field(document):
        raise ContractError("publish refused: non-empty secret field")
    try:
        _atomic_replace(destination, payload)
    except OSError as exc:
        raise ContractError("descriptor replace failed") from exc
    _write_line(_summary_line(document))


def _load_descriptor(path: Path) -> dict[str, object]:
    document = _parse_object(_read_bytes(path, "descriptor"))
    _validate_contract(document)
    return document


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check or publish the local ZCode native header contract."
    )
    parser.add_argument(
        "--descriptor",
        type=Path,
        help=(
            "Descriptor path. The default walks upward from cwd for "
            ".analysis/zai/20260922/zcode-native-header-contract.json."
        ),
    )
    parser.add_argument(
        "--appimage",
        type=Path,
        help="Hash this AppImage and compare it with source.appimage_sha256.",
    )
    parser.add_argument(
        "--publish",
        type=Path,
        help="Validate this JSON file and atomically replace --descriptor.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.appimage is not None and args.publish is not None:
            raise ContractError("pass only one of --appimage or --publish")
        descriptor = (
            args.descriptor if args.descriptor is not None else _default_descriptor()
        )
        if args.publish is not None:
            _publish(args.publish, descriptor)
            return 0
        document = _load_descriptor(descriptor)
        if args.appimage is not None:
            return _compare_appimage(args.appimage, document)
        _write_line(_summary_line(document))
        return 0
    except ContractError as exc:
        _write_line(str(exc), error=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
