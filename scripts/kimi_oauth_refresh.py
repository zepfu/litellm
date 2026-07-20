#!/usr/bin/env python3
"""Refresh the shared Kimi Code OAuth credential in place."""

from __future__ import annotations

import argparse
import json
import math
import os
import stat
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from litellm.secret_managers.credential_error_sanitizer import (
    DEFAULT_SECRET_FIELD_NAMES,
    sanitize_credential_error_message,
)
from litellm.secret_managers.credential_file_metadata import (
    resolve_credential_file_metadata,
    snapshot_credential_file_metadata,
)
from litellm.secret_managers.credential_file_write import (
    write_and_publish_private_text,
)

# Keep these portable. Kimi Code itself owns this credential and lock layout.
DEFAULT_KIMI_OAUTH_AUTH_FILE = "~/.kimi-code/credentials/kimi-code.json"
DEFAULT_KIMI_OAUTH_LOCK_FILE = "~/.kimi-code/oauth/kimi-code"
DEFAULT_KIMI_OAUTH_HOST = "https://auth.kimi.com"
DEFAULT_KIMI_OAUTH_TOKEN_ENDPOINT = f"{DEFAULT_KIMI_OAUTH_HOST}/api/oauth/token"
DEFAULT_KIMI_OAUTH_CLIENT_ID = "17e5f671-d194-4dfb-9706-5516cb48c098"
DEFAULT_KIMI_OAUTH_SCOPE = "kimi-code"
DEFAULT_KIMI_OAUTH_REFRESH_MIN_SECONDS = 300
DEFAULT_KIMI_OAUTH_REFRESH_THRESHOLD_RATIO = 0.5
DEFAULT_KIMI_OAUTH_MAX_REFRESH_ATTEMPTS = 3
DEFAULT_KIMI_OAUTH_HTTP_TIMEOUT_SECONDS = 30.0
DEFAULT_KIMI_OAUTH_AUTH_FILE_MODE = 0o600
DEFAULT_KIMI_OAUTH_ERROR_MESSAGE_LIMIT = 500

# Kimi Code 0.27.0 uses proper-lockfile with these retry/staleness settings.
DEFAULT_KIMI_OAUTH_LOCK_RETRIES = 120
DEFAULT_KIMI_OAUTH_LOCK_RETRY_FACTOR = 1.0
DEFAULT_KIMI_OAUTH_LOCK_RETRY_MIN_SECONDS = 0.5
DEFAULT_KIMI_OAUTH_LOCK_RETRY_MAX_SECONDS = 1.0
DEFAULT_KIMI_OAUTH_LOCK_STALE_SECONDS = 5.0
DEFAULT_KIMI_OAUTH_LOCK_HEARTBEAT_SECONDS = 2.5

_SECRET_FIELD_NAMES = DEFAULT_SECRET_FIELD_NAMES


class KimiOAuthError(RuntimeError):
    """Base error for a local Kimi OAuth operation."""


class KimiOAuthTransportError(KimiOAuthError):
    """A timeout, DNS, or other transport failure."""


class KimiOAuthRetryableError(KimiOAuthError):
    """A transport-equivalent HTTP response eligible for bounded retry."""


class KimiOAuthAuthorizationError(KimiOAuthError):
    """A rejected refresh token requiring the shared credential to be revoked."""


class KimiOAuthLockError(KimiOAuthError):
    """Kimi Code's native credential lock could not be prepared or acquired."""


class KimiOAuthLockOwnershipError(KimiOAuthLockError):
    """The lock directory changed ownership while this process held it."""


@dataclass(frozen=True)
class KimiOAuthSummary:
    attempted: bool
    refreshed: bool
    skipped: bool
    auth_file: str
    scope: str
    expires_at: Optional[str] = None
    auth_degraded: bool = False
    error_class: Optional[str] = None
    error_message: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "attempted": self.attempted,
            "refreshed": self.refreshed,
            "skipped": self.skipped,
            "auth_file": self.auth_file,
            "scope": self.scope,
            "expires_at": self.expires_at,
            "auth_degraded": self.auth_degraded,
            "error_class": self.error_class,
            "error_message": self.error_message,
        }


class _KimiCodeLock:
    """Directory lock interoperable with Kimi Code's proper-lockfile layout."""

    def __init__(
        self,
        sentinel_path: Path,
        *,
        retries: int = DEFAULT_KIMI_OAUTH_LOCK_RETRIES,
        factor: float = DEFAULT_KIMI_OAUTH_LOCK_RETRY_FACTOR,
        min_delay_seconds: float = DEFAULT_KIMI_OAUTH_LOCK_RETRY_MIN_SECONDS,
        max_delay_seconds: float = DEFAULT_KIMI_OAUTH_LOCK_RETRY_MAX_SECONDS,
        stale_seconds: float = DEFAULT_KIMI_OAUTH_LOCK_STALE_SECONDS,
        heartbeat_seconds: float = DEFAULT_KIMI_OAUTH_LOCK_HEARTBEAT_SECONDS,
        retry_sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], float] = time.time,
    ) -> None:
        self.sentinel_path = sentinel_path
        self.lock_path = Path(f"{sentinel_path}.lock")
        self.retries = max(0, int(retries))
        self.factor = float(factor)
        self.min_delay_seconds = float(min_delay_seconds)
        self.max_delay_seconds = float(max_delay_seconds)
        self.stale_seconds = float(stale_seconds)
        self.heartbeat_seconds = float(heartbeat_seconds)
        self.retry_sleep = retry_sleep
        self.now = now
        self._identity: Optional[Tuple[int, int, int]] = None
        self._heartbeat_failure: Optional[KimiOAuthLockOwnershipError] = None
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: Optional[threading.Thread] = None

    def __enter__(self) -> "_KimiCodeLock":
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        try:
            self.release()
        except KimiOAuthLockError:
            if exc_type is None:
                raise
        return None

    def acquire(self) -> None:
        self._prepare_sentinel()
        collisions = 0
        while True:
            try:
                # mkdir is the atomic primitive used by proper-lockfile.
                self.lock_path.mkdir()
            except FileExistsError:
                collisions += 1
                if self._remove_stale_lock_directory():
                    if collisions > self.retries + 1:
                        raise KimiOAuthLockError(
                            "Kimi OAuth lock changed repeatedly while removing stale " "lock directories."
                        )
                    continue
                if collisions > self.retries:
                    raise KimiOAuthLockError(
                        "Timed out acquiring Kimi OAuth lock directory "
                        f"{self.lock_path} after {self.retries} retries."
                    )
                self.retry_sleep(self._retry_delay(collisions - 1))
                continue
            except OSError as exc:
                raise KimiOAuthLockError(f"Unable to create Kimi OAuth lock directory {self.lock_path}: {exc}") from exc

            self._identity = self._directory_identity()
            self.assert_owned()
            self._start_heartbeat()
            return

    def assert_owned(self) -> None:
        if self._heartbeat_failure is not None:
            raise self._heartbeat_failure
        if self._identity is None:
            raise KimiOAuthLockOwnershipError("Kimi OAuth lock was never acquired.")
        try:
            current = self._directory_identity()
        except OSError as exc:
            raise KimiOAuthLockOwnershipError(
                f"Unable to inspect Kimi OAuth lock directory {self.lock_path}: {exc}"
            ) from exc
        if current != self._identity:
            raise KimiOAuthLockOwnershipError("Kimi OAuth lock ownership changed while refresh was in progress.")

    def release(self) -> None:
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=1.0)
            if self._heartbeat_thread.is_alive():
                raise KimiOAuthLockOwnershipError("Kimi OAuth lock heartbeat did not stop cleanly.")
        self.assert_owned()
        try:
            self.lock_path.rmdir()
        except OSError as exc:
            raise KimiOAuthLockError(f"Unable to release Kimi OAuth lock directory {self.lock_path}: {exc}") from exc
        self._identity = None

    def _prepare_sentinel(self) -> None:
        try:
            self.sentinel_path.parent.mkdir(parents=True, exist_ok=True)
            flags = os.O_WRONLY | os.O_CREAT
            if hasattr(os, "O_CLOEXEC"):
                flags |= os.O_CLOEXEC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(
                str(self.sentinel_path),
                flags,
                DEFAULT_KIMI_OAUTH_AUTH_FILE_MODE,
            )
            os.close(fd)
            sentinel_stat = os.lstat(self.sentinel_path)
        except OSError as exc:
            raise KimiOAuthLockError(f"Unable to prepare Kimi OAuth lock sentinel {self.sentinel_path}: {exc}") from exc
        if not stat.S_ISREG(sentinel_stat.st_mode):
            raise KimiOAuthLockError(f"Kimi OAuth lock sentinel is not a regular file: {self.sentinel_path}")

    def _remove_stale_lock_directory(self) -> bool:
        try:
            lock_stat = os.lstat(self.lock_path)
        except FileNotFoundError:
            return True
        except OSError as exc:
            raise KimiOAuthLockError(f"Unable to inspect Kimi OAuth lock directory {self.lock_path}: {exc}") from exc
        if not stat.S_ISDIR(lock_stat.st_mode):
            return False
        if self.now() - lock_stat.st_mtime <= self.stale_seconds:
            return False
        try:
            self.lock_path.rmdir()
        except FileNotFoundError:
            return True
        except OSError:
            return False
        return True

    def _directory_identity(self) -> Tuple[int, int, int]:
        lock_stat = os.lstat(self.lock_path)
        if not stat.S_ISDIR(lock_stat.st_mode):
            raise KimiOAuthLockOwnershipError(f"Kimi OAuth lock path is no longer a directory: {self.lock_path}")
        return lock_stat.st_dev, lock_stat.st_ino, lock_stat.st_ctime_ns

    def _retry_delay(self, retry_index: int) -> float:
        delay = self.min_delay_seconds * (self.factor**retry_index)
        return min(max(delay, 0.0), self.max_delay_seconds)

    def _start_heartbeat(self) -> None:
        if self.heartbeat_seconds <= 0:
            raise KimiOAuthLockError("Kimi OAuth lock heartbeat interval must be positive.")
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat,
            name="kimi-oauth-lock-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _heartbeat(self) -> None:
        while not self._heartbeat_stop.wait(self.heartbeat_seconds):
            try:
                self.assert_owned()
                try:
                    os.utime(self.lock_path, None, follow_symlinks=False)
                except TypeError:
                    os.utime(self.lock_path, None)
                # utime advances ctime, so record the post-heartbeat identity
                # only after verifying that the directory was ours beforehand.
                self._identity = self._directory_identity()
            except KimiOAuthLockOwnershipError as exc:
                self._heartbeat_failure = exc
                return
            except OSError as exc:
                self._heartbeat_failure = KimiOAuthLockOwnershipError(
                    f"Unable to heartbeat Kimi OAuth lock directory {self.lock_path}: {exc}"
                )
                return


def refresh_kimi_oauth_auth_file(
    auth_file: str | Path = DEFAULT_KIMI_OAUTH_AUTH_FILE,
    *,
    scope: str = DEFAULT_KIMI_OAUTH_SCOPE,
    force: bool = False,
    lock_file: str | Path | None = None,
    token_endpoint: str = DEFAULT_KIMI_OAUTH_TOKEN_ENDPOINT,
    client_id: str = DEFAULT_KIMI_OAUTH_CLIENT_ID,
    http_timeout_seconds: float = DEFAULT_KIMI_OAUTH_HTTP_TIMEOUT_SECONDS,
    now: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
    lock_retry_sleep: Callable[[float], None] = time.sleep,
    lock_now: Callable[[], float] = time.time,
    lock_retries: int = DEFAULT_KIMI_OAUTH_LOCK_RETRIES,
    lock_heartbeat_interval_seconds: float = (DEFAULT_KIMI_OAUTH_LOCK_HEARTBEAT_SECONDS),
) -> Dict[str, Any]:
    """Refresh the shared Kimi Code OAuth credential when its lease is near expiry."""

    auth_path = Path(auth_file).expanduser()
    lock_sentinel = _resolve_lock_sentinel(lock_file)
    fallback_scope = _safe_scope(scope)
    try:
        # Force callers need this pre-wait snapshot only to coalesce a peer's
        # completed refresh. The authoritative credential is always re-read
        # after the native lock is acquired.
        before_wait = _read_credential_payload(auth_path)
    except Exception as exc:
        return _failed_summary(auth_path, fallback_scope, exc)

    try:
        with _kimi_code_lock(
            lock_sentinel,
            retries=lock_retries,
            heartbeat_seconds=lock_heartbeat_interval_seconds,
            retry_sleep=lock_retry_sleep,
            now=lock_now,
        ) as lock:
            credential = _read_credential_payload(auth_path)
            active_scope = _credential_scope(credential, fallback_scope)
            if force and _credential_refresh_state(credential) != _credential_refresh_state(before_wait):
                return _skipped_summary(auth_path, active_scope, credential)
            if not force and not _credential_needs_refresh(credential, now=now):
                return _skipped_summary(auth_path, active_scope, credential)

            try:
                refreshed = _refresh_credential(
                    credential,
                    token_endpoint=token_endpoint,
                    client_id=client_id,
                    fallback_scope=active_scope,
                    http_timeout_seconds=http_timeout_seconds,
                    now=now,
                    sleep=sleep,
                )
            except KimiOAuthAuthorizationError as exc:
                # A Kimi Code peer may have rotated the refresh token just
                # before this rejected request reached the authorization server.
                sleep(0.1)
                lock.assert_owned()
                peer_credential = _read_credential_payload(auth_path)
                if _refresh_token_changed(credential, peer_credential):
                    return KimiOAuthSummary(
                        attempted=True,
                        refreshed=False,
                        skipped=True,
                        auth_file=str(auth_path),
                        scope=_credential_scope(peer_credential, active_scope),
                        expires_at=_format_expires_at(peer_credential.get("expires_at")),
                    ).as_dict()
                lock.assert_owned()
                _write_credential_payload(
                    auth_path,
                    _revoked_tombstone(credential, fallback_scope=active_scope),
                )
                return _failed_summary(
                    auth_path,
                    active_scope,
                    exc,
                    attempted=True,
                    auth_degraded=True,
                )

            lock.assert_owned()
            _write_credential_payload(auth_path, refreshed)
            return KimiOAuthSummary(
                attempted=True,
                refreshed=True,
                skipped=False,
                auth_file=str(auth_path),
                scope=_credential_scope(refreshed, active_scope),
                expires_at=_format_expires_at(refreshed.get("expires_at")),
            ).as_dict()
    except Exception as exc:
        return _failed_summary(auth_path, fallback_scope, exc)


def _kimi_code_lock(
    sentinel_path: Path,
    **kwargs: Any,
) -> _KimiCodeLock:
    """Return a native Kimi Code lock without resolving the sentinel path."""

    return _KimiCodeLock(sentinel_path, **kwargs)


def _resolve_lock_sentinel(lock_file: str | Path | None) -> Path:
    # Do not call resolve(): Kimi Code configures proper-lockfile with
    # realpath=false, so lexical sentinel paths are the interoperability key.
    candidate = lock_file if lock_file is not None else DEFAULT_KIMI_OAUTH_LOCK_FILE
    return Path(candidate).expanduser()


def _read_credential_payload(auth_path: Path) -> Dict[str, Any]:
    try:
        file_stat = os.lstat(auth_path)
        if stat.S_ISLNK(file_stat.st_mode):
            raise KimiOAuthError(f"Refusing symlink Kimi OAuth credential: {auth_path}")
        flags = os.O_RDONLY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(str(auth_path), flags)
        with os.fdopen(fd, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except KimiOAuthError:
        raise
    except FileNotFoundError as exc:
        raise KimiOAuthError(f"Kimi OAuth credential file does not exist: {auth_path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise KimiOAuthError(f"Unable to read Kimi OAuth credential file: {exc}") from exc
    if not isinstance(payload, dict):
        raise KimiOAuthError("Kimi OAuth credential file must contain a JSON object.")
    return payload


def _write_credential_payload(auth_path: Path, credential: Mapping[str, Any]) -> None:
    snapshot = snapshot_credential_file_metadata(
        auth_path,
        default_mode=DEFAULT_KIMI_OAUTH_AUTH_FILE_MODE,
        refuse_symlink=True,
    )
    metadata = resolve_credential_file_metadata(
        auth_path,
        default_mode=DEFAULT_KIMI_OAUTH_AUTH_FILE_MODE,
        base_metadata=snapshot,
        refuse_symlink=True,
    )
    serialized = json.dumps(dict(credential), sort_keys=True, separators=(",", ":"))
    write_and_publish_private_text(
        auth_path,
        f"{serialized}\n",
        metadata=metadata,
        default_mode=DEFAULT_KIMI_OAUTH_AUTH_FILE_MODE,
        mkdir_parents=True,
    )


def _credential_needs_refresh(
    credential: Mapping[str, Any],
    *,
    now: Callable[[], float],
) -> bool:
    expires_at = _as_finite_number(credential.get("expires_at"))
    if expires_at is None:
        return True
    return expires_at - now() <= _refresh_threshold_seconds(credential.get("expires_in"))


def _refresh_threshold_seconds(expires_in: Any) -> float:
    lifetime = _as_finite_number(expires_in)
    if lifetime is None or lifetime <= 0:
        return float(DEFAULT_KIMI_OAUTH_REFRESH_MIN_SECONDS)
    return max(
        float(DEFAULT_KIMI_OAUTH_REFRESH_MIN_SECONDS),
        lifetime * DEFAULT_KIMI_OAUTH_REFRESH_THRESHOLD_RATIO,
    )


def _refresh_credential(
    credential: Mapping[str, Any],
    *,
    token_endpoint: str,
    client_id: str,
    fallback_scope: str,
    http_timeout_seconds: float,
    now: Callable[[], float],
    sleep: Callable[[float], None],
) -> Dict[str, Any]:
    refresh_token = credential.get("refresh_token")
    if not isinstance(refresh_token, str) or not refresh_token.strip():
        raise KimiOAuthError("Kimi OAuth credential is missing refresh_token.")

    for attempt in range(DEFAULT_KIMI_OAUTH_MAX_REFRESH_ATTEMPTS):
        try:
            response = _request_refresh_token(
                token_endpoint,
                client_id=client_id,
                refresh_token=refresh_token,
                http_timeout_seconds=http_timeout_seconds,
            )
            return _normalize_refreshed_credential(
                response,
                fallback_scope=fallback_scope,
                now=now,
            )
        except (KimiOAuthTransportError, KimiOAuthRetryableError):
            if attempt == DEFAULT_KIMI_OAUTH_MAX_REFRESH_ATTEMPTS - 1:
                raise
            sleep(float(2**attempt))

    raise AssertionError("bounded Kimi OAuth retry loop unexpectedly exhausted")


def _request_refresh_token(
    token_endpoint: str,
    *,
    client_id: str,
    refresh_token: str,
    http_timeout_seconds: float,
) -> Mapping[str, Any]:
    form = urllib_parse.urlencode(
        {
            "grant_type": "refresh_token",
            "client_id": client_id,
            "refresh_token": refresh_token,
        }
    ).encode("utf-8")
    request = urllib_request.Request(
        token_endpoint,
        data=form,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    try:
        with urllib_request.urlopen(request, timeout=http_timeout_seconds) as response:
            status = int(getattr(response, "status", 200))
            payload = _read_response_payload(response.read())
    except urllib_error.HTTPError as exc:
        payload = _read_http_error_payload(exc)
        raise _classify_http_failure(exc.code, payload) from exc
    except (urllib_error.URLError, TimeoutError, OSError) as exc:
        raise KimiOAuthTransportError(f"Kimi OAuth transport failure: {exc}") from exc
    if status < 200 or status >= 300:
        raise _classify_http_failure(status, payload)
    return payload


def _read_response_payload(raw: bytes) -> Mapping[str, Any]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise KimiOAuthError(f"Kimi OAuth token response was not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise KimiOAuthError("Kimi OAuth token response must contain a JSON object.")
    return payload


def _read_http_error_payload(exc: urllib_error.HTTPError) -> Mapping[str, Any]:
    try:
        return _read_response_payload(exc.read())
    except KimiOAuthError:
        return {}


def _classify_http_failure(status: int, payload: Mapping[str, Any]) -> KimiOAuthError:
    error_code = str(payload.get("error") or "").strip().lower()
    description = str(payload.get("error_description") or payload.get("message") or "").strip()
    detail = f"Kimi OAuth token request failed with HTTP {status}"
    if error_code:
        detail = f"{detail}: {error_code}"
    if description:
        detail = f"{detail}; {description}"
    if status in {401, 403} or error_code == "invalid_grant":
        return KimiOAuthAuthorizationError(detail)
    if status in {408, 425, 429} or status >= 500:
        return KimiOAuthRetryableError(detail)
    return KimiOAuthError(detail)


def _normalize_refreshed_credential(
    response: Mapping[str, Any],
    *,
    fallback_scope: str,
    now: Callable[[], float],
) -> Dict[str, Any]:
    access_token = response.get("access_token")
    refresh_token = response.get("refresh_token")
    expires_in = _as_finite_number(response.get("expires_in"))
    if not isinstance(access_token, str) or not access_token:
        raise KimiOAuthError("Kimi OAuth token response is missing access_token.")
    if not isinstance(refresh_token, str) or not refresh_token:
        raise KimiOAuthError("Kimi OAuth token response is missing refresh_token.")
    if expires_in is None or expires_in <= 0:
        raise KimiOAuthError("Kimi OAuth token response has invalid expires_in.")
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_at": now() + expires_in,
        "expires_in": _json_number(expires_in),
        "scope": _credential_scope(response, fallback_scope),
        "token_type": _safe_token_type(response.get("token_type")),
    }


def _revoked_tombstone(
    credential: Mapping[str, Any],
    *,
    fallback_scope: str,
) -> Dict[str, Any]:
    return {
        "access_token": "",
        "refresh_token": "",
        "expires_at": 0,
        "expires_in": 0,
        "scope": _credential_scope(credential, fallback_scope),
        "token_type": "Bearer",
    }


def _credential_refresh_state(credential: Mapping[str, Any]) -> Tuple[Any, Any, Any, Any]:
    return (
        credential.get("access_token"),
        credential.get("refresh_token"),
        credential.get("expires_at"),
        credential.get("expires_in"),
    )


def _refresh_token_changed(
    credential: Mapping[str, Any],
    peer_credential: Mapping[str, Any],
) -> bool:
    return credential.get("refresh_token") != peer_credential.get("refresh_token")


def _credential_scope(credential: Mapping[str, Any], fallback_scope: str) -> str:
    return _safe_scope(credential.get("scope"), default=fallback_scope)


def _safe_scope(value: Any, *, default: str = DEFAULT_KIMI_OAUTH_SCOPE) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def _safe_token_type(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "Bearer"


def _as_finite_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _json_number(value: float) -> int | float:
    return int(value) if value.is_integer() else value


def _format_expires_at(value: Any) -> Optional[str]:
    timestamp = _as_finite_number(value)
    if timestamp is None:
        return None
    try:
        return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    except (OverflowError, OSError, ValueError):
        return None


def _skipped_summary(
    auth_path: Path,
    scope: str,
    credential: Mapping[str, Any],
) -> Dict[str, Any]:
    return KimiOAuthSummary(
        attempted=False,
        refreshed=False,
        skipped=True,
        auth_file=str(auth_path),
        scope=scope,
        expires_at=_format_expires_at(credential.get("expires_at")),
    ).as_dict()


def _failed_summary(
    auth_path: Path,
    scope: str,
    exc: BaseException,
    *,
    attempted: bool = False,
    auth_degraded: bool = False,
) -> Dict[str, Any]:
    return KimiOAuthSummary(
        attempted=attempted,
        refreshed=False,
        skipped=False,
        auth_file=str(auth_path),
        scope=scope,
        auth_degraded=auth_degraded,
        error_class=exc.__class__.__name__,
        error_message=_redacted_error_message(exc),
    ).as_dict()


def _redacted_error_message(exc: BaseException) -> str:
    message = sanitize_credential_error_message(
        str(exc),
        field_names=_SECRET_FIELD_NAMES,
        limit=DEFAULT_KIMI_OAUTH_ERROR_MESSAGE_LIMIT,
    )
    return message


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refresh the shared Kimi Code OAuth credential in place.")
    parser.add_argument("--auth-file", default=DEFAULT_KIMI_OAUTH_AUTH_FILE)
    parser.add_argument(
        "--lock-file",
        default=DEFAULT_KIMI_OAUTH_LOCK_FILE,
        help="Kimi Code lock sentinel; the native lock directory is sentinel + '.lock'.",
    )
    parser.add_argument("--scope", default=DEFAULT_KIMI_OAUTH_SCOPE)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--token-endpoint", default=DEFAULT_KIMI_OAUTH_TOKEN_ENDPOINT)
    parser.add_argument("--client-id", default=DEFAULT_KIMI_OAUTH_CLIENT_ID)
    parser.add_argument(
        "--http-timeout-seconds",
        type=float,
        default=DEFAULT_KIMI_OAUTH_HTTP_TIMEOUT_SECONDS,
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_argument_parser().parse_args(argv)
    summary = refresh_kimi_oauth_auth_file(
        args.auth_file,
        scope=args.scope,
        force=args.force,
        lock_file=args.lock_file,
        token_endpoint=args.token_endpoint,
        client_id=args.client_id,
        http_timeout_seconds=args.http_timeout_seconds,
    )
    sys.stdout.write(f"{json.dumps(summary, sort_keys=True)}\n")
    return 1 if summary["error_class"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
