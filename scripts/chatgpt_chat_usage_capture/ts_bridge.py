"""Typed bridge from the existing Python sidecar to the bounded TS worker.

The worker owns collection and pure transitions.  This module owns only child
supervision and PostgreSQL authority.  History reading stays disabled until the
authorized native history contract is separately accepted.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Optional
from uuid import uuid4

from .pg_collector_state import CollectorLease, PgCollectorState
from .pg_ledger import LedgerError, LedgerScope
from .privacy import assert_no_secrets, sanitize_token


PROTOCOL_VERSION = 1
MAX_FRAME_BYTES = 1024 * 1024
MAX_TOTAL_BYTES = 16 * 1024 * 1024
MAX_REQUESTS = 256
MAX_REQUEST_ID_LENGTH = 128
DEFAULT_CHILD_TTL_SECONDS = 3600


@dataclass(frozen=True)
class BridgeConfig:
    node_executable: str = os.environ.get(
        "AAWM_CHATGPT_ORACLE_NODE_EXECUTABLE", "node"
    )
    worker_script: str = "/app/scripts/chatgpt_chat_usage_capture/ts/dist/worker/main.js"
    max_frame_bytes: int = MAX_FRAME_BYTES
    max_total_bytes: int = MAX_TOTAL_BYTES
    max_requests: int = MAX_REQUESTS
    child_ttl_seconds: int = DEFAULT_CHILD_TTL_SECONDS

    @classmethod
    def from_runtime(cls, *, node_executable: str, worker_script: str) -> "BridgeConfig":
        return cls(
            node_executable=node_executable,
            worker_script=worker_script,
        )


class BridgeProtocolError(LedgerError):
    """The child returned a response that cannot be safely attributed."""


@dataclass
class _Child:
    process: subprocess.Popen[str]
    total_bytes: int
    request_count: int
    deadline_monotonic: float


class TsWorkerBridge:
    """One bounded worker child per collector account/profile run."""

    def __init__(self, state: PgCollectorState, config: BridgeConfig) -> None:
        self.state = state
        self.config = config
        self._children: dict[tuple[str, str], _Child] = {}
        self._lock = threading.Lock()

    def run_once(
        self,
        *,
        collector_account_id: str,
        profile_id: str,
        scope: LedgerScope,
        seen_at: Optional[datetime] = None,
    ) -> dict[str, Any]:
        account = _token(collector_account_id, "collector_account_id")
        profile = _token(profile_id, "profile_id")
        key = (account, profile)
        child = self._get_or_start_child(account, profile)
        run_id = str(uuid4())
        lease = self.state.claim_lease(
            collector_account_id=account,
            profile_id=profile,
            scope=scope,
            ttl_seconds=self.config.child_ttl_seconds,
        )
        self.state.ledger.initialize_binding(
            scope,
            seen_at=seen_at or datetime.now(timezone.utc),
        )
        header, _candidates = self.state.load_state(
            collector_account_id=account,
            profile_id=profile,
        )
        try:
            state_result = self._request(
                child,
                account=account,
                profile=profile,
                lease=lease,
                run_id=run_id,
                operation="loadState",
                payload={"kind": "schedule", "scope": {"collectorAccountId": account, "profileId": profile}},
            )
            if not state_result.get("ok"):
                return _error_result(state_result)
            native_reader = self._request(
                child,
                account=account,
                profile=profile,
                lease=lease,
                run_id=run_id,
                operation="readHistory",
                payload={"kind": "index", "capability": "native_history"},
            )
            if native_reader.get("ok"):
                raise BridgeProtocolError("native history was enabled without parent authorization")
            error = native_reader.get("error", {})
            if error.get("code") != "history_contract_unavailable":
                return _error_result(native_reader)
            finish = self._request(
                child,
                account=account,
                profile=profile,
                lease=lease,
                run_id=run_id,
                operation="finishRun",
            )
            if not finish.get("ok"):
                return _error_result(finish)
            self.state.finish_run(lease=lease, run_id=run_id)
            return {
                "runId": run_id,
                "coverageIncomplete": True,
                "historyContract": "unavailable",
                "stateVersion": header.state_version if header else None,
                "requests": child.request_count,
            }
        except BaseException:
            self.state.cancel(lease=lease, run_id=run_id)
            raise
        finally:
            self._maybe_reap(key, child)

    def _get_or_start_child(self, account: str, profile: str) -> _Child:
        with self._lock:
            child = self._children.get((account, profile))
            if child is not None and child.process.poll() is None:
                return child
            if child is not None:
                self._reap(key=(account, profile), child=child)
            deadline = time.monotonic() + self.config.child_ttl_seconds
            process = subprocess.Popen(
                [self.config.node_executable, self.config.worker_script],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            child = _Child(process, 0, 0, deadline)
            self._children[(account, profile)] = child
            return child

    def _request(
        self,
        child: _Child,
        *,
        account: str,
        profile: str,
        lease: CollectorLease,
        run_id: str,
        operation: str,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        if child.process.poll() is not None or time.monotonic() >= child.deadline_monotonic:
            raise BridgeProtocolError("worker child is no longer available")
        child.request_count += 1
        if child.request_count > self.config.max_requests:
            raise BridgeProtocolError("worker request bound exceeded")
        request_id = str(uuid4())
        request = {
            "protocolVersion": PROTOCOL_VERSION,
            "requestId": request_id,
            "runId": run_id,
            "collectorAccountId": account,
            "profileId": profile,
            "bindingGeneration": lease.binding.binding_generation,
            "leaseFencingToken": lease.lease_fencing_token,
            "operation": operation,
        }
        if payload is not None:
            request["payload"] = payload
        assert_no_secrets(request)
        encoded = json.dumps(request, separators=(",", ":"), default=str) + "\n"
        frame = encoded.encode("utf-8")
        if len(frame) > self.config.max_frame_bytes:
            raise BridgeProtocolError("worker request frame exceeds the configured bound")
        child.total_bytes += len(frame)
        if child.total_bytes > self.config.max_total_bytes:
            raise BridgeProtocolError("worker total byte bound exceeded")
        if child.process.stdin is None or child.process.stdout is None:
            raise BridgeProtocolError("worker child streams are unavailable")
        child.process.stdin.write(encoded)
        child.process.stdin.flush()
        line = child.process.stdout.readline()
        child.total_bytes += len(line.encode("utf-8"))
        if not line or len(line.encode("utf-8")) > self.config.max_frame_bytes:
            raise BridgeProtocolError("worker returned an invalid or oversized response")
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BridgeProtocolError("worker returned invalid JSON") from exc
        if not isinstance(response, Mapping):
            raise BridgeProtocolError("worker response is not an object")
        if (
            response.get("protocolVersion") != PROTOCOL_VERSION
            or response.get("requestId") != request_id
        ):
            raise BridgeProtocolError("worker response does not match its request")
        assert_no_secrets(response)
        return response

    def _maybe_reap(self, key: tuple[str, str], child: _Child) -> None:
        if child.process.poll() is None and time.monotonic() < child.deadline_monotonic:
            return
        self._reap(key=key, child=child)

    def _reap(self, *, key: tuple[str, str], child: _Child) -> None:
        process = child.process
        if process.poll() is None:
            _terminate_process_group(process)
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                _kill_process_group(process)
                process.wait(timeout=2)
        for stream in (process.stdin, process.stdout):
            if stream is not None:
                stream.close()
        self._children.pop(key, None)


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


def _kill_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _error_result(response: Mapping[str, Any]) -> dict[str, Any]:
    error = response.get("error", {})
    return {
        "ok": False,
        "errorCode": error.get("code", "history_reader_failed"),
        "retryable": bool(error.get("retryable", True)),
        "coverageIncomplete": bool(error.get("coverageIncomplete", True)),
    }


def _token(value: str, field_name: str) -> str:
    normalized = sanitize_token(value)
    if normalized is None:
        raise LedgerError(f"{field_name} is not a supported metadata token")
    return normalized
