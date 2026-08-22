"""Prefix-only Redis inspect via docker exec. Never FLUSHALL. Never write."""

from __future__ import annotations

import re
from typing import Any, Mapping

from hv2.docker_guard import assert_container_allowed, run_docker
from hv2.errors import HarnessError, ProtectedTargetError
from hv2.load_config import as_str_list

_USED_MEMORY = re.compile(r"used_memory:(\d+)")
_FORBIDDEN_REDIS_ARGV = ("FLUSHALL", "FLUSHDB", "SCRIPT", "CONFIG SET")


def _redis_spec(config: Mapping[str, Any]) -> dict[str, Any]:
    spec = config.get("redis") if isinstance(config.get("redis"), dict) else {}
    return spec


def snapshot_redis(config: Mapping[str, Any]) -> dict[str, Any]:
    spec = _redis_spec(config)
    container = str(spec.get("container") or "")
    namespace = str(spec.get("namespace") or "")
    if not container or not namespace:
        raise HarnessError("redis.container and redis.namespace must be set in targets.yaml")
    assert_container_allowed(container, config)
    if spec.get("never_flush") is False or spec.get("never_write") is False:
        raise HarnessError("redis never_flush/never_write must stay true")

    info_cmd = as_str_list(spec.get("info_command")) or ["INFO", "memory"]
    _assert_safe_redis(info_cmd)
    info_proc = run_docker(
        config,
        ["exec", container, "redis-cli", *info_cmd],
        container=container,
    )
    if info_proc.returncode != 0:
        raise HarnessError(
            f"redis INFO failed on {container}: {(info_proc.stderr or info_proc.stdout or '').strip()}"
        )
    info_text = info_proc.stdout or ""
    used = None
    match = _USED_MEMORY.search(info_text)
    if match:
        used = int(match.group(1))

    count = int(spec.get("scan_count") or 200)
    # redis-cli --scan is read-only. Prefix is required.
    scan_proc = run_docker(
        config,
        [
            "exec",
            container,
            "redis-cli",
            "--scan",
            "--count",
            str(count),
            "--pattern",
            f"{namespace}*",
        ],
        container=container,
    )
    if scan_proc.returncode != 0:
        raise HarnessError(
            f"redis SCAN failed on {container}: {(scan_proc.stderr or scan_proc.stdout or '').strip()}"
        )
    keys = [line for line in (scan_proc.stdout or "").splitlines() if line.strip()]
    ceilings = spec.get("ceilings") if isinstance(spec.get("ceilings"), dict) else {}
    warnings: list[str] = []
    failures: list[str] = []
    if used is not None:
        warn_mem = ceilings.get("warn_used_memory_bytes")
        fail_mem = ceilings.get("fail_used_memory_bytes")
        if fail_mem is not None and used >= int(fail_mem):
            failures.append(f"redis used_memory {used} >= fail ceiling {fail_mem}")
        elif warn_mem is not None and used >= int(warn_mem):
            warnings.append(f"redis used_memory {used} >= warn ceiling {warn_mem}")
    warn_keys = ceilings.get("warn_prefix_keys")
    fail_keys = ceilings.get("fail_prefix_keys")
    nkeys = len(keys)
    if fail_keys is not None and nkeys >= int(fail_keys):
        failures.append(f"redis prefix key count {nkeys} >= fail ceiling {fail_keys}")
    elif warn_keys is not None and nkeys >= int(warn_keys):
        warnings.append(f"redis prefix key count {nkeys} >= warn ceiling {warn_keys}")
    return {
        "ok": not failures,
        "failures": failures,
        "warnings": warnings,
        "container": container,
        "namespace": namespace,
        "used_memory_bytes": used,
        "prefix_key_count": nkeys,
        "sample_keys": keys[:20],
    }


def _assert_safe_redis(argv: list[str]) -> None:
    joined = " ".join(argv).upper()
    for needle in _FORBIDDEN_REDIS_ARGV:
        if needle in joined:
            raise ProtectedTargetError(f"refusing redis command containing {needle}")
