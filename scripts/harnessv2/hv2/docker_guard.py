"""Refuse protected LiteLLM containers and host ports.

Every Docker and HTTP helper must go through this module. There is no
HV2_ALLOW_PROD escape hatch.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from hv2.errors import InstanceError, ProtectedTargetError
from hv2.load_config import as_str_list, config_timeouts

PROTECTED_CONTAINER_MESSAGE = (
    "harness v2 refuses {name!r}: that container is protected "
    "(aawm-litellm / litellm-dev). Use litellm-alpha."
)
PROTECTED_PORT_MESSAGE = (
    "harness v2 refuses host port {port}: ports 4000 and 4001 are protected. "
    "Use litellm-alpha (published 4011)."
)


def protected_containers(config: Mapping[str, Any]) -> frozenset[str]:
    raw = as_str_list(config.get("protected_containers"))
    return frozenset(raw or ["aawm-litellm", "litellm-dev"])


def protected_ports(config: Mapping[str, Any]) -> frozenset[int]:
    raw = config.get("protected_ports") or []
    ports = [int(item) for item in raw] if raw else [4000, 4001]
    return frozenset(ports)


def assert_container_allowed(name: str, config: Mapping[str, Any]) -> None:
    if name in protected_containers(config):
        raise ProtectedTargetError(PROTECTED_CONTAINER_MESSAGE.format(name=name))


def assert_host_port_allowed(port: int, config: Mapping[str, Any]) -> None:
    if int(port) in protected_ports(config):
        raise ProtectedTargetError(PROTECTED_PORT_MESSAGE.format(port=int(port)))


def assert_url_allowed(url: str, config: Mapping[str, Any]) -> None:
    parsed = urlparse(url)
    if parsed.port is not None:
        assert_host_port_allowed(parsed.port, config)


def assert_argv_not_protected(
    argv: Sequence[str], config: Mapping[str, Any]
) -> None:
    blocked = protected_containers(config)
    for token in argv:
        if token in blocked:
            raise ProtectedTargetError(PROTECTED_CONTAINER_MESSAGE.format(name=token))


def run_docker(
    config: Mapping[str, Any],
    args: Sequence[str],
    *,
    container: str,
    capture_output: bool = True,
    check: bool = False,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a docker CLI command after refusing protected names."""

    assert_container_allowed(container, config)
    argv = ["docker", *[str(item) for item in args]]
    assert_argv_not_protected(argv, config)
    docker_timeout = (
        timeout if timeout is not None else config_timeouts(config)["docker_seconds"]
    )
    return subprocess.run(
        argv,
        text=True,
        capture_output=capture_output,
        check=check,
        timeout=docker_timeout,
    )


def inspect_container(container: str, config: Mapping[str, Any]) -> dict[str, Any]:
    result = run_docker(config, ["inspect", container], container=container)
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()
        raise InstanceError(f"docker inspect {container} failed: {err or 'no stderr'}")
    payload = json.loads(result.stdout or "[]")
    if isinstance(payload, list):
        if not payload:
            raise InstanceError(f"docker inspect {container} returned empty")
        first = payload[0]
        if not isinstance(first, dict):
            raise InstanceError(f"docker inspect {container} returned a non-object")
        return first
    if isinstance(payload, dict):
        return payload
    raise InstanceError(f"docker inspect {container} returned unexpected JSON")
