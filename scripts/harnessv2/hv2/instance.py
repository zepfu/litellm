"""Resolve --instance to a container name, then docker inspect for host port."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping
from urllib.parse import urlparse

from hv2.docker_guard import (
    assert_container_allowed,
    assert_host_port_allowed,
    run_docker,
)
from hv2.errors import PlanError
from hv2.load_config import as_str_list

_PORT_TOKEN = re.compile(r"--port(?:\s+|=)(\d+)")
_ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class ResolvedInstance:
    alias: str
    container: str
    host: str
    host_port: int
    container_port: int | None
    base_url: str
    inspect_env: dict[str, str] = field(default_factory=dict)
    running: bool = True


def resolve_container_name(raw: str | None, config: Mapping[str, Any]) -> str:
    default = str(config.get("default_instance") or "litellm-alpha")
    token = (raw or default).strip()
    if not token:
        raise PlanError("instance name is empty")
    assert_container_allowed(token, config)
    aliases = config.get("aliases") if isinstance(config.get("aliases"), dict) else {}
    if token in aliases:
        entry = aliases[token]
        if isinstance(entry, dict):
            if entry.get("enabled") is False:
                raise PlanError(
                    f"instance alias {token!r} is disabled in targets.yaml"
                )
            container = str(entry.get("container") or token)
        else:
            container = str(entry)
    else:
        container = token
        instances = (
            config.get("instances") if isinstance(config.get("instances"), dict) else {}
        )
        inst = instances.get(container)
        if isinstance(inst, dict) and inst.get("enabled") is False:
            raise PlanError(f"instance {container!r} is disabled in targets.yaml")
    assert_container_allowed(container, config)
    return container


def _parse_inspect(payload: Any) -> dict[str, Any]:
    if isinstance(payload, list) and payload:
        first = payload[0]
        if isinstance(first, dict):
            return first
    if isinstance(payload, dict):
        return payload
    raise PlanError("docker inspect returned no container object")


def _published_bindings(inspect_data: Mapping[str, Any]) -> list[tuple[int, str, int]]:
    """Return (container_port, host_ip, host_port) tuples."""

    network = inspect_data.get("NetworkSettings") or {}
    ports = network.get("Ports") or {}
    bindings: list[tuple[int, str, int]] = []
    if not isinstance(ports, dict):
        return bindings
    for container_key, hosts in ports.items():
        if not hosts:
            continue
        container_port_s = str(container_key).split("/", 1)[0]
        try:
            container_port = int(container_port_s)
        except ValueError:
            continue
        if not isinstance(hosts, list):
            continue
        for host in hosts:
            if not isinstance(host, dict):
                continue
            host_ip = str(host.get("HostIp") or "")
            try:
                host_port = int(host.get("HostPort"))
            except (TypeError, ValueError):
                continue
            bindings.append((container_port, host_ip, host_port))
    return bindings


def _cmd_port(inspect_data: Mapping[str, Any]) -> int | None:
    config = inspect_data.get("Config") or {}
    cmd = config.get("Cmd") or []
    joined = " ".join(str(item) for item in cmd)
    match = _PORT_TOKEN.search(joined)
    if match:
        return int(match.group(1))
    return None


def _prefer_binding(
    bindings: list[tuple[int, str, int]],
    *,
    preferred_container_port: int | None,
) -> tuple[int, str, int]:
    if not bindings:
        raise PlanError("container has no published host ports")
    ranked = list(bindings)
    if preferred_container_port is not None:
        matching = [row for row in ranked if row[0] == preferred_container_port]
        if matching:
            ranked = matching
    loopback = [
        row
        for row in ranked
        if row[1] in {"127.0.0.1", "0.0.0.0", "::", "::1", ""}
    ]
    if loopback:
        ranked = loopback
    # Prefer 127.0.0.1 over 0.0.0.0.
    loopback_exact = [row for row in ranked if row[1] in {"127.0.0.1", "::1"}]
    if loopback_exact:
        return loopback_exact[0]
    return ranked[0]


def _split_env_assignment(item: str) -> tuple[str, str] | None:
    if "=" not in item:
        return None
    key, value = item.split("=", 1)
    if not _ENV_KEY.match(key):
        return None
    return key, value


def _env_map(inspect_data: Mapping[str, Any]) -> dict[str, str]:
    config = inspect_data.get("Config") or {}
    out: dict[str, str] = {}
    for item in config.get("Env") or []:
        if not isinstance(item, str):
            continue
        parsed = _split_env_assignment(item)
        if parsed:
            out[parsed[0]] = parsed[1]
    # /usr/bin/env KEY=value … overlays beat image Env (alpha vs inherited dev).
    for source in (config.get("Cmd") or [], inspect_data.get("Args") or []):
        for item in source:
            if not isinstance(item, str):
                continue
            parsed = _split_env_assignment(item)
            if parsed:
                out[parsed[0]] = parsed[1]
    return out


def inspect_instance(
    container: str,
    config: Mapping[str, Any],
    *,
    inspect_payload: Mapping[str, Any] | None = None,
) -> ResolvedInstance:
    assert_container_allowed(container, config)
    if inspect_payload is None:
        proc = run_docker(
            config,
            ["inspect", container],
            container=container,
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            raise PlanError(
                f"docker inspect failed for {container!r}: {err or proc.returncode}"
            )
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise PlanError(f"docker inspect JSON was invalid for {container!r}") from exc
        inspect_data = _parse_inspect(payload)
    else:
        inspect_data = dict(inspect_payload)

    state = inspect_data.get("State") or {}
    running = bool(state.get("Running"))
    if not running:
        raise PlanError(f"container {container!r} is not running")

    instances = config.get("instances") if isinstance(config.get("instances"), dict) else {}
    inst_cfg = instances.get(container) if isinstance(instances.get(container), dict) else {}
    yaml_container_port = inst_cfg.get("container_port") if inst_cfg else None
    preferred = None
    if yaml_container_port is not None:
        preferred = int(yaml_container_port)
    else:
        preferred = _cmd_port(inspect_data)

    bindings = _published_bindings(inspect_data)
    container_port, _host_ip, host_port = _prefer_binding(
        bindings, preferred_container_port=preferred
    )
    assert_host_port_allowed(host_port, config)
    host = "127.0.0.1"
    env_keys = as_str_list(inst_cfg.get("inspect_env_keys") if inst_cfg else [])
    full_env = _env_map(inspect_data)
    inspect_env = {key: full_env[key] for key in env_keys if key in full_env}
    base_url = f"http://{host}:{host_port}"
    parsed = urlparse(base_url)
    if parsed.port is not None:
        assert_host_port_allowed(parsed.port, config)
    return ResolvedInstance(
        alias=container,
        container=container,
        host=host,
        host_port=host_port,
        container_port=container_port,
        base_url=base_url,
        inspect_env=inspect_env,
        running=running,
    )
