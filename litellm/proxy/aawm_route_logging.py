import atexit
import ipaddress
import json
import logging
import os
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional, Union
from urllib.parse import parse_qsl, quote, urlencode, urlparse

import httpx
from fastapi import Request

from litellm._logging import (
    register_aawm_route_access_log_replacement,
    verbose_aawm_route_logger,
)

_AAWM_ROUTE_ACCESS_LOG_SCOPE_KEY = "aawm_route_access_log_emitted"
_AAWM_ROUTE_ACCESS_LOGGER_NAME = verbose_aawm_route_logger.name
_AAWM_ROUTE_ACCESS_LOG_TYPE = "ROUTE"
_AAWM_ROUTE_LOG_MAX_FIELD_CHARS = 180
_AAWM_ROUTE_LOG_MAX_IDENTITY_CHARS = 96
_AAWM_ROUTE_LOG_DEDUP_LIMIT = 4096
_AAWM_ROUTE_ROLLUP_INTERVAL_ENV = "AAWM_ROUTE_ROLLUP_INTERVAL_SECONDS"
_AAWM_ROUTE_ROLLUP_DEFAULT_INTERVAL_SECONDS = 60
_AAWM_ROUTE_ROLLUP_MAX_GROUPS = 256
_AAWM_ROUTE_ROLLUP_MAX_SUBLINES = 16
_AAWM_ROUTE_ROLLUP_CONTEXT_METADATA_KEY = "aawm_route_rollup_context"

_AAWM_ROUTE_HOST_REVERSE_DNS_TIMEOUT_SECONDS = 0.35
_AAWM_ROUTE_HOST_REVERSE_DNS_CACHE_TTL_SECONDS = 300.0
_AAWM_ROUTE_HOST_IP_LITERAL_CACHE_TTL_SECONDS = 60.0
_AAWM_ROUTE_HOST_REVERSE_DNS_CACHE_MAX_ENTRIES = 1024
_AAWM_ROUTE_HOST_LOOPBACK_LABEL = "localhost"
_AAWM_ROUTE_HOST_DOCKER_BRIDGE_NETWORKS = (
    ipaddress.ip_network("172.16.0.0/12"),
)
_AAWM_ROUTE_HOST_TAILSCALE_CGNAT_NETWORK = ipaddress.ip_network("100.64.0.0/10")
_AAWM_ROUTE_HOST_MAGICDNS_RESOLVER = "100.100.100.100"
_AAWM_ROUTE_HOST_MAGICDNS_PORT = 53
_aawm_route_host_reverse_dns_cache_lock = threading.Lock()
_aawm_route_host_reverse_dns_cache: dict[str, tuple[str, str, float]] = {}


def _clean_aawm_route_client_ip(value: Any) -> Optional[str]:
    cleaned = _clean_aawm_route_log_field(value)
    if not cleaned:
        return None
    if "," in cleaned:
        cleaned = cleaned.split(",", 1)[0].strip()
    try:
        return str(ipaddress.ip_address(cleaned))
    except ValueError:
        return cleaned


def _normalize_aawm_route_client_ip(value: Any) -> Optional[str]:
    cleaned = _clean_aawm_route_client_ip(value)
    if not cleaned:
        return None
    try:
        parsed = ipaddress.ip_address(cleaned)
    except ValueError:
        return cleaned
    return str(parsed)


def _is_aawm_route_local_display_ip(client_ip: Optional[str]) -> tuple[bool, str]:
    if not client_ip:
        return False, "unknown"
    if client_ip.lower() == _AAWM_ROUTE_HOST_LOOPBACK_LABEL:
        return True, "loopback"
    try:
        parsed = ipaddress.ip_address(client_ip)
    except ValueError:
        return False, "ip_literal"
    if parsed.is_loopback:
        return True, "loopback"
    if parsed.is_unspecified:
        return True, "unspecified"
    if parsed.is_link_local:
        return True, "link_local"
    if parsed.version == 4:
        octets = str(parsed).split(".")
        if (
            len(octets) == 4
            and octets[-1] == "1"
            and any(
                parsed in network
                for network in _AAWM_ROUTE_HOST_DOCKER_BRIDGE_NETWORKS
            )
        ):
            return True, "docker_bridge_gateway"
    return False, "remote"


def _hostname_label_from_reverse_lookup(hostname: str) -> Optional[str]:
    cleaned = _clean_aawm_route_log_field(hostname)
    if not cleaned:
        return None
    cleaned = cleaned.rstrip(".")
    if not cleaned:
        return None
    first_label = cleaned.split(".", 1)[0].strip()
    if not first_label:
        return cleaned
    if _is_aawm_route_log_slug(first_label):
        return first_label
    return cleaned


def _is_tailscale_cgnat_client_ip(client_ip: str) -> bool:
    try:
        parsed = ipaddress.ip_address(client_ip)
    except ValueError:
        return False
    return parsed.version == 4 and parsed in _AAWM_ROUTE_HOST_TAILSCALE_CGNAT_NETWORK


def _aawm_route_host_cache_ttl_for_source(source: str) -> float:
    if source == "ip_literal":
        return _AAWM_ROUTE_HOST_IP_LITERAL_CACHE_TTL_SECONDS
    return _AAWM_ROUTE_HOST_REVERSE_DNS_CACHE_TTL_SECONDS


def _aawm_route_host_cache_source_label(source: str) -> str:
    if source.endswith("_cache"):
        return source
    return f"{source}_cache"


def _aawm_route_host_ptr_qname(client_ip: str) -> Optional[str]:
    try:
        parsed = ipaddress.ip_address(client_ip)
    except ValueError:
        return None
    if parsed.version != 4:
        return None
    return ".".join(reversed(str(parsed).split("."))) + ".in-addr.arpa"


def _encode_dns_name(name: str) -> bytes:
    encoded = bytearray()
    for label in name.split("."):
        if not label:
            continue
        label_bytes = label.encode("idna")
        if len(label_bytes) > 63:
            raise ValueError("dns label too long")
        encoded.append(len(label_bytes))
        encoded.extend(label_bytes)
    encoded.append(0)
    return bytes(encoded)


def _decode_dns_name(data: bytes, offset: int) -> tuple[str, int]:
    labels: list[str] = []
    next_offset = offset
    jumped = False
    jumps = 0
    while offset < len(data):
        length = data[offset]
        if length == 0:
            offset += 1
            if not jumped:
                next_offset = offset
            break
        if length & 0xC0 == 0xC0:
            if offset + 1 >= len(data):
                break
            pointer = ((length & 0x3F) << 8) | data[offset + 1]
            jumps += 1
            if jumps > 16:
                break
            if not jumped:
                next_offset = offset + 2
                jumped = True
            offset = pointer
            continue
        offset += 1
        end = offset + length
        labels.append(data[offset:end].decode("ascii", errors="ignore"))
        offset = end
        if not jumped:
            next_offset = offset
    return ".".join(label for label in labels if label), next_offset


def _build_dns_ptr_query(qname: str, *, transaction_id: int = 0x1357) -> bytes:
    import random
    import struct

    tid = transaction_id if transaction_id is not None else random.randint(0, 65535)
    header = struct.pack("!HHHHHH", tid, 0x0100, 1, 0, 0, 0)
    query = _encode_dns_name(qname) + struct.pack("!HH", 12, 1)
    return header + query


def _extract_ptr_hostnames_from_dns_response(data: bytes) -> list[str]:
    import struct

    if len(data) < 12:
        return []
    _, _, qdcount, ancount, _, _ = struct.unpack("!HHHHHH", data[:12])
    offset = 12
    for _ in range(qdcount):
        _, offset = _decode_dns_name(data, offset)
        if offset + 4 > len(data):
            return []
        offset += 4
    hostnames: list[str] = []
    for _ in range(ancount):
        _, offset = _decode_dns_name(data, offset)
        if offset + 10 > len(data):
            break
        rtype, _, _, rdlength = struct.unpack("!HHIH", data[offset : offset + 10])
        offset += 10
        rdata_end = offset + rdlength
        if rdata_end > len(data):
            break
        if rtype in {5, 12}:
            hostname, _ = _decode_dns_name(data, offset)
            if hostname:
                hostnames.append(hostname.rstrip("."))
        offset = rdata_end
    return hostnames


def _resolve_hostname_via_magicdns(
    client_ip: str,
    *,
    resolver: str = _AAWM_ROUTE_HOST_MAGICDNS_RESOLVER,
    port: int = _AAWM_ROUTE_HOST_MAGICDNS_PORT,
    timeout_seconds: float = _AAWM_ROUTE_HOST_REVERSE_DNS_TIMEOUT_SECONDS,
) -> Optional[str]:
    qname = _aawm_route_host_ptr_qname(client_ip)
    if not qname:
        return None
    query = _build_dns_ptr_query(qname)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.settimeout(timeout_seconds)
        sock.sendto(query, (resolver, port))
        data, _ = sock.recvfrom(4096)
    except Exception:
        return None
    finally:
        sock.close()

    for hostname in _extract_ptr_hostnames_from_dns_response(data):
        resolved = _hostname_label_from_reverse_lookup(hostname)
        if resolved:
            return resolved
    return None


def _resolve_aawm_route_host_name_from_ip(
    client_ip: Optional[str],
    *,
    monotonic_now: Optional[float] = None,
) -> tuple[Optional[str], Optional[str]]:
    normalized_ip = _normalize_aawm_route_client_ip(client_ip)
    if not normalized_ip:
        return None, None
    is_local, local_source = _is_aawm_route_local_display_ip(normalized_ip)
    if is_local:
        return _AAWM_ROUTE_HOST_LOOPBACK_LABEL, local_source
    try:
        ipaddress.ip_address(normalized_ip)
    except ValueError:
        return normalized_ip, "ip_literal"

    now = time.monotonic() if monotonic_now is None else monotonic_now
    with _aawm_route_host_reverse_dns_cache_lock:
        cached = _aawm_route_host_reverse_dns_cache.get(normalized_ip)
        if cached is not None:
            host_name, cached_source, expires_at = cached
            if expires_at > now:
                return host_name, _aawm_route_host_cache_source_label(cached_source)
            _aawm_route_host_reverse_dns_cache.pop(normalized_ip, None)

    host_name = normalized_ip
    source = "ip_literal"
    previous_timeout = socket.getdefaulttimeout()
    try:
        socket.setdefaulttimeout(_AAWM_ROUTE_HOST_REVERSE_DNS_TIMEOUT_SECONDS)
        hostname, _, _ = socket.gethostbyaddr(normalized_ip)
        resolved = _hostname_label_from_reverse_lookup(hostname)
        if resolved:
            host_name = resolved
            source = "reverse_dns"
    except Exception:
        if _is_tailscale_cgnat_client_ip(normalized_ip):
            resolved = _resolve_hostname_via_magicdns(normalized_ip)
            if resolved:
                host_name = resolved
                source = "magicdns_reverse"
    finally:
        socket.setdefaulttimeout(previous_timeout)

    with _aawm_route_host_reverse_dns_cache_lock:
        if (
            len(_aawm_route_host_reverse_dns_cache)
            >= _AAWM_ROUTE_HOST_REVERSE_DNS_CACHE_MAX_ENTRIES
        ):
            _aawm_route_host_reverse_dns_cache.clear()
        _aawm_route_host_reverse_dns_cache[normalized_ip] = (
            host_name,
            source,
            now + _aawm_route_host_cache_ttl_for_source(source),
        )
    return host_name, source


def _extract_aawm_route_request_client_ip(
    request: Request,
    *,
    general_settings: Optional[dict[str, Any]] = None,
) -> tuple[Optional[str], Optional[str]]:
    from litellm.proxy.auth.ip_address_utils import IPAddressUtils

    raw_ip = IPAddressUtils.get_mcp_client_ip(request, general_settings)
    if not raw_ip:
        scope = getattr(request, "scope", None)
        client = scope.get("client") if isinstance(scope, dict) else None
        if isinstance(client, (list, tuple)) and client:
            raw_ip = str(client[0])
        elif getattr(request, "client", None) is not None:
            raw_ip = getattr(request.client, "host", None)
    normalized_ip = _normalize_aawm_route_client_ip(raw_ip)
    if not raw_ip:
        return normalized_ip, None
    headers = dict(getattr(request, "headers", {}) or {})
    x_forwarded_for = _get_case_insensitive_header_value(
        headers,
        ("x-forwarded-for",),
    )
    if x_forwarded_for and _clean_aawm_route_client_ip(x_forwarded_for) == normalized_ip:
        return normalized_ip, "x_forwarded_for"
    return normalized_ip, "request_client"


def resolve_aawm_route_host_attribution(
    request: Request,
    *,
    general_settings: Optional[dict[str, Any]] = None,
    client_ip: Optional[str] = None,
    client_ip_source: Optional[str] = None,
) -> dict[str, Optional[str]]:
    if client_ip is None:
        client_ip, client_ip_source = _extract_aawm_route_request_client_ip(
            request,
            general_settings=general_settings,
        )
    else:
        client_ip = _normalize_aawm_route_client_ip(client_ip)
    host_name, host_name_source = _resolve_aawm_route_host_name_from_ip(client_ip)
    return {
        "client_ip": client_ip,
        "client_ip_source": client_ip_source,
        "host_name": host_name,
        "host_name_source": host_name_source,
    }


def _format_aawm_route_client_version_label(
    client_product_label: Optional[str],
) -> Optional[str]:
    if not client_product_label:
        return None
    if "/" in client_product_label:
        client_name, client_version = client_product_label.split("/", 1)
        return (
            f"{_normalize_aawm_route_log_known_client_name(client_name)}"
            f"[{client_version}]"
        )
    return _normalize_aawm_route_log_known_client_name(client_product_label)


def build_aawm_route_repo_client_host_label(
    *,
    repository: Optional[str],
    client_product_label: Optional[str],
    host_name: Optional[str],
) -> Optional[str]:
    repository_label = _normalize_aawm_route_log_repository_label(repository)
    client_label = _format_aawm_route_client_version_label(client_product_label)
    host_label = _clean_aawm_route_log_field(host_name)
    segments: list[str] = []
    if repository_label and client_label:
        segments.append(f"{repository_label}#{client_label}")
    elif repository_label:
        segments.append(repository_label)
    elif client_label:
        segments.append(client_label)
    if not segments:
        return host_label
    base = segments[0]
    if host_label:
        return f"{base}@{host_label}"
    return base


_AAWM_ROUTE_ROLLUP_STATUS_VALUES = (
    "Degraded",
    "Cooling Down",
    "Failed",
    "Exhausted",
)
_aawm_route_rollup_lock = threading.Lock()
_aawm_route_rollup_accumulator: Optional["AawmRouteRollupAccumulator"] = None
_aawm_route_rollup_flush_stop = threading.Event()
_aawm_route_rollup_flush_thread: Optional[threading.Thread] = None
_aawm_route_rollup_flush_thread_lock = threading.Lock()
_aawm_route_rollup_flush_poll_seconds = 1.0
_aawm_route_rollup_monotonic_now: Callable[[], float] = time.monotonic
_aawm_route_log_dedup_lock = threading.Lock()
_aawm_route_log_dedup_seen: dict[tuple[str, ...], float] = {}
_aawm_route_log_dedup_order: deque[tuple[str, ...]] = deque()
_AAWM_ROUTE_LOG_SAFE_QUERY_KEYS = frozenset(
    {
        "alt",
        "api-version",
        "beta",
        "stream",
    }
)
_AAWM_ROUTE_LOG_AGENT_METADATA_KEYS = (
    "agent_name",
    "aawm_agent_name",
    "aawm_claude_agent_name",
)
_AAWM_ROUTE_LOG_AGENT_ID_METADATA_KEYS = (
    "agent_id",
    "aawm_agent_id",
    "aawm_claude_agent_id",
    "claude_agent_id",
    "codex_agent_id",
)
_AAWM_ROUTE_LOG_AGENT_HEADER_KEYS = (
    "x-aawm-agent-name",
    "x-litellm-agent-name",
    "x-agent-name",
)
_AAWM_ROUTE_LOG_AGENT_ID_HEADER_KEYS = (
    "x-aawm-agent-id",
    "x-litellm-agent-id",
    "x-agent-id",
    "x-claude-agent-id",
    "x-codex-agent-id",
)
_AAWM_ROUTE_LOG_REPOSITORY_METADATA_KEYS = (
    "repository",
    "aawm_repository",
    "source_repository",
    "repo",
    "repo_name",
    "repository_name",
    "git_repository",
    "vcs_repository",
    "workspace_root",
    "workspaceRoot",
    "project_root",
    "projectRoot",
    "root_path",
    "rootPath",
    "working_directory",
    "workingDirectory",
    "cwd_path",
    "cwdPath",
    "cwd_uri",
    "cwdUri",
    "aawm_claude_project",
)
_AAWM_ROUTE_LOG_REPOSITORY_HEADER_KEYS = (
    "x-aawm-repository",
    "x-litellm-repository",
    "x-repository",
    "x-git-repository",
)
_AAWM_ROUTE_LOG_REPOSITORY_TENANT_HEADER_KEYS = (
    "x-aawm-tenant-id",
    "x-litellm-tenant-id",
    "x-litellm-organization-id",
    "x-litellm-org-id",
    "x-organization-id",
    "x-org-id",
    "x-litellm-team-id",
    "x-team-id",
)
_AAWM_ROUTE_LOG_CLIENT_LABEL_METADATA_KEYS = (
    "client_name_version",
    "client_label",
    "aawm_client_label",
    "client_user_agent",
    "user_agent",
)
_AAWM_ROUTE_LOG_CLIENT_NAME_METADATA_KEYS = (
    "client_name",
    "aawm_client_name",
    "application_name",
    "app_name",
)
_AAWM_ROUTE_LOG_CLIENT_VERSION_METADATA_KEYS = (
    "client_version",
    "aawm_client_version",
    "application_version",
    "app_version",
)
_AAWM_ROUTE_LOG_CLIENT_LABEL_HEADER_KEYS = (
    "x-aawm-client",
    "x-litellm-client",
    "x-client",
    "x-client-name-version",
    "user-agent",
)
_AAWM_ROUTE_LOG_CLIENT_NAME_HEADER_KEYS = (
    "x-aawm-client-name",
    "x-litellm-client-name",
    "x-client-name",
)
_AAWM_ROUTE_LOG_CLIENT_VERSION_HEADER_KEYS = (
    "x-aawm-client-version",
    "x-litellm-client-version",
    "x-client-version",
)
_AAWM_ROUTE_LOG_MODEL_ALIAS_METADATA_KEYS = (
    "inbound_model_alias",
    "requested_model_alias",
    "model_alias_label",
    "anthropic_auto_agent_alias",
    "codex_auto_agent_alias",
)
_AAWM_ROUTE_LOG_SELECTED_MODEL_METADATA_KEYS = (
    "codex_auto_agent_selected_model",
    "anthropic_auto_agent_selected_model",
    "anthropic_adapter_model",
    "xai_oauth_public_model",
    "xai_oauth_upstream_model",
    "grok_model_override",
)
_AAWM_ROUTE_LOG_TOP_LEVEL_METADATA_KEYS = (
    _AAWM_ROUTE_LOG_AGENT_METADATA_KEYS
    + _AAWM_ROUTE_LOG_AGENT_ID_METADATA_KEYS
    + _AAWM_ROUTE_LOG_REPOSITORY_METADATA_KEYS
    + _AAWM_ROUTE_LOG_CLIENT_LABEL_METADATA_KEYS
    + _AAWM_ROUTE_LOG_CLIENT_NAME_METADATA_KEYS
    + _AAWM_ROUTE_LOG_CLIENT_VERSION_METADATA_KEYS
    + _AAWM_ROUTE_LOG_MODEL_ALIAS_METADATA_KEYS
    + _AAWM_ROUTE_LOG_SELECTED_MODEL_METADATA_KEYS
    + ("trace_name", "trace_user_id")
)
_AAWM_ROUTE_LOG_TENANT_REPOSITORY_FRAGMENTS = (
    "harness",
    "validation",
)
_AAWM_ROUTE_LOG_REPOSITORY_PLACEHOLDER_VALUES = {
    ".analysis",
    ".codex",
    "agent-ok",
    "deep",
    "docker-compose.yml",
    "fixture",
    "myapp",
    "nonexistent-worktree",
    "two",
    "wt",
    "wt-ops-xyz",
    "x",
}


def _get_aawm_route_log_dedup_window_seconds() -> float:
    raw_value = os.getenv("AAWM_ROUTE_LOG_DEDUP_WINDOW_SECONDS", "5")
    try:
        return max(0.0, float(raw_value))
    except Exception:
        return 5.0


def clear_aawm_route_log_dedup_state() -> None:
    with _aawm_route_log_dedup_lock:
        _aawm_route_log_dedup_seen.clear()
        _aawm_route_log_dedup_order.clear()


def get_aawm_route_rollup_interval_seconds() -> int:
    raw_value = os.getenv(
        _AAWM_ROUTE_ROLLUP_INTERVAL_ENV,
        str(_AAWM_ROUTE_ROLLUP_DEFAULT_INTERVAL_SECONDS),
    )
    try:
        return max(0, int(float(raw_value)))
    except Exception:
        return _AAWM_ROUTE_ROLLUP_DEFAULT_INTERVAL_SECONDS


def aawm_route_rollups_enabled() -> bool:
    return get_aawm_route_rollup_interval_seconds() > 0


def _normalize_aawm_route_rollup_status(status: Optional[str]) -> Optional[str]:
    if status is None:
        return None
    cleaned = " ".join(str(status).strip().split())
    if not cleaned:
        return None
    for allowed_status in _AAWM_ROUTE_ROLLUP_STATUS_VALUES:
        if cleaned.casefold() == allowed_status.casefold():
            return allowed_status
    return None


def _format_aawm_route_rollup_client_context_label(
    *,
    group_header_label: str,
    client_product_label: Optional[str],
    host_name: Optional[str] = None,
) -> Optional[str]:
    if "@" in group_header_label or "#" in group_header_label:
        cleaned = _clean_aawm_route_log_field(group_header_label)
        if cleaned:
            return cleaned
    return build_aawm_route_repo_client_host_label(
        repository=group_header_label,
        client_product_label=client_product_label,
        host_name=host_name,
    )


def build_aawm_route_rollup_group_header_label(
    *,
    repository: Optional[str],
    client_product_label: Optional[str],
    host_name: Optional[str] = None,
) -> Optional[str]:
    return build_aawm_route_repo_client_host_label(
        repository=repository,
        client_product_label=client_product_label,
        host_name=host_name,
    )


def _format_aawm_route_rollup_status_tag(status: Optional[str]) -> str:
    normalized_status = _normalize_aawm_route_rollup_status(status)
    if not normalized_status:
        return ""
    return f" [{normalized_status}]"


def _resolve_aawm_route_rollup_default_destination(
    incoming_endpoint: str,
) -> Optional[str]:
    parsed_endpoint = urlparse(incoming_endpoint)
    endpoint_path = parsed_endpoint.path
    if endpoint_path.startswith("/anthropic/"):
        return f"api.anthropic.com{endpoint_path.removeprefix('/anthropic')}"
    if endpoint_path.startswith("/openai_passthrough/"):
        provider_path = endpoint_path.removeprefix("/openai_passthrough/")
        if provider_path.startswith("v1/"):
            provider_path = provider_path.removeprefix("v1/")
        if provider_path:
            return f"chatgpt.com/backend-api/codex/{provider_path}"
    return None


def _resolve_aawm_route_rollup_redundant_destination(
    *,
    incoming_endpoint: str,
    sublines: list[tuple[str, int, Optional[str], str]],
) -> Optional[str]:
    default_destination = _resolve_aawm_route_rollup_default_destination(
        incoming_endpoint
    )
    if default_destination is not None:
        return default_destination
    if not sublines:
        return None

    destinations: set[str] = set()
    for _, _, _, outgoing_target in sublines:
        if not outgoing_target:
            continue
        destinations.add(outgoing_target)
        if len(destinations) > 1:
            return None
    if not destinations:
        return None
    return next(iter(destinations))


def _format_aawm_route_rollup_subline_destination_suffix(
    *,
    outgoing_target: str,
    common_destination: Optional[str],
) -> str:
    if not outgoing_target:
        return ""
    if common_destination is not None and outgoing_target == common_destination:
        return ""
    return f" -> {outgoing_target}"


def _format_aawm_route_rollup_lines(
    *,
    group_header_label: str,
    incoming_endpoint: str,
    sublines: list[tuple[str, int, Optional[str], str]],
    now: Optional[datetime] = None,
    early: bool = False,
) -> list[str]:
    timestamp = (now or datetime.now()).strftime("%Y%m%d %H:%M:%S")
    header_segments = [timestamp]
    if early:
        header_segments.append("[EARLY]")
    header_segments.extend(
        [
            group_header_label,
            incoming_endpoint,
        ]
    )
    lines = [" ".join(header_segments)]
    common_destination = _resolve_aawm_route_rollup_redundant_destination(
        incoming_endpoint=incoming_endpoint,
        sublines=sublines,
    )
    for model_label, turns, status, outgoing_target in sublines:
        lines.append(
            f" - {model_label} - Turns: {turns}"
            f"{_format_aawm_route_rollup_status_tag(status)}"
            f"{_format_aawm_route_rollup_subline_destination_suffix(outgoing_target=outgoing_target, common_destination=common_destination)}"
        )
    return lines


@dataclass
class _AawmRouteRollupSubline:
    turns: int = 0
    status: Optional[str] = None
    status_sequence: int = 0


@dataclass
class _AawmRouteRollupGroup:
    group_header_label: str
    incoming_endpoint: str
    sublines: dict[tuple[str, str], _AawmRouteRollupSubline] = field(
        default_factory=dict
    )
    subline_order: list[tuple[str, str]] = field(default_factory=list)
    event_sequence: int = 0

    def ordered_sublines(self) -> list[tuple[str, int, Optional[str], str]]:
        return [
            (
                subline_key[0],
                self.sublines[subline_key].turns,
                self.sublines[subline_key].status,
                subline_key[1],
            )
            for subline_key in self.subline_order
            if subline_key in self.sublines
        ]


class AawmRouteRollupAccumulator:
    def __init__(
        self,
        *,
        interval_seconds: Optional[int] = None,
        max_groups: int = _AAWM_ROUTE_ROLLUP_MAX_GROUPS,
        max_sublines: int = _AAWM_ROUTE_ROLLUP_MAX_SUBLINES,
    ) -> None:
        self._interval_seconds = (
            get_aawm_route_rollup_interval_seconds()
            if interval_seconds is None
            else max(0, int(interval_seconds))
        )
        self._max_groups = max(1, max_groups)
        self._max_sublines = max(1, max_sublines)
        self._groups: dict[tuple[str, str], _AawmRouteRollupGroup] = {}
        self._last_flush_monotonic = time.monotonic()

    def interval_seconds(self) -> int:
        return self._interval_seconds

    def enabled(self) -> bool:
        return self._interval_seconds > 0

    def clear(self) -> None:
        self._groups.clear()
        self._last_flush_monotonic = time.monotonic()

    def record(
        self,
        *,
        group_header_label: str,
        incoming_endpoint: str,
        outgoing_target: str,
        model_label: str,
        turns: int = 1,
        status: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> list[str]:
        if not self.enabled():
            return []

        cleaned_group_header = _clean_aawm_route_log_field(group_header_label)
        cleaned_incoming_endpoint = _clean_aawm_route_log_field(incoming_endpoint)
        cleaned_outgoing_target = _clean_aawm_route_log_field(outgoing_target)
        cleaned_model_label = _clean_aawm_route_log_field(model_label)
        if (
            not cleaned_group_header
            or not cleaned_incoming_endpoint
            or not cleaned_outgoing_target
            or not cleaned_model_label
        ):
            return []

        normalized_status = _normalize_aawm_route_rollup_status(status)
        emitted_lines: list[str] = []
        group_key = (
            cleaned_group_header,
            cleaned_incoming_endpoint,
        )
        group = self._groups.get(group_key)
        if group is None and len(self._groups) >= self._max_groups:
            emitted_lines.extend(self.flush(force=True, now=now, early=True))
            group = None
        if group is None:
            group = _AawmRouteRollupGroup(
                group_header_label=cleaned_group_header,
                incoming_endpoint=cleaned_incoming_endpoint,
            )
            self._groups[group_key] = group

        subline_key = (cleaned_model_label, cleaned_outgoing_target)
        subline = group.sublines.get(subline_key)
        if subline is None:
            if len(group.subline_order) >= self._max_sublines:
                emitted_lines.extend(
                    self._flush_group(group, now=now, early=True, remove=True)
                )
                group = _AawmRouteRollupGroup(
                    group_header_label=cleaned_group_header,
                    incoming_endpoint=cleaned_incoming_endpoint,
                )
                self._groups[group_key] = group
                subline = None

        if subline is None:
            subline = _AawmRouteRollupSubline()
            group.sublines[subline_key] = subline
            group.subline_order.append(subline_key)

        if turns > 0:
            subline.turns += turns
        if normalized_status is not None:
            group.event_sequence += 1
            subline.status = normalized_status
            subline.status_sequence = group.event_sequence

        emitted_lines.extend(self.flush_due(now=now))
        return emitted_lines

    def flush_due(
        self,
        *,
        now: Optional[datetime] = None,
        monotonic_now: Optional[float] = None,
    ) -> list[str]:
        if not self.enabled() or not self._groups:
            return []
        current_monotonic = (
            time.monotonic() if monotonic_now is None else monotonic_now
        )
        if current_monotonic - self._last_flush_monotonic < self._interval_seconds:
            return []
        return self.flush(force=True, now=now)

    def flush(
        self,
        *,
        force: bool = False,
        now: Optional[datetime] = None,
        early: bool = False,
    ) -> list[str]:
        if not self._groups:
            return []
        if not force and not self.enabled():
            return []

        emitted_lines: list[str] = []
        for group_key, group in list(self._groups.items()):
            emitted_lines.extend(
                self._flush_group(group, now=now, early=early, remove=True)
            )
            self._groups.pop(group_key, None)
        self._last_flush_monotonic = time.monotonic()
        return emitted_lines

    def _flush_group(
        self,
        group: _AawmRouteRollupGroup,
        *,
        now: Optional[datetime] = None,
        early: bool = False,
        remove: bool,
    ) -> list[str]:
        if not group.sublines:
            return []
        lines = _format_aawm_route_rollup_lines(
            group_header_label=group.group_header_label,
            incoming_endpoint=group.incoming_endpoint,
            sublines=group.ordered_sublines(),
            now=now,
            early=early,
        )
        if remove:
            group_key = (
                group.group_header_label,
                group.incoming_endpoint,
            )
            self._groups.pop(group_key, None)
        return lines


def get_aawm_route_rollup_accumulator() -> AawmRouteRollupAccumulator:
    global _aawm_route_rollup_accumulator
    with _aawm_route_rollup_lock:
        interval_seconds = get_aawm_route_rollup_interval_seconds()
        if (
            _aawm_route_rollup_accumulator is None
            or _aawm_route_rollup_accumulator.interval_seconds() != interval_seconds
        ):
            _aawm_route_rollup_accumulator = AawmRouteRollupAccumulator()
        accumulator = _aawm_route_rollup_accumulator
    _ensure_aawm_route_rollup_flush_worker()
    return accumulator


def clear_aawm_route_rollups() -> None:
    global _aawm_route_rollup_accumulator
    with _aawm_route_rollup_lock:
        if _aawm_route_rollup_accumulator is None:
            _aawm_route_rollup_accumulator = AawmRouteRollupAccumulator()
        else:
            _aawm_route_rollup_accumulator.clear()


def _set_aawm_route_rollup_monotonic_now_for_tests(
    monotonic_now: Optional[Callable[[], float]],
) -> None:
    global _aawm_route_rollup_monotonic_now
    _aawm_route_rollup_monotonic_now = (
        time.monotonic if monotonic_now is None else monotonic_now
    )


def _tick_aawm_route_rollup_interval_flush() -> None:
    if not aawm_route_rollups_enabled():
        return
    with _aawm_route_rollup_lock:
        accumulator = _aawm_route_rollup_accumulator
        if accumulator is None or not accumulator.enabled():
            return
        lines = accumulator.flush_due(monotonic_now=_aawm_route_rollup_monotonic_now())
    _emit_aawm_route_rollup_lines(lines)


def _aawm_route_rollup_flush_worker_main() -> None:
    while not _aawm_route_rollup_flush_stop.is_set():
        interval_seconds = get_aawm_route_rollup_interval_seconds()
        if interval_seconds <= 0:
            if _aawm_route_rollup_flush_stop.wait(timeout=_aawm_route_rollup_flush_poll_seconds):
                break
            continue
        _tick_aawm_route_rollup_interval_flush()
        if _aawm_route_rollup_flush_stop.wait(
            timeout=min(float(interval_seconds), _aawm_route_rollup_flush_poll_seconds)
        ):
            break


def _stop_aawm_route_rollup_flush_worker() -> None:
    global _aawm_route_rollup_flush_thread
    _aawm_route_rollup_flush_stop.set()
    with _aawm_route_rollup_flush_thread_lock:
        worker = _aawm_route_rollup_flush_thread
        _aawm_route_rollup_flush_thread = None
    if worker is not None and worker.is_alive() and worker is not threading.current_thread():
        worker.join(timeout=1.0)
    _aawm_route_rollup_flush_stop.clear()


def _ensure_aawm_route_rollup_flush_worker() -> None:
    global _aawm_route_rollup_flush_thread
    interval_seconds = get_aawm_route_rollup_interval_seconds()
    if interval_seconds <= 0:
        _stop_aawm_route_rollup_flush_worker()
        return
    with _aawm_route_rollup_flush_thread_lock:
        worker = _aawm_route_rollup_flush_thread
        if worker is not None and worker.is_alive():
            return
        _aawm_route_rollup_flush_stop.clear()
        _aawm_route_rollup_flush_thread = threading.Thread(
            target=_aawm_route_rollup_flush_worker_main,
            name="aawm-route-rollup-flush",
            daemon=True,
        )
        _aawm_route_rollup_flush_thread.start()


def flush_aawm_route_rollups(
    *,
    force: bool = True,
    now: Optional[datetime] = None,
    early: bool = False,
) -> list[str]:
    accumulator = get_aawm_route_rollup_accumulator()
    with _aawm_route_rollup_lock:
        return accumulator.flush(force=force, now=now, early=early)


def _emit_aawm_route_rollup_lines(lines: list[str]) -> None:
    if not lines:
        return
    logging.getLogger(_AAWM_ROUTE_ACCESS_LOGGER_NAME).info("%s", "\n".join(lines))


def emit_flush_aawm_route_rollups(
    *,
    force: bool = True,
    now: Optional[datetime] = None,
    early: bool = False,
) -> None:
    _emit_aawm_route_rollup_lines(
        flush_aawm_route_rollups(force=force, now=now, early=early)
    )


def _get_aawm_route_rollup_metadata(kwargs: Optional[dict]) -> Optional[dict[str, Any]]:
    if not isinstance(kwargs, dict):
        return None
    litellm_params = kwargs.get("litellm_params")
    if isinstance(litellm_params, dict):
        metadata = litellm_params.get("metadata")
        if isinstance(metadata, dict):
            return metadata
    metadata = kwargs.get("metadata")
    if isinstance(metadata, dict):
        return metadata
    return None


def _set_aawm_route_rollup_metadata(
    kwargs: Optional[dict],
) -> Optional[dict[str, Any]]:
    if not isinstance(kwargs, dict):
        return None
    litellm_params = kwargs.get("litellm_params")
    if not isinstance(litellm_params, dict):
        litellm_params = {}
        kwargs["litellm_params"] = litellm_params
    metadata = litellm_params.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        litellm_params["metadata"] = metadata
    return metadata


def _get_aawm_route_rollup_model_label(
    *,
    model_label: Optional[str],
) -> Optional[str]:
    if not model_label:
        return None
    return model_label


def build_aawm_route_rollup_context(
    *,
    request: Request,
    target: Union[str, httpx.URL],
    request_body: Optional[dict[str, Any]] = None,
    kwargs: Optional[dict] = None,
    route_type: Optional[str] = None,
) -> Optional[dict[str, Optional[str]]]:
    metadata = _extract_aawm_route_log_metadata(request_body, kwargs)
    headers = dict(getattr(request, "headers", {}) or {})
    client_product_label = _get_aawm_route_log_client_product_label(metadata, headers)
    repository = _get_aawm_route_log_codex_turn_repository(
        headers
    ) or _first_aawm_route_log_value(
        metadata,
        keys=_AAWM_ROUTE_LOG_REPOSITORY_METADATA_KEYS,
        normalizer=_normalize_aawm_route_log_repository_label,
    ) or _get_case_insensitive_header_value(
        headers,
        _AAWM_ROUTE_LOG_REPOSITORY_HEADER_KEYS,
        normalizer=_normalize_aawm_route_log_repository_label,
    ) or _get_aawm_route_log_trace_user_repository(
        metadata,
        headers,
    ) or _get_case_insensitive_header_value(
        headers,
        _AAWM_ROUTE_LOG_REPOSITORY_TENANT_HEADER_KEYS,
        normalizer=_normalize_aawm_route_log_tenant_repository_label,
    )
    host_attribution = resolve_aawm_route_host_attribution(
        request,
        client_ip=_clean_aawm_route_client_ip(metadata.get("client_ip"))
        or _clean_aawm_route_client_ip(metadata.get("requester_ip_address")),
        client_ip_source=_clean_aawm_route_log_field(metadata.get("client_ip_source")),
    )
    group_header_label = build_aawm_route_rollup_group_header_label(
        repository=repository,
        client_product_label=client_product_label,
        host_name=host_attribution.get("host_name"),
    )
    model_label = _get_aawm_route_rollup_model_label(
        model_label=_get_aawm_route_log_model_label(request_body, metadata),
    )
    incoming_endpoint = _safe_aawm_route_endpoint_label(request)
    outgoing_target = _safe_aawm_route_target_label(target)
    log_type = _normalize_aawm_route_log_type(route_type, incoming_endpoint)
    if not group_header_label or not model_label:
        return None
    return {
        "group_header_label": group_header_label,
        "incoming_endpoint": incoming_endpoint,
        "outgoing_target": outgoing_target,
        "model_label": model_label,
        "route_type": log_type,
        "client_ip": host_attribution.get("client_ip"),
        "client_ip_source": host_attribution.get("client_ip_source"),
        "host_name": host_attribution.get("host_name"),
        "host_name_source": host_attribution.get("host_name_source"),
    }


def attach_aawm_route_rollup_context(
    *,
    request: Request,
    target: Union[str, httpx.URL],
    request_body: Optional[dict[str, Any]] = None,
    kwargs: Optional[dict] = None,
    route_type: Optional[str] = None,
) -> Optional[dict[str, Optional[str]]]:
    context = build_aawm_route_rollup_context(
        request=request,
        target=target,
        request_body=request_body,
        kwargs=kwargs,
        route_type=route_type,
    )
    if context is None:
        return None
    metadata = _set_aawm_route_rollup_metadata(kwargs)
    if metadata is not None:
        metadata[_AAWM_ROUTE_ROLLUP_CONTEXT_METADATA_KEY] = context
        for key in ("client_ip", "client_ip_source", "host_name", "host_name_source"):
            value = context.get(key)
            if value:
                metadata[key] = value
    return context


def record_aawm_route_rollup(
    *,
    group_header_label: str,
    incoming_endpoint: str,
    outgoing_target: str,
    model_label: str,
    turns: int = 1,
    status: Optional[str] = None,
    now: Optional[datetime] = None,
) -> None:
    accumulator = get_aawm_route_rollup_accumulator()
    with _aawm_route_rollup_lock:
        lines = accumulator.record(
            group_header_label=group_header_label,
            incoming_endpoint=incoming_endpoint,
            outgoing_target=outgoing_target,
            model_label=model_label,
            turns=turns,
            status=status,
            now=now,
        )
    _emit_aawm_route_rollup_lines(lines)


def record_aawm_route_rollup_turn(
    kwargs: Optional[dict],
    *,
    turns: int = 1,
    now: Optional[datetime] = None,
) -> None:
    if not aawm_route_rollups_enabled():
        return
    metadata = _get_aawm_route_rollup_metadata(kwargs)
    if not isinstance(metadata, dict):
        return
    context = metadata.get(_AAWM_ROUTE_ROLLUP_CONTEXT_METADATA_KEY)
    if not isinstance(context, dict):
        return
    if metadata.get("aawm_route_rollup_turn_recorded"):
        return
    metadata["aawm_route_rollup_turn_recorded"] = True
    record_aawm_route_rollup(
        group_header_label=str(context.get("group_header_label") or ""),
        incoming_endpoint=str(context.get("incoming_endpoint") or ""),
        outgoing_target=str(context.get("outgoing_target") or ""),
        model_label=str(context.get("model_label") or ""),
        turns=turns,
        now=now,
    )


def emit_aawm_route_status_event(
    *,
    alias_model: Optional[str],
    model_label: Optional[str],
    status: str,
    message: Optional[str],
    now: Optional[datetime] = None,
) -> None:
    normalized_status = _normalize_aawm_route_rollup_status(status) or status
    alias = _clean_aawm_route_log_field(alias_model) or "unknown-alias"
    model = _clean_aawm_route_log_field(model_label) or "unknown-model"
    detail = _clean_aawm_route_log_field(message) or "no detail"
    timestamp = (now or datetime.now()).strftime("%Y%m%d %H:%M:%S")
    logging.getLogger(_AAWM_ROUTE_ACCESS_LOGGER_NAME).warning(
        "%s - %s: %s Status: %s - Message: %s",
        timestamp,
        alias,
        model,
        normalized_status,
        detail,
    )


def _flush_aawm_route_rollups_at_exit() -> None:
    try:
        emit_flush_aawm_route_rollups(force=True)
    except Exception:
        logging.getLogger(_AAWM_ROUTE_ACCESS_LOGGER_NAME).debug(
            "Failed to flush AAWM route rollups at process exit",
            exc_info=True,
        )


atexit.register(_flush_aawm_route_rollups_at_exit)


def _should_emit_aawm_route_access_log_key(key: tuple[str, ...]) -> bool:
    window_seconds = _get_aawm_route_log_dedup_window_seconds()
    if window_seconds <= 0:
        return True

    now = time.monotonic()
    with _aawm_route_log_dedup_lock:
        expiry = _aawm_route_log_dedup_seen.get(key)
        if expiry is not None and expiry > now:
            return False

        _aawm_route_log_dedup_seen[key] = now + window_seconds
        _aawm_route_log_dedup_order.append(key)
        while _aawm_route_log_dedup_order:
            oldest_key = _aawm_route_log_dedup_order[0]
            oldest_expiry = _aawm_route_log_dedup_seen.get(oldest_key)
            if (
                oldest_expiry is not None
                and oldest_expiry > now
                and len(_aawm_route_log_dedup_order) <= _AAWM_ROUTE_LOG_DEDUP_LIMIT
            ):
                break
            _aawm_route_log_dedup_order.popleft()
            if oldest_expiry is None or oldest_expiry <= now:
                _aawm_route_log_dedup_seen.pop(oldest_key, None)
        return True


def _normalize_aawm_route_log_type(
    route_type: Optional[str],
    incoming_endpoint: Optional[str] = None,
) -> str:
    route_type_label = (route_type or "").lower().strip()
    endpoint_label = (incoming_endpoint or "").lower()
    if route_type_label in {"aembedding", "embedding", "embeddings"}:
        return "EMBED"
    if route_type_label in {"arerank", "rerank"}:
        return "RERANK"
    if "/embeddings" in endpoint_label:
        return "EMBED"
    if "/rerank" in endpoint_label:
        return "RERANK"
    return _AAWM_ROUTE_ACCESS_LOG_TYPE


def _clean_aawm_route_log_field(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (dict, list, tuple, set)) or not isinstance(
        value, (str, int, float)
    ):
        return None

    cleaned = "".join(
        char if char.isprintable() and char not in "\r\n\t" else " "
        for char in str(value).strip()
    )
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        return None

    lower_cleaned = cleaned.lower()
    if lower_cleaned.startswith(("bearer ", "sk-", "pk-", "xai-", "ya29.")):
        return None

    if len(cleaned) > _AAWM_ROUTE_LOG_MAX_FIELD_CHARS:
        cleaned = cleaned[: _AAWM_ROUTE_LOG_MAX_FIELD_CHARS - 3] + "..."
    return cleaned


def _is_aawm_route_log_slug(value: str) -> bool:
    return bool(value) and any(char.isalnum() for char in value) and all(
        char.isalnum() or char in "._-" for char in value
    )


def _normalize_aawm_route_log_agent_label(value: Any) -> Optional[str]:
    cleaned = _clean_aawm_route_log_field(value)
    if not cleaned or len(cleaned) > _AAWM_ROUTE_LOG_MAX_IDENTITY_CHARS:
        return None

    if "://" in cleaned or any(char in cleaned for char in "@;,=:{}`[]<>|\\"):
        return None
    if cleaned.endswith(".") or len(cleaned.split()) > 6:
        return None
    if not any(char.isalnum() for char in cleaned):
        return None
    if not all(char.isalnum() or char in " ._/-" for char in cleaned):
        return None
    return cleaned


def _normalize_aawm_route_log_agent_id(value: Any) -> Optional[str]:
    cleaned = _clean_aawm_route_log_field(value)
    if not cleaned or len(cleaned) > _AAWM_ROUTE_LOG_MAX_IDENTITY_CHARS:
        return None
    if _is_aawm_route_log_slug(cleaned):
        return cleaned
    return None


def _normalize_aawm_route_log_repository_label(value: Any) -> Optional[str]:
    cleaned = _clean_aawm_route_log_field(value)
    if not cleaned or len(cleaned) > _AAWM_ROUTE_LOG_MAX_IDENTITY_CHARS:
        return None
    if any(char in cleaned for char in " @;,=:{}`[]<>|\\"):
        return None

    parsed = urlparse(cleaned)
    if parsed.scheme and parsed.path:
        cleaned = parsed.path

    if "/" in cleaned:
        path_parts = [part for part in cleaned.rstrip("/").split("/") if part]
        if not path_parts:
            return None
        if not cleaned.startswith("/") and len(path_parts) == 2:
            owner, repo = path_parts
            if _is_aawm_route_log_slug(owner) and _is_aawm_route_log_slug(repo):
                return f"{owner}/{repo}"
        cleaned = path_parts[-1]

    if _is_aawm_route_log_slug(cleaned):
        if cleaned.lower() in _AAWM_ROUTE_LOG_REPOSITORY_PLACEHOLDER_VALUES:
            return None
        return cleaned
    return None


def _normalize_aawm_route_log_tenant_repository_label(value: Any) -> Optional[str]:
    repository = _normalize_aawm_route_log_repository_label(value)
    if repository is None:
        return None

    normalized = repository.lower()
    if any(
        fragment in normalized
        for fragment in _AAWM_ROUTE_LOG_TENANT_REPOSITORY_FRAGMENTS
    ):
        return "litellm"
    if normalized.endswith("-dev") or "tenant" in normalized:
        return None
    return repository


def _normalize_aawm_route_log_known_client_name(name: str) -> str:
    normalized_name = name.lower().replace("_", "-")
    if normalized_name in {"claude", "claude-cli", "claude-code"}:
        return "Claude"
    if normalized_name in {"codex", "codex-cli", "codex-tui", "codex-cli-rs"}:
        return "Codex"
    if normalized_name in {"grok", "grok-build", "grok-pager"}:
        return "Grok"
    if normalized_name in {"gemini", "gemini-cli"}:
        return "Gemini"
    if normalized_name in {"opencode", "opencode-tui"}:
        return "OpenCode"
    if normalized_name in {"cursor", "cursor-cli"}:
        return "Cursor"
    return name


def _normalize_aawm_route_log_client_product(value: Any) -> Optional[str]:
    cleaned = _clean_aawm_route_log_field(value)
    if not cleaned or len(cleaned) > _AAWM_ROUTE_LOG_MAX_IDENTITY_CHARS:
        return None

    product = cleaned.split()[0].strip("()")
    if not product or any(char in product for char in " @;,=:{}`[]<>|\\"):
        return None
    if "/" in product:
        name, version = product.split("/", 1)
        if _is_aawm_route_log_slug(name) and _is_aawm_route_log_slug(version):
            return f"{_normalize_aawm_route_log_known_client_name(name)}/{version}"
        return None
    if _is_aawm_route_log_slug(product):
        return _normalize_aawm_route_log_known_client_name(product)
    return None


def _first_aawm_route_log_value(
    *sources: Optional[dict[str, Any]],
    keys: tuple[str, ...],
    normalizer: Callable[[Any], Optional[str]] = _clean_aawm_route_log_field,
) -> Optional[str]:
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in keys:
            value = normalizer(source.get(key))
            if value:
                return value
    return None


def _extract_aawm_route_log_metadata(
    request_body: Optional[dict[str, Any]],
    kwargs: Optional[dict],
) -> dict[str, Any]:
    body_metadata: dict[str, Any] = {}
    if isinstance(request_body, dict):
        for key in _AAWM_ROUTE_LOG_TOP_LEVEL_METADATA_KEYS:
            if key in request_body:
                body_metadata[key] = request_body[key]
        for metadata_key in ("litellm_metadata", "metadata"):
            metadata_value = request_body.get(metadata_key)
            if isinstance(metadata_value, dict):
                body_metadata.update(
                    {
                        key: value
                        for key, value in metadata_value.items()
                        if key in _AAWM_ROUTE_LOG_TOP_LEVEL_METADATA_KEYS
                    }
                )

    kwargs_metadata: dict[str, Any] = {}
    if isinstance(kwargs, dict):
        litellm_params = kwargs.get("litellm_params")
        if isinstance(litellm_params, dict):
            metadata = litellm_params.get("metadata")
            if isinstance(metadata, dict):
                kwargs_metadata = metadata
    return {**body_metadata, **kwargs_metadata}


def _get_case_insensitive_header_value(
    headers: Optional[dict[str, Any]],
    keys: tuple[str, ...],
    normalizer: Callable[[Any], Optional[str]] = _clean_aawm_route_log_field,
) -> Optional[str]:
    if not isinstance(headers, dict):
        return None
    normalized_headers = {str(key).lower(): value for key, value in headers.items()}
    for key in keys:
        value = normalizer(normalized_headers.get(key.lower()))
        if value:
            return value
    return None


def _get_aawm_route_log_codex_turn_repository(
    headers: Optional[dict[str, Any]],
) -> Optional[str]:
    raw_value = _get_case_insensitive_header_value(
        headers,
        ("x-codex-turn-metadata",),
        normalizer=_clean_aawm_route_log_field,
    )
    if not raw_value:
        return None
    try:
        parsed_value = json.loads(raw_value)
    except Exception:
        return None
    if not isinstance(parsed_value, dict):
        return None
    return _normalize_aawm_route_log_repository_label(
        parsed_value.get("project_path")
    )


def _get_aawm_route_log_client_product_label(
    metadata: dict[str, Any],
    headers: dict[str, Any],
) -> Optional[str]:
    direct_label = _first_aawm_route_log_value(
        metadata,
        keys=_AAWM_ROUTE_LOG_CLIENT_LABEL_METADATA_KEYS,
        normalizer=_normalize_aawm_route_log_client_product,
    ) or _get_case_insensitive_header_value(
        headers,
        _AAWM_ROUTE_LOG_CLIENT_LABEL_HEADER_KEYS,
        normalizer=_normalize_aawm_route_log_client_product,
    )
    if direct_label:
        return direct_label

    client_name = _first_aawm_route_log_value(
        metadata,
        keys=_AAWM_ROUTE_LOG_CLIENT_NAME_METADATA_KEYS,
        normalizer=_normalize_aawm_route_log_client_product,
    ) or _get_case_insensitive_header_value(
        headers,
        _AAWM_ROUTE_LOG_CLIENT_NAME_HEADER_KEYS,
        normalizer=_normalize_aawm_route_log_client_product,
    )
    if not client_name:
        return None

    client_version = _first_aawm_route_log_value(
        metadata,
        keys=_AAWM_ROUTE_LOG_CLIENT_VERSION_METADATA_KEYS,
        normalizer=_normalize_aawm_route_log_client_product,
    ) or _get_case_insensitive_header_value(
        headers,
        _AAWM_ROUTE_LOG_CLIENT_VERSION_HEADER_KEYS,
        normalizer=_normalize_aawm_route_log_client_product,
    )
    if client_version:
        return f"{client_name}/{client_version}"
    return client_name


def _safe_aawm_route_endpoint_label(request: Request) -> str:
    request_url = getattr(request, "url", None)
    parsed_url = urlparse(str(request_url or ""))
    path = parsed_url.path or getattr(request, "path", None) or "/"
    query_pairs = []
    for key, value in parse_qsl(parsed_url.query, keep_blank_values=True):
        normalized_key = key.lower()
        if normalized_key not in _AAWM_ROUTE_LOG_SAFE_QUERY_KEYS:
            continue
        safe_key = _clean_aawm_route_log_field(key)
        safe_value = _clean_aawm_route_log_field(value)
        if not safe_key or safe_value is None or "->" in safe_value:
            continue
        query_pairs.append((safe_key, safe_value))

    if not query_pairs:
        return path
    return f"{path}?{urlencode(query_pairs)}"


def _safe_aawm_route_target_label(target: Union[str, httpx.URL]) -> str:
    parsed_url = urlparse(str(target))
    hostname = parsed_url.hostname or "unknown-target"
    path = parsed_url.path or "/"
    return f"{hostname}{path}"


def _get_aawm_route_client_label(request: Request) -> Optional[str]:
    scope = getattr(request, "scope", None)
    client = scope.get("client") if isinstance(scope, dict) else None
    if isinstance(client, tuple) and len(client) >= 2:
        host = _clean_aawm_route_log_field(client[0])
        port = _clean_aawm_route_log_field(client[1])
        if host and port:
            return f"{host}:{port}"

    request_client = getattr(request, "client", None)
    host = _clean_aawm_route_log_field(getattr(request_client, "host", None))
    port = _clean_aawm_route_log_field(getattr(request_client, "port", None))
    if host and port:
        return f"{host}:{port}"
    return None


def _get_aawm_route_native_access_log_path(request: Request) -> Optional[str]:
    scope = getattr(request, "scope", None)
    if not isinstance(scope, dict):
        return None

    path = scope.get("path")
    if not isinstance(path, str):
        return None

    full_path = quote(path)
    query_string = scope.get("query_string")
    if not query_string:
        return full_path

    try:
        if isinstance(query_string, bytes):
            query_label = query_string.decode("ascii")
        else:
            query_label = str(query_string)
    except UnicodeDecodeError:
        return full_path

    return f"{full_path}?{query_label}"


def _register_aawm_route_access_log_replacement(request: Request) -> None:
    scope = getattr(request, "scope", None)
    if not isinstance(scope, dict):
        return

    client = scope.get("client")
    client_addr = None
    if isinstance(client, (list, tuple)) and len(client) >= 2:
        client_addr = f"{client[0]}:{client[1]}"

    register_aawm_route_access_log_replacement(
        client_addr=client_addr,
        method=str(scope.get("method") or getattr(request, "method", "") or ""),
        full_path=_get_aawm_route_native_access_log_path(request),
        http_version=str(scope.get("http_version") or ""),
    )


def _get_aawm_route_log_model_label(
    request_body: Optional[dict[str, Any]],
    metadata: dict[str, Any],
) -> Optional[str]:
    model = None
    if isinstance(request_body, dict):
        model = _clean_aawm_route_log_field(request_body.get("model"))
    if model is None:
        model = _first_aawm_route_log_value(
            metadata,
            keys=_AAWM_ROUTE_LOG_SELECTED_MODEL_METADATA_KEYS,
        )

    alias = _first_aawm_route_log_value(
        metadata,
        keys=_AAWM_ROUTE_LOG_MODEL_ALIAS_METADATA_KEYS,
    )
    if model and alias and alias != model:
        return f"{model}({alias})"
    return model or alias


def _get_aawm_route_log_context_label(
    *,
    agent_name: Optional[str],
    agent_id: Optional[str],
    repository: Optional[str],
    model_label: Optional[str],
) -> Optional[str]:
    agent_label = agent_name
    if agent_label and agent_id:
        agent_label = f"{agent_label}#{agent_id}"
    elif agent_id:
        agent_label = f"#{agent_id}"

    if agent_label and repository:
        owner_label = f"{agent_label}@{repository}"
    else:
        owner_label = agent_label or repository

    if owner_label and model_label:
        return f"{owner_label}.{model_label}"
    return owner_label or model_label


def _get_aawm_route_log_trace_user_repository(
    metadata: dict[str, Any],
    headers: dict[str, Any],
) -> Optional[str]:
    trace_name = _first_aawm_route_log_value(
        metadata,
        keys=("trace_name",),
    ) or _get_case_insensitive_header_value(
        headers,
        ("langfuse_trace_name",),
    )
    if not trace_name:
        return None

    normalized_trace_name = trace_name.lower()
    if not normalized_trace_name.startswith(
        ("claude-code", "codex", "grok-build", "grok")
    ):
        return None

    return _first_aawm_route_log_value(
        metadata,
        keys=("trace_user_id",),
        normalizer=_normalize_aawm_route_log_repository_label,
    ) or _get_case_insensitive_header_value(
        headers,
        ("langfuse_trace_user_id",),
        normalizer=_normalize_aawm_route_log_repository_label,
    )


def _build_aawm_route_access_log_line_and_key(
    *,
    request: Request,
    target: Union[str, httpx.URL],
    request_body: Optional[dict[str, Any]] = None,
    kwargs: Optional[dict] = None,
    route_type: Optional[str] = None,
    now: Optional[datetime] = None,
) -> tuple[str, tuple[str, ...]]:
    metadata = _extract_aawm_route_log_metadata(request_body, kwargs)
    headers = dict(getattr(request, "headers", {}) or {})
    client_product_label = _get_aawm_route_log_client_product_label(
        metadata,
        headers,
    )
    agent_name = _first_aawm_route_log_value(
        metadata,
        keys=_AAWM_ROUTE_LOG_AGENT_METADATA_KEYS,
        normalizer=_normalize_aawm_route_log_agent_label,
    ) or _get_case_insensitive_header_value(
        headers,
        _AAWM_ROUTE_LOG_AGENT_HEADER_KEYS,
        normalizer=_normalize_aawm_route_log_agent_label,
    )
    agent_id = _first_aawm_route_log_value(
        metadata,
        keys=_AAWM_ROUTE_LOG_AGENT_ID_METADATA_KEYS,
        normalizer=_normalize_aawm_route_log_agent_id,
    ) or _get_case_insensitive_header_value(
        headers,
        _AAWM_ROUTE_LOG_AGENT_ID_HEADER_KEYS,
        normalizer=_normalize_aawm_route_log_agent_id,
    )
    repository = _first_aawm_route_log_value(
        metadata,
        keys=_AAWM_ROUTE_LOG_REPOSITORY_METADATA_KEYS,
        normalizer=_normalize_aawm_route_log_repository_label,
    ) or _get_case_insensitive_header_value(
        headers,
        _AAWM_ROUTE_LOG_REPOSITORY_HEADER_KEYS,
        normalizer=_normalize_aawm_route_log_repository_label,
    ) or _get_aawm_route_log_trace_user_repository(
        metadata,
        headers,
    ) or _get_case_insensitive_header_value(
        headers,
        _AAWM_ROUTE_LOG_REPOSITORY_TENANT_HEADER_KEYS,
        normalizer=_normalize_aawm_route_log_tenant_repository_label,
    )
    model_label = _get_aawm_route_log_model_label(request_body, metadata)
    context_label = _get_aawm_route_log_context_label(
        agent_name=agent_name,
        agent_id=agent_id,
        repository=repository,
        model_label=model_label,
    )
    client_label = _get_aawm_route_client_label(request)
    method = _clean_aawm_route_log_field(getattr(request, "method", None)) or "REQUEST"
    incoming_endpoint = _safe_aawm_route_endpoint_label(request)
    log_type = _normalize_aawm_route_log_type(route_type, incoming_endpoint)
    outgoing_target = _safe_aawm_route_target_label(target)
    timestamp = (now or datetime.now()).strftime("%Y%m%d %H:%M:%S")

    segments: list[str] = [timestamp, f"[{log_type}]"]
    if client_product_label:
        segments.append(client_product_label)
    if context_label:
        segments.append(f"- {context_label}")
    segments.append(method)
    if client_label:
        segments.append(client_label)
    segments.append(f"{incoming_endpoint} -> {outgoing_target}")
    return " ".join(segments), (
        log_type,
        client_product_label or "",
        context_label or "",
        method,
        incoming_endpoint,
        outgoing_target,
    )


def build_aawm_route_access_log_line(
    *,
    request: Request,
    target: Union[str, httpx.URL],
    request_body: Optional[dict[str, Any]] = None,
    kwargs: Optional[dict] = None,
    route_type: Optional[str] = None,
    now: Optional[datetime] = None,
) -> str:
    line, _dedup_key = _build_aawm_route_access_log_line_and_key(
        request=request,
        target=target,
        request_body=request_body,
        kwargs=kwargs,
        route_type=route_type,
        now=now,
    )
    return line


def emit_aawm_route_access_log(
    *,
    request: Request,
    target: Union[str, httpx.URL],
    request_body: Optional[dict[str, Any]] = None,
    kwargs: Optional[dict] = None,
    route_type: Optional[str] = None,
    completed: bool = False,
) -> None:
    scope = getattr(request, "scope", None)
    if isinstance(scope, dict):
        if scope.get(_AAWM_ROUTE_ACCESS_LOG_SCOPE_KEY):
            _register_aawm_route_access_log_replacement(request)
            return
        scope[_AAWM_ROUTE_ACCESS_LOG_SCOPE_KEY] = True

    line, dedup_key = _build_aawm_route_access_log_line_and_key(
        request=request,
        target=target,
        request_body=request_body,
        kwargs=kwargs,
        route_type=route_type,
    )
    _register_aawm_route_access_log_replacement(request)
    if aawm_route_rollups_enabled():
        attach_aawm_route_rollup_context(
            request=request,
            target=target,
            request_body=request_body,
            kwargs=kwargs,
            route_type=route_type,
        )
        if completed:
            record_aawm_route_rollup_turn(kwargs)
        return
    if not _should_emit_aawm_route_access_log_key(dedup_key):
        return
    logging.getLogger(_AAWM_ROUTE_ACCESS_LOGGER_NAME).info("%s", line)
