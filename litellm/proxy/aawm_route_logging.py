import atexit
import asyncio
import concurrent.futures
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
_AAWM_ROUTE_HOST_LOCAL_LOOKUP_CACHE_KEY = "__aawm_local_display_host__"
_AAWM_ROUTE_HOST_LOCAL_DISCOVERY_MAX_CANDIDATES = 4
_AAWM_ROUTE_HOST_LOCAL_DISCOVERY_TIMEOUT_SECONDS = 0.2
_AAWM_ROUTE_HOST_RESOLV_CONF_PATH = "/etc/resolv.conf"
_AAWM_ROUTE_HOST_FIB_TRIE_PATH = "/proc/net/fib_trie"
_AAWM_ROUTE_HOST_HOSTNAME_LOOKUP_PATHS = (
    "/etc/hostname",
    "/proc/sys/kernel/hostname",
    "/host/etc/hostname",
    "/host/proc/sys/kernel/hostname",
)
_AAWM_ROUTE_HOST_TAILSCALE_SELF_SNAPSHOT_DEFAULT_PATHS = (
    "/host/aawm/tailscale-self.json",
    "/app/.analysis/tailscale-self.json",
)
_AAWM_ROUTE_HOST_TAILSCALE_SELF_SNAPSHOT_ENV = (
    "AAWM_ROUTE_HOST_TAILSCALE_SELF_SNAPSHOT_PATH"
)
_AAWM_ROUTE_HOST_TAILSCALE_SELF_CACHE_KEY = (
    "__aawm_tailscale_self_display_host__"
)


def _aawm_route_host_tailscale_self_snapshot_paths() -> tuple[str, ...]:
    override = _clean_aawm_route_log_field(
        os.environ.get(_AAWM_ROUTE_HOST_TAILSCALE_SELF_SNAPSHOT_ENV)
    )
    if override:
        return (override,)
    return _AAWM_ROUTE_HOST_TAILSCALE_SELF_SNAPSHOT_DEFAULT_PATHS


def _parse_tailscale_self_snapshot_payload(
    payload: Any,
) -> Optional[dict[str, Any]]:
    if not isinstance(payload, dict):
        return None

    self_section = payload.get("Self")
    if not isinstance(self_section, dict):
        self_section = {}

    dns_name = (
        payload.get("self_dns_name")
        or payload.get("SelfDNSName")
        or self_section.get("DNSName")
    )
    dns_name = _clean_aawm_route_log_field(dns_name)
    if dns_name:
        dns_name = dns_name.rstrip(".")

    tailscale_ips_raw = (
        payload.get("self_tailscale_ips")
        or payload.get("SelfTailscaleIPs")
        or self_section.get("TailscaleIPs")
    )
    tailscale_ips: list[str] = []
    if isinstance(tailscale_ips_raw, (list, tuple)):
        for raw_ip in tailscale_ips_raw:
            normalized_ip = _normalize_aawm_route_client_ip(raw_ip)
            if normalized_ip and _is_tailscale_cgnat_client_ip(normalized_ip):
                if normalized_ip not in tailscale_ips:
                    tailscale_ips.append(normalized_ip)

    magic_dns_suffix = payload.get("magic_dns_suffix") or payload.get(
        "MagicDNSSuffix"
    )
    current_tailnet = payload.get("CurrentTailnet")
    if not magic_dns_suffix and isinstance(current_tailnet, dict):
        magic_dns_suffix = current_tailnet.get("MagicDNSSuffix")
    magic_dns_suffix = _clean_aawm_route_log_field(magic_dns_suffix)
    if magic_dns_suffix:
        magic_dns_suffix = magic_dns_suffix.rstrip(".")

    if not dns_name and not tailscale_ips:
        return None

    return {
        "dns_name": dns_name,
        "tailscale_ips": tailscale_ips,
        "magic_dns_suffix": magic_dns_suffix,
    }


def _hostname_label_from_tailscale_self_dns_name(dns_name: str) -> Optional[str]:
    cleaned = _clean_aawm_route_log_field(dns_name)
    if not cleaned:
        return None
    cleaned = cleaned.rstrip(".")
    if not _is_aawm_route_tailnet_domain(cleaned):
        return None
    return _hostname_label_from_reverse_lookup(cleaned)


def _load_tailscale_self_identity_snapshot(
    *,
    snapshot_paths: Optional[tuple[str, ...]] = None,
) -> Optional[dict[str, Any]]:
    paths = (
        snapshot_paths
        if snapshot_paths is not None
        else _aawm_route_host_tailscale_self_snapshot_paths()
    )
    for snapshot_path in paths:
        try:
            with open(snapshot_path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
        parsed = _parse_tailscale_self_snapshot_payload(payload)
        if parsed is not None:
            parsed["snapshot_path"] = snapshot_path
            return parsed
    return None


def _resolve_aawm_route_host_name_from_tailscale_self_identity(
    *,
    snapshot_paths: Optional[tuple[str, ...]] = None,
    monotonic_now: Optional[float] = None,
) -> Optional[tuple[str, str]]:
    now = time.monotonic() if monotonic_now is None else monotonic_now
    with _aawm_route_host_reverse_dns_cache_lock:
        cached = _aawm_route_host_reverse_dns_cache.get(
            _AAWM_ROUTE_HOST_TAILSCALE_SELF_CACHE_KEY
        )
        if cached is not None:
            host_name, cached_source, expires_at = cached
            if expires_at > now:
                return host_name, _aawm_route_host_local_cache_source_label(
                    cached_source
                )
            _aawm_route_host_reverse_dns_cache.pop(
                _AAWM_ROUTE_HOST_TAILSCALE_SELF_CACHE_KEY, None
            )

    snapshot = _load_tailscale_self_identity_snapshot(snapshot_paths=snapshot_paths)
    if snapshot is None:
        return None

    resolved_host: Optional[str] = None
    dns_name = snapshot.get("dns_name")
    if isinstance(dns_name, str) and dns_name:
        resolved_host = _hostname_label_from_tailscale_self_dns_name(dns_name)

    if not resolved_host:
        for candidate_ip in snapshot.get("tailscale_ips") or []:
            resolved_host = _resolve_hostname_via_magicdns(candidate_ip)
            if resolved_host:
                break

    if not resolved_host:
        return None

    with _aawm_route_host_reverse_dns_cache_lock:
        if (
            len(_aawm_route_host_reverse_dns_cache)
            >= _AAWM_ROUTE_HOST_REVERSE_DNS_CACHE_MAX_ENTRIES
        ):
            _aawm_route_host_reverse_dns_cache.clear()
        _aawm_route_host_reverse_dns_cache[
            _AAWM_ROUTE_HOST_TAILSCALE_SELF_CACHE_KEY
        ] = (
            resolved_host,
            "tailscale_self",
            now + _aawm_route_host_local_cache_ttl_for_source("tailscale_self"),
        )
    return resolved_host, "tailscale_self"


_AAWM_ROUTE_HOST_LOCAL_FALLBACK_SOURCES = {
    "loopback",
    "unspecified",
    "link_local",
    "docker_bridge_gateway",
}
_aawm_route_host_reverse_dns_cache_lock = threading.Lock()
_aawm_route_host_reverse_dns_cache: dict[str, tuple[str, str, float]] = {}

# Dedicated pools so hot-path callers never block the event loop on reverse DNS /
# MagicDNS, and so timeout wrappers never deadlock with enrichment workers.
_aawm_route_host_dns_lookup_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="aawm-host-dns-lookup",
)
_aawm_route_host_enrichment_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="aawm-host-dns-enrich",
)
_aawm_route_host_enrichment_lock = threading.Lock()
_aawm_route_host_enrichment_inflight: set[str] = set()
_aawm_route_host_enrichment_local_key = "__aawm_local_display_host_enrich__"


def _close_aawm_route_host_dns_executors() -> None:
    for executor in (
        _aawm_route_host_dns_lookup_executor,
        _aawm_route_host_enrichment_executor,
    ):
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            # Python <3.9 has no cancel_futures
            try:
                executor.shutdown(wait=False)
            except Exception:
                pass
        except Exception:
            pass


atexit.register(_close_aawm_route_host_dns_executors)


def _run_socket_call_with_timeout(
    fn: Callable[..., Any],
    timeout_seconds: float,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Run a blocking socket/NSS helper with a per-call timeout.

    Never mutates process-global ``socket.setdefaulttimeout``.
    """
    future = _aawm_route_host_dns_lookup_executor.submit(fn, *args, **kwargs)
    try:
        return future.result(timeout=timeout_seconds)
    except concurrent.futures.TimeoutError as exc:
        raise TimeoutError(
            f"socket call timed out after {timeout_seconds}s"
        ) from exc


def _gethostbyaddr_with_timeout(
    client_ip: str,
    *,
    timeout_seconds: float = _AAWM_ROUTE_HOST_REVERSE_DNS_TIMEOUT_SECONDS,
) -> Optional[str]:
    try:
        hostname, _, _ = _run_socket_call_with_timeout(
            socket.gethostbyaddr,
            timeout_seconds,
            client_ip,
        )
    except Exception:
        return None
    return hostname


def _getaddrinfo_with_timeout(
    name: str,
    *,
    timeout_seconds: float = _AAWM_ROUTE_HOST_LOCAL_DISCOVERY_TIMEOUT_SECONDS,
) -> list[Any]:
    try:
        return list(
            _run_socket_call_with_timeout(
                socket.getaddrinfo,
                timeout_seconds,
                name,
                None,
                socket.AF_INET,
                socket.SOCK_DGRAM,
            )
        )
    except Exception:
        return []


def _store_aawm_route_host_reverse_dns_cache(
    cache_key: str,
    host_name: str,
    source: str,
    *,
    now: float,
    ttl_for_source: Optional[Callable[[str], float]] = None,
) -> None:
    ttl_fn = ttl_for_source or _aawm_route_host_cache_ttl_for_source
    with _aawm_route_host_reverse_dns_cache_lock:
        if (
            len(_aawm_route_host_reverse_dns_cache)
            >= _AAWM_ROUTE_HOST_REVERSE_DNS_CACHE_MAX_ENTRIES
        ):
            _aawm_route_host_reverse_dns_cache.clear()
        _aawm_route_host_reverse_dns_cache[cache_key] = (
            host_name,
            source,
            now + ttl_fn(source),
        )


def _schedule_aawm_route_host_name_enrichment(
    client_ip: str,
    *,
    local_source: Optional[str] = None,
) -> None:
    """Background-enrich reverse DNS / MagicDNS without blocking the caller."""
    if local_source is not None:
        work_key = f"{_aawm_route_host_enrichment_local_key}:{local_source}"
    else:
        normalized = _normalize_aawm_route_client_ip(client_ip)
        if not normalized:
            return
        work_key = normalized
        client_ip = normalized

    with _aawm_route_host_enrichment_lock:
        if work_key in _aawm_route_host_enrichment_inflight:
            return
        _aawm_route_host_enrichment_inflight.add(work_key)

    def _work() -> None:
        try:
            if local_source is not None:
                _resolve_aawm_route_local_display_host(
                    local_source=local_source,
                    allow_blocking_lookup=True,
                )
            else:
                _resolve_aawm_route_host_name_from_ip(
                    client_ip,
                    allow_blocking_lookup=True,
                )
        except Exception:
            pass
        finally:
            with _aawm_route_host_enrichment_lock:
                _aawm_route_host_enrichment_inflight.discard(work_key)

    try:
        _aawm_route_host_enrichment_executor.submit(_work)
    except RuntimeError:
        with _aawm_route_host_enrichment_lock:
            _aawm_route_host_enrichment_inflight.discard(work_key)


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


def _aawm_route_host_local_cache_ttl_for_source(source: str) -> float:
    if source in _AAWM_ROUTE_HOST_LOCAL_FALLBACK_SOURCES:
        return _AAWM_ROUTE_HOST_IP_LITERAL_CACHE_TTL_SECONDS
    return _AAWM_ROUTE_HOST_REVERSE_DNS_CACHE_TTL_SECONDS


def _aawm_route_host_local_cache_key(local_source: str) -> str:
    return f"{_AAWM_ROUTE_HOST_LOCAL_LOOKUP_CACHE_KEY}:{local_source}"


def _aawm_route_host_local_cache_source_label(source: str) -> str:
    if source in _AAWM_ROUTE_HOST_LOCAL_FALLBACK_SOURCES:
        return source
    return _aawm_route_host_cache_source_label(source)


def _is_aawm_route_tailnet_domain(value: str) -> bool:
    lowered = value.rstrip(".").lower()
    return (
        lowered == "ts.net"
        or lowered.endswith(".ts.net")
        or lowered == "tailscale.net"
        or lowered.endswith(".tailscale.net")
    )


def _tailnet_search_domains_from_resolv_conf(
    *,
    resolv_conf_path: str = _AAWM_ROUTE_HOST_RESOLV_CONF_PATH,
    max_domains: int = 8,
) -> list[str]:
    domains: list[str] = []
    try:
        with open(resolv_conf_path, encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or stripped.startswith(";"):
                    continue
                lower = stripped.lower()
                if lower.startswith("search "):
                    tokens = stripped.split()[1:]
                elif lower.startswith("domain "):
                    tokens = stripped.split()[1:2]
                else:
                    continue
                for token in tokens:
                    cleaned = token.strip().rstrip(".")
                    if not cleaned or cleaned in domains:
                        continue
                    if _is_aawm_route_tailnet_domain(cleaned):
                        domains.append(cleaned)
                    if len(domains) >= max_domains:
                        return domains
    except OSError:
        return domains
    return domains


def _parse_proc_net_fib_trie_tailscale_local_ips(
    *,
    fib_trie_path: str = _AAWM_ROUTE_HOST_FIB_TRIE_PATH,
    max_candidates: int = _AAWM_ROUTE_HOST_LOCAL_DISCOVERY_MAX_CANDIDATES,
) -> list[str]:
    import re

    candidates: list[str] = []
    seen: set[str] = set()
    current_ip: Optional[str] = None
    try:
        with open(fib_trie_path, encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                match = re.match(r"\s*\|--\s+(\d+\.\d+\.\d+\.\d+)", line)
                if match:
                    current_ip = match.group(1)
                    continue
                if current_ip is None or "host LOCAL" not in line:
                    continue
                try:
                    parsed = ipaddress.ip_address(current_ip)
                except ValueError:
                    current_ip = None
                    continue
                if (
                    parsed.version == 4
                    and parsed in _AAWM_ROUTE_HOST_TAILSCALE_CGNAT_NETWORK
                    and current_ip not in seen
                ):
                    candidates.append(current_ip)
                    seen.add(current_ip)
                    if len(candidates) >= max_candidates:
                        break
                current_ip = None
    except OSError:
        return candidates
    return candidates


def _hostname_lookup_file_candidates(
    *,
    lookup_paths: tuple[str, ...] = _AAWM_ROUTE_HOST_HOSTNAME_LOOKUP_PATHS,
    max_candidates: int = _AAWM_ROUTE_HOST_LOCAL_DISCOVERY_MAX_CANDIDATES,
) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for path in lookup_paths:
        try:
            with open(path, encoding="utf-8", errors="ignore") as handle:
                raw_value = handle.readline()
        except OSError:
            continue
        cleaned = _clean_aawm_route_log_field(raw_value)
        normalized = cleaned.rstrip(".") if cleaned else None
        if (
            not normalized
            or normalized.lower() in {"localhost", "localhost.localdomain"}
            or normalized in seen
        ):
            continue
        candidates.append(normalized)
        seen.add(normalized)
        if len(candidates) >= max_candidates:
            break
    return candidates


def _local_tailscale_hostname_lookup_names() -> list[str]:
    try:
        host_name = _clean_aawm_route_log_field(socket.gethostname())
    except OSError:
        host_name = None
    candidate_names: list[str] = []
    seen_names: set[str] = set()
    try:
        fqdn = socket.getfqdn()
    except OSError:
        fqdn = None
    for raw_name in (fqdn, host_name):
        cleaned = _clean_aawm_route_log_field(raw_name)
        normalized_name = cleaned.rstrip(".") if cleaned else None
        if normalized_name and normalized_name not in seen_names:
            candidate_names.append(normalized_name)
            seen_names.add(normalized_name)
    for file_host_name in _hostname_lookup_file_candidates():
        if file_host_name not in seen_names:
            candidate_names.append(file_host_name)
            seen_names.add(file_host_name)
    for domain in _tailnet_search_domains_from_resolv_conf():
        for base_name in list(candidate_names):
            if "." in base_name:
                continue
            fqdn = f"{base_name}.{domain}".rstrip(".")
            if fqdn not in seen_names:
                candidate_names.append(fqdn)
                seen_names.add(fqdn)
    return candidate_names


def _discover_hostname_tailscale_ipv4_candidates(
    *,
    max_candidates: int = _AAWM_ROUTE_HOST_LOCAL_DISCOVERY_MAX_CANDIDATES,
    timeout_seconds: float = _AAWM_ROUTE_HOST_LOCAL_DISCOVERY_TIMEOUT_SECONDS,
) -> list[str]:
    candidate_names = _local_tailscale_hostname_lookup_names()
    candidates: list[str] = []
    seen_ips: set[str] = set()
    # Resolver-local timeouts only — never mutate process-global
    # socket.setdefaulttimeout (RR-041 / RR-049 / D1-529).
    for name in candidate_names:
        infos = _getaddrinfo_with_timeout(name, timeout_seconds=timeout_seconds)
        for info in infos:
            sockaddr = info[4]
            if not isinstance(sockaddr, tuple) or not sockaddr:
                continue
            ip = str(sockaddr[0])
            if not _is_tailscale_cgnat_client_ip(ip) or ip in seen_ips:
                continue
            candidates.append(ip)
            seen_ips.add(ip)
            if len(candidates) >= max_candidates:
                return candidates
        if _is_aawm_route_tailnet_domain(name):
            for ip in _resolve_ipv4_via_magicdns(name):
                if ip in seen_ips:
                    continue
                candidates.append(ip)
                seen_ips.add(ip)
                if len(candidates) >= max_candidates:
                    return candidates
    return candidates


def _discover_local_tailscale_ipv4_candidates(
    *,
    max_candidates: int = _AAWM_ROUTE_HOST_LOCAL_DISCOVERY_MAX_CANDIDATES,
) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for ip in _parse_proc_net_fib_trie_tailscale_local_ips(
        max_candidates=max_candidates,
    ):
        if ip not in seen:
            candidates.append(ip)
            seen.add(ip)
    if len(candidates) < max_candidates:
        for ip in _discover_hostname_tailscale_ipv4_candidates(
            max_candidates=max_candidates - len(candidates),
        ):
            if ip not in seen:
                candidates.append(ip)
                seen.add(ip)
            if len(candidates) >= max_candidates:
                break
    return candidates


def _resolve_aawm_route_local_display_host(
    *,
    local_source: str,
    monotonic_now: Optional[float] = None,
    snapshot_paths: Optional[tuple[str, ...]] = None,
    allow_blocking_lookup: bool = True,
) -> tuple[str, str]:
    now = time.monotonic() if monotonic_now is None else monotonic_now
    fallback_cache_key = _aawm_route_host_local_cache_key(local_source)
    with _aawm_route_host_reverse_dns_cache_lock:
        cached = _aawm_route_host_reverse_dns_cache.get(
            _AAWM_ROUTE_HOST_LOCAL_LOOKUP_CACHE_KEY
        )
        if cached is not None:
            host_name, cached_source, expires_at = cached
            if expires_at > now:
                return host_name, _aawm_route_host_local_cache_source_label(
                    cached_source
                )
            _aawm_route_host_reverse_dns_cache.pop(
                _AAWM_ROUTE_HOST_LOCAL_LOOKUP_CACHE_KEY, None
            )
        cached = _aawm_route_host_reverse_dns_cache.get(fallback_cache_key)
        if cached is not None:
            host_name, cached_source, expires_at = cached
            if expires_at > now:
                return host_name, _aawm_route_host_local_cache_source_label(
                    cached_source
                )
            _aawm_route_host_reverse_dns_cache.pop(fallback_cache_key, None)

    # Tailscale self-identity snapshot is local file I/O only (no DNS).
    tailscale_self = _resolve_aawm_route_host_name_from_tailscale_self_identity(
        monotonic_now=now,
        snapshot_paths=snapshot_paths,
    )
    if tailscale_self is not None:
        return tailscale_self

    if not allow_blocking_lookup:
        # Fast path for hot request handlers: return loopback immediately and
        # enrich MagicDNS / hostname discovery in the background.
        _schedule_aawm_route_host_name_enrichment(
            local_source,
            local_source=local_source,
        )
        return _AAWM_ROUTE_HOST_LOOPBACK_LABEL, local_source

    resolved_host: Optional[str] = None
    resolved_source = "magicdns_local"
    for candidate_ip in _discover_local_tailscale_ipv4_candidates():
        resolved_host = _resolve_hostname_via_magicdns(candidate_ip)
        if resolved_host:
            break

    if resolved_host:
        _store_aawm_route_host_reverse_dns_cache(
            _AAWM_ROUTE_HOST_LOCAL_LOOKUP_CACHE_KEY,
            resolved_host,
            resolved_source,
            now=now,
            ttl_for_source=_aawm_route_host_local_cache_ttl_for_source,
        )
        return resolved_host, resolved_source

    _store_aawm_route_host_reverse_dns_cache(
        fallback_cache_key,
        _AAWM_ROUTE_HOST_LOOPBACK_LABEL,
        local_source,
        now=now,
        ttl_for_source=_aawm_route_host_local_cache_ttl_for_source,
    )
    return _AAWM_ROUTE_HOST_LOOPBACK_LABEL, local_source


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


def _build_dns_query(
    qname: str,
    *,
    query_type: int,
    transaction_id: int = 0x1357,
) -> bytes:
    import random
    import struct

    tid = transaction_id if transaction_id is not None else random.randint(0, 65535)
    header = struct.pack("!HHHHHH", tid, 0x0100, 1, 0, 0, 0)
    query = _encode_dns_name(qname) + struct.pack("!HH", query_type, 1)
    return header + query


def _build_dns_ptr_query(qname: str, *, transaction_id: int = 0x1357) -> bytes:
    return _build_dns_query(
        qname,
        query_type=12,
        transaction_id=transaction_id,
    )


def _build_dns_a_query(qname: str, *, transaction_id: int = 0x2468) -> bytes:
    return _build_dns_query(
        qname,
        query_type=1,
        transaction_id=transaction_id,
    )


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


def _extract_ipv4_addresses_from_dns_response(data: bytes) -> list[str]:
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
    addresses: list[str] = []
    for _ in range(ancount):
        _, offset = _decode_dns_name(data, offset)
        if offset + 10 > len(data):
            break
        rtype, _, _, rdlength = struct.unpack("!HHIH", data[offset : offset + 10])
        offset += 10
        rdata_end = offset + rdlength
        if rdata_end > len(data):
            break
        if rtype == 1 and rdlength == 4:
            try:
                addresses.append(str(ipaddress.ip_address(data[offset:rdata_end])))
            except ValueError:
                pass
        offset = rdata_end
    return addresses


def _resolve_ipv4_via_magicdns(
    hostname: str,
    *,
    resolver: str = _AAWM_ROUTE_HOST_MAGICDNS_RESOLVER,
    port: int = _AAWM_ROUTE_HOST_MAGICDNS_PORT,
    timeout_seconds: float = _AAWM_ROUTE_HOST_REVERSE_DNS_TIMEOUT_SECONDS,
) -> list[str]:
    cleaned = _clean_aawm_route_log_field(hostname)
    if not cleaned or not _is_aawm_route_tailnet_domain(cleaned):
        return []
    query = _build_dns_a_query(cleaned.rstrip("."))
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.settimeout(timeout_seconds)
        sock.sendto(query, (resolver, port))
        data, _ = sock.recvfrom(4096)
    except Exception:
        return []
    finally:
        sock.close()

    addresses: list[str] = []
    for address in _extract_ipv4_addresses_from_dns_response(data):
        if _is_tailscale_cgnat_client_ip(address):
            addresses.append(address)
    return addresses


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
    allow_blocking_lookup: bool = True,
) -> tuple[Optional[str], Optional[str]]:
    normalized_ip = _normalize_aawm_route_client_ip(client_ip)
    if not normalized_ip:
        return None, None
    is_local, local_source = _is_aawm_route_local_display_ip(normalized_ip)
    if is_local:
        return _resolve_aawm_route_local_display_host(
            local_source=local_source,
            monotonic_now=monotonic_now,
            allow_blocking_lookup=allow_blocking_lookup,
        )
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

    if not allow_blocking_lookup:
        # Critical path: return IP attribution immediately. Do not cache the
        # provisional IP-literal so background enrichment can still perform
        # reverse DNS / MagicDNS and populate the cache for later requests
        # (RR-041 / RR-049 / D1-529).
        _schedule_aawm_route_host_name_enrichment(normalized_ip)
        return normalized_ip, "ip_literal"

    host_name = normalized_ip
    source = "ip_literal"
    # Never touch process-global socket.setdefaulttimeout. Use a dedicated
    # executor + per-call future timeout for gethostbyaddr; MagicDNS already
    # uses sock.settimeout on its own UDP socket.
    hostname = _gethostbyaddr_with_timeout(normalized_ip)
    if hostname:
        resolved = _hostname_label_from_reverse_lookup(hostname)
        if resolved:
            host_name = resolved
            source = "reverse_dns"
    if source == "ip_literal" and _is_tailscale_cgnat_client_ip(normalized_ip):
        resolved = _resolve_hostname_via_magicdns(normalized_ip)
        if resolved:
            host_name = resolved
            source = "magicdns_reverse"

    _store_aawm_route_host_reverse_dns_cache(
        normalized_ip,
        host_name,
        source,
        now=now,
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
    allow_blocking_lookup: bool = False,
) -> dict[str, Optional[str]]:
    """Resolve client IP + host label for route attribution.

    By default this is a non-blocking fast path suitable for hot request
    handlers: cache hits and local-file Tailscale identity return immediately,
    while reverse DNS / MagicDNS are scheduled as background enrichment.
    Pass ``allow_blocking_lookup=True`` (or use
    ``aresolve_aawm_route_host_attribution``) when the caller can wait for a
    full resolution off the event loop.
    """
    if client_ip is None:
        client_ip, client_ip_source = _extract_aawm_route_request_client_ip(
            request,
            general_settings=general_settings,
        )
    else:
        client_ip = _normalize_aawm_route_client_ip(client_ip)
    host_name, host_name_source = _resolve_aawm_route_host_name_from_ip(
        client_ip,
        allow_blocking_lookup=allow_blocking_lookup,
    )
    return {
        "client_ip": client_ip,
        "client_ip_source": client_ip_source,
        "host_name": host_name,
        "host_name_source": host_name_source,
    }


async def aresolve_aawm_route_host_attribution(
    request: Request,
    *,
    general_settings: Optional[dict[str, Any]] = None,
    client_ip: Optional[str] = None,
    client_ip_source: Optional[str] = None,
    allow_blocking_lookup: bool = True,
) -> dict[str, Optional[str]]:
    """Async host attribution that never blocks the event loop on DNS.

    When ``allow_blocking_lookup`` is True (default), reverse DNS / MagicDNS
    run via ``asyncio.to_thread``. When False, returns the same immediate
    IP-literal / cache fast path as the sync API.
    """
    if not allow_blocking_lookup:
        return resolve_aawm_route_host_attribution(
            request,
            general_settings=general_settings,
            client_ip=client_ip,
            client_ip_source=client_ip_source,
            allow_blocking_lookup=False,
        )
    return await asyncio.to_thread(
        resolve_aawm_route_host_attribution,
        request,
        general_settings=general_settings,
        client_ip=client_ip,
        client_ip_source=client_ip_source,
        allow_blocking_lookup=True,
    )


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


def _get_or_replace_aawm_route_rollup_accumulator_locked() -> "AawmRouteRollupAccumulator":
    """Return the configured singleton.

    Caller must hold ``_aawm_route_rollup_lock``. Replaces the singleton when it
    is missing or when the configured interval no longer matches so callers that
    mutate rollup state never operate on an orphaned instance.
    """
    global _aawm_route_rollup_accumulator
    interval_seconds = get_aawm_route_rollup_interval_seconds()
    if (
        _aawm_route_rollup_accumulator is None
        or _aawm_route_rollup_accumulator.interval_seconds() != interval_seconds
    ):
        _aawm_route_rollup_accumulator = AawmRouteRollupAccumulator()
    return _aawm_route_rollup_accumulator


def get_aawm_route_rollup_accumulator() -> AawmRouteRollupAccumulator:
    with _aawm_route_rollup_lock:
        accumulator = _get_or_replace_aawm_route_rollup_accumulator_locked()
    # Flush-worker lifecycle uses a separate lock and may join a worker that
    # itself takes `_aawm_route_rollup_lock`; never call it while holding that lock.
    _ensure_aawm_route_rollup_flush_worker()
    return accumulator


def clear_aawm_route_rollups() -> None:
    with _aawm_route_rollup_lock:
        _get_or_replace_aawm_route_rollup_accumulator_locked().clear()


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
        # Obtain/replace under the same critical section as flush_due so a
        # concurrent interval change cannot leave this tick mutating an orphan.
        accumulator = _get_or_replace_aawm_route_rollup_accumulator_locked()
        if not accumulator.enabled():
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
    with _aawm_route_rollup_lock:
        accumulator = _get_or_replace_aawm_route_rollup_accumulator_locked()
        lines = accumulator.flush(force=force, now=now, early=early)
    # Ensure the background flush worker tracks the current interval without
    # holding the rollup lock (worker stop may re-enter that lock).
    _ensure_aawm_route_rollup_flush_worker()
    return lines


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
    with _aawm_route_rollup_lock:
        # Get-or-replace and record under one continuous critical section so an
        # interval-driven singleton swap cannot orphan this mutation.
        accumulator = _get_or_replace_aawm_route_rollup_accumulator_locked()
        lines = accumulator.record(
            group_header_label=group_header_label,
            incoming_endpoint=incoming_endpoint,
            outgoing_target=outgoing_target,
            model_label=model_label,
            turns=turns,
            status=status,
            now=now,
        )
    _ensure_aawm_route_rollup_flush_worker()
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
