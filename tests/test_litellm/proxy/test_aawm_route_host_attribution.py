import socket
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi import Request

from litellm.proxy.aawm_route_logging import (
    _normalize_aawm_route_client_ip,
    attach_aawm_route_rollup_context,
    build_aawm_route_repo_client_host_label,
    build_aawm_route_rollup_group_header_label,
    resolve_aawm_route_host_attribution,
)


def test_normalize_aawm_route_client_ip_preserves_canonical_ip_literals():
    assert _normalize_aawm_route_client_ip("127.0.0.1") == "127.0.0.1"
    assert _normalize_aawm_route_client_ip("::1") == "::1"
    assert _normalize_aawm_route_client_ip("172.19.0.1") == "172.19.0.1"
    assert _normalize_aawm_route_client_ip("100.99.1.5") == "100.99.1.5"


def test_build_aawm_route_repo_client_host_label_exact_format():
    label = build_aawm_route_repo_client_host_label(
        repository="aawm-infrastructure",
        client_product_label="codex-cli/0.142.5",
        host_name="thoth",
    )
    assert label == "aawm-infrastructure#Codex[0.142.5]@thoth"


def test_build_aawm_route_rollup_group_header_label_includes_host():
    label = build_aawm_route_rollup_group_header_label(
        repository="litellm",
        client_product_label="codex-cli/0.141.0",
        host_name="localhost",
    )
    assert label == "litellm#Codex[0.141.0]@localhost"


def test_resolve_aawm_route_host_attribution_loopback_resolves_local_magicdns(
    monkeypatch,
):
    from litellm.proxy import aawm_route_logging as route_logging

    route_logging._aawm_route_host_reverse_dns_cache.clear()
    request = Mock(spec=Request)
    request.headers = {}
    request.client = SimpleNamespace(host="127.0.0.1", port=54321)
    request.scope = {"client": ("127.0.0.1", 54321)}

    monkeypatch.setattr(
        "litellm.proxy.auth.ip_address_utils.IPAddressUtils.get_mcp_client_ip",
        lambda request, general_settings=None: "127.0.0.1",
    )
    monkeypatch.setattr(
        route_logging,
        "_discover_local_tailscale_ipv4_candidates",
        Mock(return_value=["100.99.166.16"]),
    )
    monkeypatch.setattr(
        route_logging,
        "_resolve_hostname_via_magicdns",
        Mock(return_value="seshat"),
    )

    attribution = resolve_aawm_route_host_attribution(
        request,
        allow_blocking_lookup=True,
    )
    assert attribution["client_ip"] == "127.0.0.1"
    assert attribution["host_name"] == "seshat"
    assert attribution["host_name_source"] == "magicdns_local"


def test_resolve_aawm_route_host_attribution_docker_gateway_resolves_local_magicdns(
    monkeypatch,
):
    from litellm.proxy import aawm_route_logging as route_logging

    route_logging._aawm_route_host_reverse_dns_cache.clear()
    request = Mock(spec=Request)
    request.headers = {}
    request.client = SimpleNamespace(host="172.19.0.1", port=54321)
    request.scope = {"client": ("172.19.0.1", 54321)}

    monkeypatch.setattr(
        "litellm.proxy.auth.ip_address_utils.IPAddressUtils.get_mcp_client_ip",
        lambda request, general_settings=None: "172.19.0.1",
    )
    monkeypatch.setattr(
        route_logging,
        "_discover_local_tailscale_ipv4_candidates",
        Mock(return_value=["100.99.1.5"]),
    )
    monkeypatch.setattr(
        route_logging,
        "_resolve_hostname_via_magicdns",
        Mock(return_value="thoth"),
    )

    attribution = resolve_aawm_route_host_attribution(
        request,
        allow_blocking_lookup=True,
    )
    assert attribution["client_ip"] == "172.19.0.1"
    assert attribution["host_name"] == "thoth"
    assert attribution["host_name_source"] == "magicdns_local"


def test_resolve_aawm_route_host_attribution_trusted_docker_gateway_xff_uses_magicdns_host(
    monkeypatch,
):
    from litellm.proxy import aawm_route_logging as route_logging

    route_logging._aawm_route_host_reverse_dns_cache.clear()
    request = Mock(spec=Request)
    request.headers = {"x-forwarded-for": "100.100.7.5"}
    request.client = SimpleNamespace(host="172.18.0.1", port=54321)
    request.scope = {"client": ("172.18.0.1", 54321)}

    general_settings = {
        "aawm_route_use_x_forwarded_for": True,
        "aawm_route_trusted_proxy_ranges": ["172.18.0.1/32"],
    }
    monkeypatch.setattr(
        route_logging.socket,
        "gethostbyaddr",
        Mock(side_effect=OSError("reverse dns not used for cgnat test")),
    )
    monkeypatch.setattr(
        route_logging,
        "_resolve_hostname_via_magicdns",
        Mock(return_value="desktop-qjhrj1m-wsl"),
    )

    attribution = resolve_aawm_route_host_attribution(
        request,
        general_settings=general_settings,
        allow_blocking_lookup=True,
    )
    assert attribution["client_ip"] == "100.100.7.5"
    assert attribution["client_ip_source"] == "x_forwarded_for"
    assert attribution["host_name"] == "desktop-qjhrj1m-wsl"
    assert attribution["host_name_source"] == "magicdns_reverse"


def test_resolve_aawm_route_host_attribution_untrusted_xff_source_is_ignored(
    monkeypatch,
):
    request = Mock(spec=Request)
    request.headers = {"x-forwarded-for": "100.100.7.5"}
    request.client = SimpleNamespace(host="203.0.113.19", port=54321)
    request.scope = {"client": ("203.0.113.19", 54321)}

    general_settings = {
        "aawm_route_use_x_forwarded_for": True,
        "aawm_route_trusted_proxy_ranges": ["172.18.0.1/32"],
    }

    attribution = resolve_aawm_route_host_attribution(
        request,
        general_settings=general_settings,
    )
    assert attribution["client_ip"] == "203.0.113.19"
    assert attribution["client_ip_source"] == "request_client"
    assert attribution["host_name"] == "203.0.113.19"
    assert attribution["host_name_source"] == "ip_literal"


def test_resolve_aawm_route_host_attribution_malformed_trusted_xff_falls_back_to_request_client(
    monkeypatch,
):
    from litellm.proxy import aawm_route_logging as route_logging

    route_logging._aawm_route_host_reverse_dns_cache.clear()
    request = Mock(spec=Request)
    request.headers = {"x-forwarded-for": "not-an-ip"}
    request.client = SimpleNamespace(host="172.18.0.1", port=54321)
    request.scope = {"client": ("172.18.0.1", 54321)}

    general_settings = {
        "aawm_route_use_x_forwarded_for": True,
        "aawm_route_trusted_proxy_ranges": ["172.18.0.1/32"],
    }
    monkeypatch.setattr(
        route_logging.socket,
        "gethostbyaddr",
        Mock(side_effect=OSError("reverse dns not used for local paths")),
    )
    monkeypatch.setattr(
        route_logging,
        "_discover_local_tailscale_ipv4_candidates",
        Mock(return_value=["100.99.1.5"]),
    )
    monkeypatch.setattr(
        route_logging,
        "_resolve_hostname_via_magicdns",
        Mock(return_value="thoth"),
    )

    attribution = resolve_aawm_route_host_attribution(
        request,
        general_settings=general_settings,
        allow_blocking_lookup=True,
    )
    assert attribution["client_ip"] == "172.18.0.1"
    assert attribution["client_ip_source"] == "request_client"
    assert attribution["host_name"] == "thoth"
    assert attribution["host_name_source"] == "magicdns_local"


def test_resolve_aawm_route_host_attribution_keeps_lan_ip_fallback(monkeypatch):
    request = Mock(spec=Request)
    request.headers = {}
    request.client = SimpleNamespace(host="192.168.1.42", port=54321)
    request.scope = {"client": ("192.168.1.42", 54321)}

    monkeypatch.setattr(
        "litellm.proxy.auth.ip_address_utils.IPAddressUtils.get_mcp_client_ip",
        lambda request, general_settings=None: "192.168.1.42",
    )
    gethostbyaddr = Mock(side_effect=OSError("no reverse dns"))
    monkeypatch.setattr(
        "litellm.proxy.aawm_route_logging.socket.gethostbyaddr",
        gethostbyaddr,
    )

    # Default sync API is non-blocking: reverse DNS is not required to return.
    attribution = resolve_aawm_route_host_attribution(request)
    assert attribution["client_ip"] == "192.168.1.42"
    assert attribution["host_name"] == "192.168.1.42"
    assert attribution["host_name_source"] == "ip_literal"
    gethostbyaddr.assert_not_called()


def test_resolve_aawm_route_host_attribution_reverse_dns_fallback(monkeypatch):
    request = Mock(spec=Request)
    request.headers = {}
    request.client = SimpleNamespace(host="100.99.1.5", port=12345)
    request.scope = {"client": ("100.99.1.5", 12345)}

    monkeypatch.setattr(
        "litellm.proxy.auth.ip_address_utils.IPAddressUtils.get_mcp_client_ip",
        lambda request, general_settings=None: "100.99.1.5",
    )
    monkeypatch.setattr(
        "litellm.proxy.aawm_route_logging.socket.gethostbyaddr",
        lambda ip: ("thoth.tailnet.ts.net", [], ["100.99.1.5"]),
    )

    attribution = resolve_aawm_route_host_attribution(
        request,
        allow_blocking_lookup=True,
    )
    assert attribution["client_ip"] == "100.99.1.5"
    assert attribution["host_name"] == "thoth"
    assert attribution["host_name_source"] in {"reverse_dns", "reverse_dns_cache"}


def test_attach_aawm_route_rollup_context_copies_host_metadata(monkeypatch):
    from litellm.proxy import aawm_route_logging as route_logging

    route_logging._aawm_route_host_reverse_dns_cache.clear()
    request = Mock(spec=Request)
    request.method = "POST"
    request.url = "http://127.0.0.1:4001/openai_passthrough/responses"
    request.headers = {"user-agent": "codex-cli/0.142.5"}
    request.client = SimpleNamespace(host="100.99.1.5", port=12345)
    request.scope = {
        "path": "/openai_passthrough/responses",
        "query_string": b"",
        "client": ("100.99.1.5", 12345),
    }
    kwargs = {
        "litellm_params": {
            "metadata": {
                "repository": "aawm-infrastructure",
            }
        }
    }

    monkeypatch.setattr(
        "litellm.proxy.auth.ip_address_utils.IPAddressUtils.get_mcp_client_ip",
        lambda request, general_settings=None: "100.99.1.5",
    )
    monkeypatch.setattr(
        "litellm.proxy.aawm_route_logging.socket.gethostbyaddr",
        lambda ip: ("thoth.tailnet.ts.net", [], ["100.99.1.5"]),
    )
    # Pre-warm host cache with a blocking lookup so rollup can use the
    # non-blocking sync path without waiting on background enrichment.
    route_logging._resolve_aawm_route_host_name_from_ip(
        "100.99.1.5",
        allow_blocking_lookup=True,
    )

    context = attach_aawm_route_rollup_context(
        request=request,
        target="https://chatgpt.com/backend-api/codex/responses",
        request_body={"model": "gpt-5.5"},
        kwargs=kwargs,
    )

    metadata = kwargs["litellm_params"]["metadata"]
    assert context is not None
    assert context["group_header_label"] == (
        "aawm-infrastructure#Codex[0.142.5]@thoth"
    )
    assert metadata["client_ip"] == "100.99.1.5"
    assert metadata["host_name"] == "thoth"
    assert metadata["host_name_source"] in {"reverse_dns", "reverse_dns_cache"}
    assert metadata["aawm_route_rollup_context"]["host_name"] == "thoth"


def test_is_tailscale_cgnat_client_ip_detects_100_64_range():
    from litellm.proxy.aawm_route_logging import _is_tailscale_cgnat_client_ip

    assert _is_tailscale_cgnat_client_ip("100.99.166.16") is True
    assert _is_tailscale_cgnat_client_ip("100.64.0.1") is True
    assert _is_tailscale_cgnat_client_ip("100.63.255.255") is False
    assert _is_tailscale_cgnat_client_ip("192.168.1.1") is False


def test_resolve_aawm_route_host_name_magicdns_fallback_on_cgnat_miss(monkeypatch):
    from litellm.proxy import aawm_route_logging as route_logging

    route_logging._aawm_route_host_reverse_dns_cache.clear()
    monkeypatch.setattr(
        route_logging.socket,
        "gethostbyaddr",
        Mock(side_effect=OSError("no reverse dns")),
    )
    monkeypatch.setattr(
        route_logging,
        "_resolve_hostname_via_magicdns",
        Mock(return_value="seshat"),
    )

    host_name, source = route_logging._resolve_aawm_route_host_name_from_ip(
        "100.99.166.16",
        monotonic_now=1000.0,
    )
    assert host_name == "seshat"
    assert source == "magicdns_reverse"


def test_resolve_aawm_route_host_name_prefers_reverse_dns_over_magicdns(monkeypatch):
    from litellm.proxy import aawm_route_logging as route_logging

    route_logging._aawm_route_host_reverse_dns_cache.clear()
    monkeypatch.setattr(
        route_logging.socket,
        "gethostbyaddr",
        lambda ip: ("thoth.tailnet.ts.net", [], ["100.99.1.5"]),
    )
    magic = Mock(return_value="should-not-be-used")
    monkeypatch.setattr(route_logging, "_resolve_hostname_via_magicdns", magic)

    host_name, source = route_logging._resolve_aawm_route_host_name_from_ip(
        "100.99.1.5",
        monotonic_now=2000.0,
    )
    assert host_name == "thoth"
    assert source == "reverse_dns"
    magic.assert_not_called()


def test_resolve_aawm_route_host_name_cgnat_full_ip_fallback(monkeypatch):
    from litellm.proxy import aawm_route_logging as route_logging

    route_logging._aawm_route_host_reverse_dns_cache.clear()
    monkeypatch.setattr(
        route_logging.socket,
        "gethostbyaddr",
        Mock(side_effect=OSError("no reverse dns")),
    )
    monkeypatch.setattr(
        route_logging,
        "_resolve_hostname_via_magicdns",
        Mock(return_value=None),
    )

    host_name, source = route_logging._resolve_aawm_route_host_name_from_ip(
        "100.99.166.16",
        monotonic_now=3000.0,
    )
    assert host_name == "100.99.166.16"
    assert source == "ip_literal"


def test_resolve_aawm_route_host_name_cache_preserves_source_and_ttl(monkeypatch):
    from litellm.proxy import aawm_route_logging as route_logging

    route_logging._aawm_route_host_reverse_dns_cache.clear()
    monkeypatch.setattr(
        route_logging.socket,
        "gethostbyaddr",
        Mock(side_effect=OSError("no reverse dns")),
    )
    monkeypatch.setattr(
        route_logging,
        "_resolve_hostname_via_magicdns",
        Mock(return_value="seshat"),
    )

    first_host, first_source = route_logging._resolve_aawm_route_host_name_from_ip(
        "100.99.166.16",
        monotonic_now=4000.0,
    )
    second_host, second_source = route_logging._resolve_aawm_route_host_name_from_ip(
        "100.99.166.16",
        monotonic_now=4001.0,
    )
    assert first_host == second_host == "seshat"
    assert first_source == "magicdns_reverse"
    assert second_source == "magicdns_reverse_cache"

    cached = route_logging._aawm_route_host_reverse_dns_cache["100.99.166.16"]
    host_name, cached_source, expires_at = cached
    assert cached_source == "magicdns_reverse"
    assert expires_at == 4000.0 + route_logging._AAWM_ROUTE_HOST_REVERSE_DNS_CACHE_TTL_SECONDS

    route_logging._aawm_route_host_reverse_dns_cache.clear()
    monkeypatch.setattr(
        route_logging,
        "_resolve_hostname_via_magicdns",
        Mock(return_value=None),
    )
    _, ip_source = route_logging._resolve_aawm_route_host_name_from_ip(
        "100.99.166.17",
        monotonic_now=5000.0,
    )
    ip_cached = route_logging._aawm_route_host_reverse_dns_cache["100.99.166.17"]
    _, ip_cached_source, ip_expires = ip_cached
    assert ip_source == "ip_literal"
    assert ip_cached_source == "ip_literal"
    assert ip_expires == 5000.0 + route_logging._AAWM_ROUTE_HOST_IP_LITERAL_CACHE_TTL_SECONDS


def test_resolve_hostname_via_magicdns_parses_ptr_response(monkeypatch):
    import struct

    from litellm.proxy import aawm_route_logging as route_logging

    qname = "16.166.99.100.in-addr.arpa"
    question = route_logging._encode_dns_name(qname) + struct.pack("!HH", 12, 1)
    ptr_name = route_logging._encode_dns_name("seshat.tailf1878c.ts.net")
    response = (
        struct.pack("!HHHHHH", 0x1357, 0x8180, 1, 1, 0, 0)
        + question
        + b"\xc0\x0c"
        + struct.pack("!HHIH", 12, 1, 30, len(ptr_name))
        + ptr_name
    )

    class FakeSocket:
        def __init__(self):
            self.timeout = None
            self.sent_to = None
            self.closed = False

        def settimeout(self, timeout):
            self.timeout = timeout

        def sendto(self, payload, target):
            self.sent_to = target
            assert payload == route_logging._build_dns_ptr_query(qname)
            assert target == ("100.100.100.100", 53)

        def recvfrom(self, size):
            assert size == 4096
            return response, ("100.100.100.100", 53)

        def close(self):
            self.closed = True

    fake_socket = FakeSocket()
    monkeypatch.setattr(
        route_logging.socket,
        "socket",
        lambda family, sock_type: fake_socket,
    )

    assert route_logging._resolve_hostname_via_magicdns("100.99.166.16") == "seshat"
    assert fake_socket.timeout == route_logging._AAWM_ROUTE_HOST_REVERSE_DNS_TIMEOUT_SECONDS
    assert fake_socket.closed is True


def test_resolve_ipv4_via_magicdns_parses_a_response(monkeypatch):
    import struct

    from litellm.proxy import aawm_route_logging as route_logging

    qname = "desktop-qjhrj1m.tailf1878c.ts.net"
    question = route_logging._encode_dns_name(qname) + struct.pack("!HH", 1, 1)
    response = (
        struct.pack("!HHHHHH", 0x2468, 0x8580, 1, 1, 0, 0)
        + question
        + b"\xc0\x0c"
        + struct.pack("!HHIH", 1, 1, 600, 4)
        + bytes([100, 100, 7, 5])
    )

    class FakeSocket:
        def __init__(self):
            self.timeout = None
            self.sent_to = None
            self.closed = False

        def settimeout(self, timeout):
            self.timeout = timeout

        def sendto(self, payload, target):
            self.sent_to = target
            assert payload == route_logging._build_dns_a_query(qname)
            assert target == ("100.100.100.100", 53)

        def recvfrom(self, size):
            assert size == 4096
            return response, ("100.100.100.100", 53)

        def close(self):
            self.closed = True

    fake_socket = FakeSocket()
    monkeypatch.setattr(
        route_logging.socket,
        "socket",
        lambda family, sock_type: fake_socket,
    )

    assert route_logging._resolve_ipv4_via_magicdns(qname) == ["100.100.7.5"]
    assert fake_socket.timeout == route_logging._AAWM_ROUTE_HOST_REVERSE_DNS_TIMEOUT_SECONDS
    assert fake_socket.closed is True


def test_resolve_aawm_route_local_display_host_falls_back_to_localhost(monkeypatch):
    from litellm.proxy import aawm_route_logging as route_logging

    route_logging._aawm_route_host_reverse_dns_cache.clear()
    monkeypatch.setattr(
        route_logging,
        "_discover_local_tailscale_ipv4_candidates",
        Mock(return_value=[]),
    )
    monkeypatch.setattr(
        route_logging.socket,
        "getfqdn",
        Mock(return_value="localhost"),
    )
    monkeypatch.setattr(
        route_logging.socket,
        "gethostname",
        Mock(return_value="localhost"),
    )

    host_name, source = route_logging._resolve_aawm_route_local_display_host(
        local_source="loopback",
        monotonic_now=9000.0,
    )
    assert host_name == "localhost"
    assert source == "loopback"


def test_resolve_aawm_route_local_display_host_fallback_cache_is_source_specific(
    monkeypatch,
):
    from litellm.proxy import aawm_route_logging as route_logging

    route_logging._aawm_route_host_reverse_dns_cache.clear()
    monkeypatch.setattr(
        route_logging,
        "_discover_local_tailscale_ipv4_candidates",
        Mock(return_value=[]),
    )
    monkeypatch.setattr(
        route_logging.socket,
        "getfqdn",
        Mock(return_value="localhost"),
    )
    monkeypatch.setattr(
        route_logging.socket,
        "gethostname",
        Mock(return_value="localhost"),
    )

    loopback_host, loopback_source = route_logging._resolve_aawm_route_local_display_host(
        local_source="loopback",
        monotonic_now=9500.0,
    )
    docker_host, docker_source = route_logging._resolve_aawm_route_local_display_host(
        local_source="docker_bridge_gateway",
        monotonic_now=9501.0,
    )

    assert loopback_host == docker_host == "localhost"
    assert loopback_source == "loopback"
    assert docker_source == "docker_bridge_gateway"


def test_resolve_aawm_route_local_display_host_uses_magicdns_local_cache(
    monkeypatch,
):
    from litellm.proxy import aawm_route_logging as route_logging

    route_logging._aawm_route_host_reverse_dns_cache.clear()
    discover = Mock(return_value=["100.99.166.16"])
    magic = Mock(return_value="seshat")
    monkeypatch.setattr(
        route_logging,
        "_discover_local_tailscale_ipv4_candidates",
        discover,
    )
    monkeypatch.setattr(route_logging, "_resolve_hostname_via_magicdns", magic)

    first_host, first_source = route_logging._resolve_aawm_route_local_display_host(
        local_source="docker_bridge_gateway",
        monotonic_now=10000.0,
    )
    second_host, second_source = route_logging._resolve_aawm_route_local_display_host(
        local_source="docker_bridge_gateway",
        monotonic_now=10001.0,
    )
    assert first_host == second_host == "seshat"
    assert first_source == "magicdns_local"
    assert second_source == "magicdns_local_cache"
    discover.assert_called_once()
    magic.assert_called_once()


def test_discover_hostname_tailscale_ipv4_candidates_uses_tailnet_search_domain(
    monkeypatch,
):
    from litellm.proxy import aawm_route_logging as route_logging

    monkeypatch.setattr(
        route_logging.socket,
        "getfqdn",
        Mock(return_value="desktop-qjhrj1m.localdomain"),
    )
    monkeypatch.setattr(
        route_logging.socket,
        "gethostname",
        Mock(return_value="desktop-qjhrj1m"),
    )
    monkeypatch.setattr(
        route_logging,
        "_tailnet_search_domains_from_resolv_conf",
        Mock(return_value=["tailf1878c.ts.net"]),
    )
    monkeypatch.setattr(
        route_logging,
        "_hostname_lookup_file_candidates",
        Mock(return_value=[]),
    )

    monkeypatch.setattr(
        route_logging.socket,
        "getaddrinfo",
        Mock(side_effect=OSError("not found")),
    )
    magicdns_a = Mock(return_value=["100.100.7.5"])
    monkeypatch.setattr(route_logging, "_resolve_ipv4_via_magicdns", magicdns_a)

    assert route_logging._discover_hostname_tailscale_ipv4_candidates() == [
        "100.100.7.5"
    ]
    magicdns_a.assert_called_once_with("desktop-qjhrj1m.tailf1878c.ts.net")


def test_hostname_lookup_file_candidates_reads_host_hostname(tmp_path):
    from litellm.proxy import aawm_route_logging as route_logging

    host_hostname = tmp_path / "hostname"
    host_hostname.write_text("DESKTOP-QJHRJ1M\n", encoding="utf-8")

    assert route_logging._hostname_lookup_file_candidates(
        lookup_paths=(str(host_hostname),),
    ) == ["DESKTOP-QJHRJ1M"]


def test_discover_local_tailscale_ipv4_candidates_merges_fib_trie_and_hostname(
    monkeypatch,
):
    from litellm.proxy import aawm_route_logging as route_logging

    monkeypatch.setattr(
        route_logging,
        "_parse_proc_net_fib_trie_tailscale_local_ips",
        Mock(return_value=["100.99.166.16"]),
    )
    monkeypatch.setattr(
        route_logging,
        "_discover_hostname_tailscale_ipv4_candidates",
        Mock(return_value=["100.99.1.5"]),
    )

    assert route_logging._discover_local_tailscale_ipv4_candidates() == [
        "100.99.166.16",
        "100.99.1.5",
    ]


def test_aawm_route_rollup_accumulator_cooldown_line_includes_host():
    from datetime import datetime

    from litellm.proxy.aawm_route_logging import AawmRouteRollupAccumulator

    now = datetime(2026, 7, 6, 16, 4, 6)
    accumulator = AawmRouteRollupAccumulator(interval_seconds=60)
    header = "litellm#Codex[0.142.5]@thoth"
    endpoint = "/openai_passthrough/responses"
    target = "openrouter.ai/api/v1/chat/completions"
    model = "openrouter/cohere/north-mini-code:free(aawm-low)"

    accumulator.record(
        group_header_label=header,
        incoming_endpoint=endpoint,
        outgoing_target=target,
        model_label=model,
        turns=0,
        status="Cooling Down",
        now=now,
    )
    flushed = accumulator.flush(force=True, now=now)
    assert flushed[0].startswith(
        "20260706 16:04:06 litellm#Codex[0.142.5]@thoth /openai_passthrough/responses"
    )
    assert " [Cooling Down]" in flushed[1]
    assert " -> openrouter.ai/api/v1/chat/completions" in flushed[1]


def test_aawm_route_rollup_accumulator_cooldown_line_omits_host_when_unresolved():
    from datetime import datetime

    from litellm.proxy.aawm_route_logging import AawmRouteRollupAccumulator

    now = datetime(2026, 7, 6, 16, 4, 6)
    accumulator = AawmRouteRollupAccumulator(interval_seconds=60)
    header = "litellm#Codex[0.142.5]"
    endpoint = "/openai_passthrough/responses"
    target = "openrouter.ai/api/v1/chat/completions"
    model = "openrouter/cohere/north-mini-code:free(aawm-low)"

    accumulator.record(
        group_header_label=header,
        incoming_endpoint=endpoint,
        outgoing_target=target,
        model_label=model,
        turns=0,
        status="Cooling Down",
        now=now,
    )
    flushed = accumulator.flush(force=True, now=now)
    assert "@thoth" not in flushed[0]
    assert flushed[0].startswith(
        "20260706 16:04:06 litellm#Codex[0.142.5] /openai_passthrough/responses"
    )


def test_parse_tailscale_self_snapshot_payload_accepts_sanitized_schema():
    from litellm.proxy import aawm_route_logging as route_logging

    parsed = route_logging._parse_tailscale_self_snapshot_payload(
        {
            "schema_version": 1,
            "self_dns_name": "desktop-qjhrj1m-wsl.tailf1878c.ts.net",
            "self_tailscale_ips": ["100.100.7.5", "fd7a:115c:a1e0::c133:706"],
            "magic_dns_suffix": "tailf1878c.ts.net",
        }
    )
    assert parsed == {
        "dns_name": "desktop-qjhrj1m-wsl.tailf1878c.ts.net",
        "tailscale_ips": ["100.100.7.5"],
        "magic_dns_suffix": "tailf1878c.ts.net",
    }


def test_resolve_local_display_host_prefers_tailscale_self_over_stale_hostname(
    monkeypatch, tmp_path,
):
    from litellm.proxy import aawm_route_logging as route_logging

    route_logging._aawm_route_host_reverse_dns_cache.clear()
    snapshot = tmp_path / "tailscale-self.json"
    snapshot.write_text(
        """{
  "self_dns_name": "desktop-qjhrj1m-wsl.tailf1878c.ts.net",
  "self_tailscale_ips": ["100.100.7.5"],
  "magic_dns_suffix": "tailf1878c.ts.net"
}
""",
        encoding="utf-8",
    )

    discover = Mock(return_value=["100.100.7.5"])
    magic = Mock(return_value="DESKTOP-QJHRJ1M")
    monkeypatch.setattr(
        route_logging,
        "_discover_local_tailscale_ipv4_candidates",
        discover,
    )
    monkeypatch.setattr(route_logging, "_resolve_hostname_via_magicdns", magic)
    monkeypatch.setattr(
        route_logging,
        "_hostname_lookup_file_candidates",
        Mock(return_value=["DESKTOP-QJHRJ1M"]),
    )

    host_name, source = route_logging._resolve_aawm_route_local_display_host(
        local_source="loopback",
        monotonic_now=12000.0,
        snapshot_paths=(str(snapshot),),
    )
    assert host_name == "desktop-qjhrj1m-wsl"
    assert source == "tailscale_self"
    discover.assert_not_called()
    magic.assert_not_called()


def test_resolve_local_display_host_tailscale_self_cache_label(monkeypatch, tmp_path):
    from litellm.proxy import aawm_route_logging as route_logging

    route_logging._aawm_route_host_reverse_dns_cache.clear()
    snapshot = tmp_path / "tailscale-self.json"
    snapshot.write_text(
        '{"self_dns_name": "desktop-qjhrj1m-wsl.tailf1878c.ts.net"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        route_logging,
        "_discover_local_tailscale_ipv4_candidates",
        Mock(return_value=[]),
    )

    first_host, first_source = route_logging._resolve_aawm_route_local_display_host(
        local_source="docker_bridge_gateway",
        monotonic_now=13000.0,
        snapshot_paths=(str(snapshot),),
    )
    second_host, second_source = route_logging._resolve_aawm_route_local_display_host(
        local_source="docker_bridge_gateway",
        monotonic_now=13001.0,
        snapshot_paths=(str(snapshot),),
    )
    assert first_host == second_host == "desktop-qjhrj1m-wsl"
    assert first_source == "tailscale_self"
    assert second_source == "tailscale_self_cache"


def test_resolve_local_display_host_ignores_non_tailnet_self_dns_name(
    monkeypatch, tmp_path,
):
    from litellm.proxy import aawm_route_logging as route_logging

    route_logging._aawm_route_host_reverse_dns_cache.clear()
    snapshot = tmp_path / "tailscale-self.json"
    snapshot.write_text(
        '{"self_dns_name": "desktop-qjhrj1m.localdomain"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        route_logging,
        "_discover_local_tailscale_ipv4_candidates",
        Mock(return_value=["100.99.166.16"]),
    )
    monkeypatch.setattr(
        route_logging,
        "_resolve_hostname_via_magicdns",
        Mock(return_value="seshat"),
    )

    host_name, source = route_logging._resolve_aawm_route_local_display_host(
        local_source="loopback",
        monotonic_now=13500.0,
        snapshot_paths=(str(snapshot),),
    )
    assert host_name == "seshat"
    assert source == "magicdns_local"


def test_resolve_aawm_route_host_attribution_loopback_uses_tailscale_self_snapshot(
    monkeypatch, tmp_path,
):
    from litellm.proxy import aawm_route_logging as route_logging

    route_logging._aawm_route_host_reverse_dns_cache.clear()
    snapshot = tmp_path / "tailscale-self.json"
    snapshot.write_text(
        '{"self_dns_name": "desktop-qjhrj1m-wsl.tailf1878c.ts.net"}',
        encoding="utf-8",
    )

    request = Mock(spec=Request)
    request.headers = {}
    request.client = SimpleNamespace(host="127.0.0.1", port=54321)
    request.scope = {"client": ("127.0.0.1", 54321)}

    monkeypatch.setattr(
        "litellm.proxy.auth.ip_address_utils.IPAddressUtils.get_mcp_client_ip",
        lambda request, general_settings=None: "127.0.0.1",
    )
    monkeypatch.setattr(
        route_logging,
        "_aawm_route_host_tailscale_self_snapshot_paths",
        lambda: (str(snapshot),),
    )
    monkeypatch.setattr(
        route_logging,
        "_discover_local_tailscale_ipv4_candidates",
        Mock(return_value=["100.100.7.5"]),
    )
    monkeypatch.setattr(
        route_logging,
        "_resolve_hostname_via_magicdns",
        Mock(return_value="wrong-host"),
    )

    attribution = resolve_aawm_route_host_attribution(request)
    assert attribution["host_name"] == "desktop-qjhrj1m-wsl"
    assert attribution["host_name_source"] == "tailscale_self"


def test_resolve_local_display_host_falls_back_when_tailscale_self_snapshot_invalid(
    monkeypatch, tmp_path,
):
    from litellm.proxy import aawm_route_logging as route_logging

    route_logging._aawm_route_host_reverse_dns_cache.clear()
    snapshot = tmp_path / "tailscale-self.json"
    snapshot.write_text("not-json", encoding="utf-8")

    monkeypatch.setattr(
        route_logging,
        "_discover_local_tailscale_ipv4_candidates",
        Mock(return_value=["100.99.166.16"]),
    )
    monkeypatch.setattr(
        route_logging,
        "_resolve_hostname_via_magicdns",
        Mock(return_value="seshat"),
    )

    host_name, source = route_logging._resolve_aawm_route_local_display_host(
        local_source="loopback",
        monotonic_now=14000.0,
        snapshot_paths=(str(snapshot),),
    )
    assert host_name == "seshat"
    assert source == "magicdns_local"


def test_resolve_host_name_never_mutates_socket_default_timeout(monkeypatch):
    """B1 / RR-041 / RR-049: no process-global socket.setdefaulttimeout mutation."""
    from litellm.proxy import aawm_route_logging as route_logging

    route_logging._aawm_route_host_reverse_dns_cache.clear()
    original_default = socket.getdefaulttimeout()
    setdefault_calls: list = []

    real_setdefault = socket.setdefaulttimeout

    def tracking_setdefault(value):
        setdefault_calls.append(value)
        return real_setdefault(value)

    monkeypatch.setattr(route_logging.socket, "setdefaulttimeout", tracking_setdefault)
    monkeypatch.setattr(socket, "setdefaulttimeout", tracking_setdefault)
    monkeypatch.setattr(
        route_logging.socket,
        "gethostbyaddr",
        lambda ip: ("thoth.tailnet.ts.net", [], [ip]),
    )
    monkeypatch.setattr(
        route_logging,
        "_resolve_hostname_via_magicdns",
        Mock(return_value=None),
    )

    host_name, source = route_logging._resolve_aawm_route_host_name_from_ip(
        "100.99.1.5",
        monotonic_now=6000.0,
        allow_blocking_lookup=True,
    )
    assert host_name == "thoth"
    assert source == "reverse_dns"
    assert setdefault_calls == []
    assert socket.getdefaulttimeout() == original_default

    # Local discovery path also must not mutate the global default.
    monkeypatch.setattr(
        route_logging,
        "_local_tailscale_hostname_lookup_names",
        Mock(return_value=["desktop-qjhrj1m.tailf1878c.ts.net"]),
    )
    monkeypatch.setattr(
        route_logging,
        "_getaddrinfo_with_timeout",
        Mock(return_value=[]),
    )
    monkeypatch.setattr(
        route_logging,
        "_resolve_ipv4_via_magicdns",
        Mock(return_value=[]),
    )
    route_logging._discover_hostname_tailscale_ipv4_candidates()
    assert setdefault_calls == []
    assert socket.getdefaulttimeout() == original_default


def test_resolve_host_attribution_default_returns_ip_without_reverse_dns(monkeypatch):
    """B1: sync hot-path returns IP attribution immediately without reverse DNS."""
    from litellm.proxy import aawm_route_logging as route_logging

    route_logging._aawm_route_host_reverse_dns_cache.clear()
    request = Mock(spec=Request)
    request.headers = {}
    request.client = SimpleNamespace(host="100.99.1.5", port=12345)
    request.scope = {"client": ("100.99.1.5", 12345)}

    monkeypatch.setattr(
        "litellm.proxy.auth.ip_address_utils.IPAddressUtils.get_mcp_client_ip",
        lambda request, general_settings=None: "100.99.1.5",
    )
    gethostbyaddr = Mock(
        side_effect=AssertionError("gethostbyaddr must not run on default path")
    )
    monkeypatch.setattr(route_logging.socket, "gethostbyaddr", gethostbyaddr)
    magic = Mock(
        side_effect=AssertionError("magicdns must not run on default path")
    )
    monkeypatch.setattr(route_logging, "_resolve_hostname_via_magicdns", magic)
    # Suppress background enrichment so it cannot call DNS during the test.
    monkeypatch.setattr(
        route_logging,
        "_schedule_aawm_route_host_name_enrichment",
        Mock(),
    )

    attribution = resolve_aawm_route_host_attribution(request)
    assert attribution["client_ip"] == "100.99.1.5"
    assert attribution["host_name"] == "100.99.1.5"
    assert attribution["host_name_source"] == "ip_literal"
    gethostbyaddr.assert_not_called()
    magic.assert_not_called()


def test_resolve_host_name_fast_path_schedules_background_enrichment(monkeypatch):
    from litellm.proxy import aawm_route_logging as route_logging

    route_logging._aawm_route_host_reverse_dns_cache.clear()
    scheduled: list = []

    def capture_schedule(client_ip, *, local_source=None):
        scheduled.append((client_ip, local_source))

    monkeypatch.setattr(
        route_logging,
        "_schedule_aawm_route_host_name_enrichment",
        capture_schedule,
    )
    gethostbyaddr = Mock(
        side_effect=AssertionError("blocking reverse dns not allowed on fast path")
    )
    monkeypatch.setattr(route_logging.socket, "gethostbyaddr", gethostbyaddr)

    host_name, source = route_logging._resolve_aawm_route_host_name_from_ip(
        "100.99.166.16",
        monotonic_now=7000.0,
        allow_blocking_lookup=False,
    )
    assert host_name == "100.99.166.16"
    assert source == "ip_literal"
    assert scheduled == [("100.99.166.16", None)]
    gethostbyaddr.assert_not_called()
    # Provisional IP must not be cached so enrichment can still resolve later.
    assert "100.99.166.16" not in route_logging._aawm_route_host_reverse_dns_cache


def test_aresolve_host_attribution_offloads_blocking_lookup(monkeypatch):
    import asyncio

    from litellm.proxy import aawm_route_logging as route_logging
    from litellm.proxy.aawm_route_logging import aresolve_aawm_route_host_attribution

    route_logging._aawm_route_host_reverse_dns_cache.clear()
    request = Mock(spec=Request)
    request.headers = {}
    request.client = SimpleNamespace(host="100.99.1.5", port=12345)
    request.scope = {"client": ("100.99.1.5", 12345)}

    monkeypatch.setattr(
        "litellm.proxy.auth.ip_address_utils.IPAddressUtils.get_mcp_client_ip",
        lambda request, general_settings=None: "100.99.1.5",
    )
    monkeypatch.setattr(
        route_logging.socket,
        "gethostbyaddr",
        lambda ip: ("thoth.tailnet.ts.net", [], [ip]),
    )

    to_thread_calls: list = []
    real_to_thread = asyncio.to_thread

    async def tracking_to_thread(fn, *args, **kwargs):
        to_thread_calls.append((fn, args, kwargs))
        return await real_to_thread(fn, *args, **kwargs)

    monkeypatch.setattr(route_logging.asyncio, "to_thread", tracking_to_thread)

    attribution = asyncio.run(
        aresolve_aawm_route_host_attribution(
            request,
            allow_blocking_lookup=True,
        )
    )
    assert attribution["host_name"] == "thoth"
    assert attribution["host_name_source"] == "reverse_dns"
    assert len(to_thread_calls) == 1
    assert to_thread_calls[0][0] is route_logging.resolve_aawm_route_host_attribution
    assert to_thread_calls[0][2].get("allow_blocking_lookup") is True


def test_aresolve_host_attribution_fast_path_skips_to_thread(monkeypatch):
    import asyncio

    from litellm.proxy import aawm_route_logging as route_logging
    from litellm.proxy.aawm_route_logging import aresolve_aawm_route_host_attribution

    route_logging._aawm_route_host_reverse_dns_cache.clear()
    request = Mock(spec=Request)
    request.headers = {}
    request.client = SimpleNamespace(host="192.168.1.42", port=54321)
    request.scope = {"client": ("192.168.1.42", 54321)}

    monkeypatch.setattr(
        "litellm.proxy.auth.ip_address_utils.IPAddressUtils.get_mcp_client_ip",
        lambda request, general_settings=None: "192.168.1.42",
    )
    monkeypatch.setattr(
        route_logging,
        "_schedule_aawm_route_host_name_enrichment",
        Mock(),
    )

    async def fail_to_thread(*args, **kwargs):
        raise AssertionError("to_thread must not be used on non-blocking path")

    monkeypatch.setattr(route_logging.asyncio, "to_thread", fail_to_thread)

    attribution = asyncio.run(
        aresolve_aawm_route_host_attribution(
            request,
            allow_blocking_lookup=False,
        )
    )
    assert attribution["host_name"] == "192.168.1.42"
    assert attribution["host_name_source"] == "ip_literal"
