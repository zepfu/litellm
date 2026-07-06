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

    attribution = resolve_aawm_route_host_attribution(request)
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

    attribution = resolve_aawm_route_host_attribution(request)
    assert attribution["client_ip"] == "172.19.0.1"
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
    monkeypatch.setattr(
        "litellm.proxy.aawm_route_logging.socket.gethostbyaddr",
        Mock(side_effect=OSError("no reverse dns")),
    )

    attribution = resolve_aawm_route_host_attribution(request)
    assert attribution["client_ip"] == "192.168.1.42"
    assert attribution["host_name"] == "192.168.1.42"
    assert attribution["host_name_source"] == "ip_literal"


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

    attribution = resolve_aawm_route_host_attribution(request)
    assert attribution["client_ip"] == "100.99.1.5"
    assert attribution["host_name"] == "thoth"
    assert attribution["host_name_source"] in {"reverse_dns", "reverse_dns_cache"}


def test_attach_aawm_route_rollup_context_copies_host_metadata(monkeypatch):
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
    assert metadata["host_name_source"] == "reverse_dns"
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
