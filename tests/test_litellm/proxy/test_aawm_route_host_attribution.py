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


def test_resolve_aawm_route_host_attribution_uses_loopback_without_dns(monkeypatch):
    request = Mock(spec=Request)
    request.headers = {}
    request.client = SimpleNamespace(host="127.0.0.1", port=54321)
    request.scope = {"client": ("127.0.0.1", 54321)}

    monkeypatch.setattr(
        "litellm.proxy.auth.ip_address_utils.IPAddressUtils.get_mcp_client_ip",
        lambda request, general_settings=None: "127.0.0.1",
    )

    attribution = resolve_aawm_route_host_attribution(request)
    assert attribution["client_ip"] == "127.0.0.1"
    assert attribution["host_name"] == "localhost"
    assert attribution["host_name_source"] == "loopback"


def test_resolve_aawm_route_host_attribution_uses_docker_gateway_as_localhost(
    monkeypatch,
):
    request = Mock(spec=Request)
    request.headers = {}
    request.client = SimpleNamespace(host="172.19.0.1", port=54321)
    request.scope = {"client": ("172.19.0.1", 54321)}

    monkeypatch.setattr(
        "litellm.proxy.auth.ip_address_utils.IPAddressUtils.get_mcp_client_ip",
        lambda request, general_settings=None: "172.19.0.1",
    )

    attribution = resolve_aawm_route_host_attribution(request)
    assert attribution["client_ip"] == "172.19.0.1"
    assert attribution["host_name"] == "localhost"
    assert attribution["host_name_source"] == "docker_bridge_gateway"


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
