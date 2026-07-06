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
