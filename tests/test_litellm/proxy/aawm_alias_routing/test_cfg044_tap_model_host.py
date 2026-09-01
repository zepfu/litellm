"""CFG-044: one host override drives the eight TAP routes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from litellm.constants import AAWM_TAP_MODEL_HOST_DEFAULT

DEV_CONFIG_PATH = Path(__file__).resolve().parents[4] / "litellm-dev-config.yaml"
ENV_REFERENCE = "os.environ/AAWM_TAP_MODEL_HOST"

TAP_ROUTES: Dict[str, Dict[str, Any]] = {
    "tei-medcpt-article": {
        "model": "local_embed/ncbi/MedCPT-Article-Encoder",
        "mode": "embedding",
        "base_model": "ncbi/MedCPT-Article-Encoder",
        "port": 8083,
        "suffix": "/v1",
        "input_cost_per_token": 4.6e-09,
        "output_cost_per_token": 0.0,
    },
    "tei-medcpt-query": {
        "model": "local_embed/ncbi/MedCPT-Query-Encoder",
        "mode": "embedding",
        "base_model": "ncbi/MedCPT-Query-Encoder",
        "port": 8084,
        "suffix": "/v1",
        "input_cost_per_token": 2.8e-09,
        "output_cost_per_token": 0.0,
    },
    "specter2-adapter": {
        "model": "local_embed/allenai/specter2_base",
        "mode": "embedding",
        "base_model": "allenai/specter2_base",
        "port": 8086,
        "suffix": "/v1",
        "input_cost_per_token": 4.6e-09,
        "output_cost_per_token": 0.0,
    },
    "tei-indus": {
        "model": "local_embed/nasa-impact/nasa-ibm-st.38m",
        "mode": "embedding",
        "base_model": "nasa-impact/nasa-ibm-st.38m",
        "port": 8087,
        "suffix": "/v1",
        "input_cost_per_token": 5.6e-09,
        "output_cost_per_token": 0.0,
    },
    "tei-indus-v2": {
        "model": "local_embed/nasa-impact/nasa-smd-ibm-st-v2",
        "mode": "embedding",
        "base_model": "nasa-impact/nasa-smd-ibm-st-v2",
        "port": 8091,
        "suffix": "/v1",
        "input_cost_per_token": 5.6e-09,
        "output_cost_per_token": 0.0,
    },
    "tei-sapbert": {
        "model": "local_embed/cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        "mode": "embedding",
        "base_model": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        "port": 8088,
        "suffix": "/v1",
        "input_cost_per_token": 4.6e-09,
        "output_cost_per_token": 0.0,
    },
    "tei-sapbert-source": {
        "model": "local_embed/cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        "mode": "embedding",
        "base_model": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        "port": 8092,
        "suffix": "/v1",
        "input_cost_per_token": 4.6e-09,
        "output_cost_per_token": 0.0,
    },
    "tei-reranker": {
        "model": "local_rerank/BAAI/bge-reranker-v2-m3",
        "mode": "rerank",
        "base_model": "BAAI/bge-reranker-v2-m3",
        "port": 8090,
        "suffix": "",
        "input_cost_per_query": 0.0,
        "input_cost_per_token": 2.5e-08,
        "output_cost_per_token": 0.0,
    },
}


def _load_dev_config() -> dict:
    with DEV_CONFIG_PATH.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def _render_dev_config() -> dict:
    from litellm.proxy.proxy_server import ProxyConfig

    return ProxyConfig()._check_for_os_environ_vars(config=_load_dev_config())


def _deployment(config: dict, model_name: str) -> dict:
    for deployment in config["model_list"]:
        if deployment["model_name"] == model_name:
            return deployment
    raise AssertionError(f"{model_name} missing from {DEV_CONFIG_PATH}")


def _expected_source_api_base(route: Dict[str, Any]) -> str:
    return f"http://{ENV_REFERENCE}:{route['port']}{route['suffix']}"


def _expected_api_base(host: str, route: Dict[str, Any]) -> str:
    return f"http://{host}:{route['port']}{route['suffix']}"


@pytest.mark.parametrize("model_name", TAP_ROUTES)
def test_tap_route_source_contract(model_name: str) -> None:
    route = TAP_ROUTES[model_name]
    deployment = _deployment(_load_dev_config(), model_name)
    params = deployment["litellm_params"]
    model_info = deployment["model_info"]

    assert params["model"] == route["model"]
    assert params["api_base"] == _expected_source_api_base(route)
    assert params["api_key"] == "fake-api-key"
    assert model_info["mode"] == route["mode"]
    assert model_info["base_model"] == route["base_model"]

    for cost_key in (
        "input_cost_per_query",
        "input_cost_per_token",
        "output_cost_per_token",
    ):
        if cost_key in route:
            assert model_info[cost_key] == route[cost_key]


def test_default_render_uses_mahaf_for_all_tap_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AAWM_TAP_MODEL_HOST", raising=False)

    assert AAWM_TAP_MODEL_HOST_DEFAULT == "mahaf.tailf1878c.ts.net"
    rendered = _render_dev_config()

    for model_name, route in TAP_ROUTES.items():
        api_base = _deployment(rendered, model_name)["litellm_params"]["api_base"]
        assert api_base == _expected_api_base(AAWM_TAP_MODEL_HOST_DEFAULT, route)
        assert "172.20.0.1" not in api_base
        assert "host.docker.internal" not in api_base


def test_one_override_renders_all_tap_hosts_without_source_entry_edits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_before = _load_dev_config()
    source_routes_before = {
        model_name: _deployment(source_before, model_name)["litellm_params"]["api_base"] for model_name in TAP_ROUTES
    }

    monkeypatch.setenv("AAWM_TAP_MODEL_HOST", "tap-models.internal")
    rendered = _render_dev_config()

    for model_name, route in TAP_ROUTES.items():
        api_base = _deployment(rendered, model_name)["litellm_params"]["api_base"]
        assert api_base == _expected_api_base("tap-models.internal", route)

    source_after = _load_dev_config()
    source_routes_after = {
        model_name: _deployment(source_after, model_name)["litellm_params"]["api_base"] for model_name in TAP_ROUTES
    }
    assert source_routes_before == {
        model_name: _expected_source_api_base(route) for model_name, route in TAP_ROUTES.items()
    }
    assert source_routes_after == source_routes_before
