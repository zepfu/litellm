"""Harness v2: argparse, YAML config, docker_guard, and log checks."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence
from unittest.mock import patch

import pytest

_REPO = Path(__file__).resolve().parents[3]
_HV2 = _REPO / "scripts" / "harnessv2"
_FIXTURES = _HV2 / "fixtures" / "logs"
_ORCH_BASELINE_CHILDREN = (
    "basic",
    "work",
    "expert",
    "sota",
    "sota-xai",
    "sota-alibaba",
    "sota-moonshot",
    "sota-zai",
    "auto-review",
)

if str(_HV2) not in sys.path:
    sys.path.insert(0, str(_HV2))


def _load() -> Any:
    importlib.invalidate_caches()
    from hv2.cli import build_parser, parse_args, split_csv
    from hv2.docker_guard import (
        assert_container_allowed,
        assert_host_port_allowed,
        run_docker,
    )
    from hv2.checks.logs import leftover_uvicorn_allow_paths, leftover_uvicorn_regex, scan_log_text
    from hv2.checks.redis_scan import _assert_safe_redis
    from hv2.envscrub import scrubbed_child_env
    from hv2.errors import PlanError, ProtectedTargetError
    from hv2.instance import inspect_instance
    from hv2.kinds.runner import run_plan
    from hv2.load_config import load_config
    from hv2.plan import build_plan, compiled_aliases, expand_group

    return SimpleNamespace(
        build_parser=build_parser,
        parse_args=parse_args,
        split_csv=split_csv,
        assert_container_allowed=assert_container_allowed,
        assert_host_port_allowed=assert_host_port_allowed,
        run_docker=run_docker,
        leftover_uvicorn_allow_paths=leftover_uvicorn_allow_paths,
        leftover_uvicorn_regex=leftover_uvicorn_regex,
        scan_log_text=scan_log_text,
        assert_safe_redis=_assert_safe_redis,
        scrubbed_child_env=scrubbed_child_env,
        PlanError=PlanError,
        ProtectedTargetError=ProtectedTargetError,
        inspect_instance=inspect_instance,
        run_plan=run_plan,
        load_config=load_config,
        build_plan=build_plan,
        compiled_aliases=compiled_aliases,
        expand_group=expand_group,
    )


@pytest.fixture(scope="module")
def hv():
    return _load()


@pytest.fixture(scope="module")
def config(hv):
    return hv.load_config()


def test_should_require_test_flag(hv) -> None:
    parser = hv.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_should_treat_instance_container_target_as_the_same_flag(hv) -> None:
    parser = hv.build_parser()
    a = parser.parse_args(["--test", "platform", "--instance", "litellm-alpha"])
    b = parser.parse_args(["--test", "platform", "--container", "litellm-alpha"])
    c = parser.parse_args(["--test", "platform", "--target", "alpha"])
    assert a.instance == "litellm-alpha"
    assert b.instance == "litellm-alpha"
    assert c.instance == "alpha"
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--test", "platform", "--instance", "x", "--target", "y"]
        )


def test_should_split_comma_and_repeatable_models(hv) -> None:
    args = hv.parse_args(
        ["--test", "model", "--tui", "ohmypi", "--model", "work,expert", "--model", "sota"]
    )
    assert hv.split_csv(args.model) == ["work", "expert", "sota"]


def test_should_load_yaml_includes_and_compiled_aliases(hv, config) -> None:
    aliases = hv.compiled_aliases(config)
    assert "work" in aliases
    assert "sota-openai" in aliases
    assert "sota-zai" in aliases
    assert "aawm-sota-zai" not in aliases
    assert config["default_instance"] == "litellm-alpha"
    assert "aawm-litellm" in config["protected_containers"]
    assert "litellm-dev" in config["protected_containers"]
    assert 4000 in config["protected_ports"]
    assert 4001 in config["protected_ports"]
    assert config["tuis"]["out_of_scope"] == ["claude"]
    assert "ohmypi" in config["tuis"]["implemented"]
    assert "-p" in config["tuis"]["ohmypi"]["forbid_flags"]
    assert "--print" in config["tuis"]["ohmypi"]["forbid_flags"]
    assert "-p" not in config["tuis"]["ohmypi"]["argv_launch_model"]
    assert config["redis"]["container"] == "litellm-aawm-alias-routing-redis-1"
    assert config["redis"]["namespace"] == "aawm-routing-alpha-v1"
    assert config["redis"]["never_flush"] is True
    select_model = config["tuis"]["ohmypi"]["select_model"]
    provider_404 = select_model["provider_404_needles"]
    assert provider_404 == [
        "No endpoints found for",
        "404 Not Found",
        "status code 404",
        "Error: 404",
    ]
    assert select_model["reply_needles"] == ["※ recap:"] + provider_404
    assert select_model["reply_needles"] == _OHMYPI_MODEL_REPLY_NEEDLES
    assert "PONG" not in select_model["reply_needles"]
    assert "PONG" not in _OHMYPI_MODEL_REPLY_NEEDLES


def test_should_deep_merge_overlay(hv, tmp_path: Path) -> None:
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
        "redis:\n  ceilings:\n    warn_prefix_keys: 7\n",
        encoding="utf-8",
    )
    merged = hv.load_config(overlay=overlay)
    assert merged["redis"]["ceilings"]["warn_prefix_keys"] == 7
    assert merged["redis"]["namespace"] == "aawm-routing-alpha-v1"


def test_should_expand_model_groups_from_yaml(hv, config) -> None:
    all_models = hv.expand_group("all", config)
    assert all_models == hv.compiled_aliases(config)
    assert "claude-opus-5" not in all_models
    assert "auto-review" in all_models
    assert "auto-review" in hv.compiled_aliases(config)
    assert "codex-auto-review" in hv.compiled_aliases(config)
    picker = hv.expand_group("catalog_picker_sample", config)
    assert picker == ["work", "sota-zai"]
    children = hv.expand_group("orchestration_children", config)
    assert children == list(_ORCH_BASELINE_CHILDREN)
    assert "auto-review" in children
    assert "codex-auto-review" not in children
    assert "work-other" not in children
    assert "sota-deepseek" not in children
    assert config["tuis"]["ohmypi"]["orchestration_child_agents"] == list(
        _ORCH_BASELINE_CHILDREN
    )
    kinds_src = (_HV2 / "config" / "kinds.yaml").read_text(encoding="utf-8")
    assert "platform → catalog → orchestration" in kinds_src
    model_desc = str(config["kinds"]["model"].get("description") or "").lower()
    assert "optional" in model_desc or "not a baseline" in model_desc
    orch_desc = str(config["kinds"]["orchestration"].get("description") or "").lower()
    assert "extra" in orch_desc or "baseline children" in orch_desc


def test_should_refuse_claude_tui(hv, config) -> None:
    with pytest.raises(hv.PlanError, match="out of scope"):
        hv.build_plan(
            config=config,
            kind="model",
            instance_token="alpha",
            tui="claude",
            models=["work"],
            orchestration_parent=None,
            orchestration_children=None,
            dry_run=True,
            write_artifact=None,
        )


def test_should_refuse_stub_tui(hv, config) -> None:
    with pytest.raises(hv.PlanError, match="not implemented"):
        hv.build_plan(
            config=config,
            kind="model",
            instance_token="alpha",
            tui="codex",
            models=["work"],
            orchestration_parent=None,
            orchestration_children=None,
            dry_run=True,
            write_artifact=None,
        )


def test_should_forbid_tui_on_platform(hv, config) -> None:
    with pytest.raises(hv.PlanError, match="forbidden"):
        hv.build_plan(
            config=config,
            kind="platform",
            instance_token="alpha",
            tui="ohmypi",
            models=None,
            orchestration_parent=None,
            orchestration_children=None,
            dry_run=True,
            write_artifact=None,
        )


def test_should_require_tui_for_model(hv, config) -> None:
    with pytest.raises(hv.PlanError, match="required"):
        hv.build_plan(
            config=config,
            kind="model",
            instance_token="alpha",
            tui=None,
            models=["work"],
            orchestration_parent=None,
            orchestration_children=None,
            dry_run=True,
            write_artifact=None,
        )


def test_should_fail_closed_on_protected_container_names(hv, config) -> None:
    for name in ("aawm-litellm", "litellm-dev"):
        with pytest.raises(hv.ProtectedTargetError, match="protected"):
            hv.assert_container_allowed(name, config)
        with pytest.raises((hv.PlanError, hv.ProtectedTargetError)):
            hv.build_plan(
                config=config,
                kind="platform",
                instance_token=name,
                tui=None,
                models=None,
                orchestration_parent=None,
                orchestration_children=None,
                dry_run=True,
                write_artifact=None,
            )
    with pytest.raises(hv.PlanError, match="disabled"):
        hv.build_plan(
            config=config,
            kind="platform",
            instance_token="prod",
            tui=None,
            models=None,
            orchestration_parent=None,
            orchestration_children=None,
            dry_run=True,
            write_artifact=None,
        )


def test_should_refuse_protected_host_ports(hv, config) -> None:
    with pytest.raises(hv.ProtectedTargetError, match="4000"):
        hv.assert_host_port_allowed(4000, config)
    with pytest.raises(hv.ProtectedTargetError, match="4001"):
        hv.assert_host_port_allowed(4001, config)


def test_should_union_immutable_protected_targets_with_config_replacements() -> None:
    from hv2.docker_guard import protected_containers, protected_ports

    replacement = {
        "protected_containers": ["litellm-alpha"],
        "protected_ports": [4011],
    }
    containers = protected_containers(replacement)
    ports = protected_ports(replacement)
    assert isinstance(containers, frozenset)
    assert isinstance(ports, frozenset)
    assert containers == frozenset({"aawm-litellm", "litellm-dev", "litellm-alpha"})
    assert ports == frozenset({4000, 4001, 4011})
    assert protected_containers({}) == frozenset({"aawm-litellm", "litellm-dev"})
    assert protected_ports({}) == frozenset({4000, 4001})


def test_should_not_call_docker_for_protected_container(hv, config) -> None:
    with patch("hv2.docker_guard.subprocess.run") as mocked:
        with pytest.raises(hv.ProtectedTargetError):
            hv.run_docker(config, ["logs", "litellm-dev"], container="litellm-dev")
        mocked.assert_not_called()


def test_should_refuse_inspect_when_published_port_is_protected(hv, config) -> None:
    payload = {
        "State": {"Running": True},
        "Config": {"Cmd": ["litellm", "--port", "4000"], "Env": []},
        "NetworkSettings": {
            "Ports": {"4000/tcp": [{"HostIp": "127.0.0.1", "HostPort": "4000"}]}
        },
    }
    with pytest.raises(hv.ProtectedTargetError, match="4000"):
        hv.inspect_instance("not-prod", config, inspect_payload=payload)


def test_should_resolve_loopback_host_port_from_inspect(hv, config) -> None:
    payload = {
        "State": {"Running": True},
        "Config": {
            "Cmd": ["litellm", "--port", "4011"],
            "Env": ["AAWM_LITELLM_ENVIRONMENT=litellm-alpha"],
        },
        "NetworkSettings": {
            "Ports": {
                "4011/tcp": [
                    {"HostIp": "100.109.19.233", "HostPort": "4011"},
                    {"HostIp": "127.0.0.1", "HostPort": "4011"},
                ]
            }
        },
    }
    resolved = hv.inspect_instance("litellm-alpha", config, inspect_payload=payload)
    assert resolved.host_port == 4011
    assert resolved.host == "127.0.0.1"
    assert resolved.base_url == "http://127.0.0.1:4011"


def test_should_prefer_cmd_prefix_env_over_image_env(hv, config) -> None:
    payload = {
        "State": {"Running": True},
        "Config": {
            "Cmd": [
                "/usr/bin/env",
                "AAWM_LITELLM_ENVIRONMENT=litellm-alpha",
                "LITELLM_AAWM_ERROR_LOG_ENV=alpha",
                "watchfiles",
                "litellm --config /app/litellm-dev-config.yaml --port 4011",
            ],
            "Env": [
                "AAWM_LITELLM_ENVIRONMENT=litellm-dev",
                "LITELLM_AAWM_ERROR_LOG_ENV=dev",
                "AAWM_ALIAS_ROUTING_STATE_NAMESPACE=aawm-routing-alpha-v1",
            ],
        },
        "NetworkSettings": {
            "Ports": {"4011/tcp": [{"HostIp": "127.0.0.1", "HostPort": "4011"}]}
        },
    }
    resolved = hv.inspect_instance("litellm-alpha", config, inspect_payload=payload)
    assert resolved.inspect_env["AAWM_LITELLM_ENVIRONMENT"] == "litellm-alpha"
    assert resolved.inspect_env["LITELLM_AAWM_ERROR_LOG_ENV"] == "alpha"


def test_should_flag_forbidden_log_substrings(hv, config) -> None:
    text = (_FIXTURES / "forbidden_asgi.txt").read_text(encoding="utf-8")
    scan = hv.scan_log_text(text, config)
    assert scan["ok"] is False
    joined = " ".join(scan["failures"])
    assert "Exception in ASGI application" in joined
    assert "Traceback (most recent call last)" in joined


def test_should_flag_leftover_uvicorn_on_replaced_routes(hv, config) -> None:
    text = (_FIXTURES / "leftover_uvicorn.txt").read_text(encoding="utf-8")
    pattern = hv.leftover_uvicorn_regex(config)
    assert pattern is not None
    scan = hv.scan_log_text(text, config)
    assert scan["ok"] is False
    leftover = [
        hit["substring"]
        for hit in scan["forbidden_hits"]
        if hit["kind"] == "leftover_uvicorn"
    ]
    joined = " ".join(scan["failures"])
    assert "/openai_passthrough/v1/responses" in joined
    assert any("POST /openai_passthrough/v1/responses" in line for line in leftover)
    assert any("GET /openai_passthrough/v1/models" in line for line in leftover)
    assert any("POST /v1/chat/completions" in line for line in leftover)
    assert any("GET /openai_passthrough/v1/model/info" in line for line in leftover)
    assert any("GET /model_group/info" in line for line in leftover)


def test_should_flag_leftover_uvicorn_on_catalog_gets(hv, config) -> None:
    text = (
        'INFO:     172.18.0.1:50324 - "GET /openai_passthrough/v1/models HTTP/1.1" 200 OK\n'
        'INFO:     172.18.0.1:50325 - "GET /openai_passthrough/models HTTP/1.1" 200 OK\n'
    )
    scan = hv.scan_log_text(text, config)
    assert scan["ok"] is False
    leftover = [
        hit["substring"]
        for hit in scan["forbidden_hits"]
        if hit["kind"] == "leftover_uvicorn"
    ]
    joined = " ".join(scan["failures"])
    assert "GET /openai_passthrough/v1/models" in joined
    assert "GET /openai_passthrough/models" in joined
    assert any("GET /openai_passthrough/v1/models" in line for line in leftover)
    assert any("GET /openai_passthrough/models" in line for line in leftover)


def test_should_allow_leftover_uvicorn_on_health_paths(hv, config) -> None:
    assert "/health/liveliness" in hv.leftover_uvicorn_allow_paths(config)
    text = (
        "INFO:     Started server process [1]\n"
        "INFO:     Waiting for application startup.\n"
        "INFO:     Application startup complete.\n"
        "INFO:     Uvicorn running on http://0.0.0.0:4011 (Press CTRL+C to quit)\n"
        'INFO:     127.0.0.1:1 - "GET /health HTTP/1.1" 200 OK\n'
        'INFO:     127.0.0.1:1 - "GET /health/ HTTP/1.1" 200 OK\n'
        'INFO:     127.0.0.1:1 - "GET /health/liveliness HTTP/1.1" 200 OK\n'
        'INFO:     127.0.0.1:1 - "GET /health/liveliness?verbose=1 HTTP/1.1" 200 OK\n'
        'INFO:     127.0.0.1:1 - "GET /health/readiness HTTP/1.1" 200 OK\n'
        'INFO:     127.0.0.1:1 - "GET /health/services HTTP/1.1" 200 OK\n'
        'INFO:     127.0.0.1:1 - "HEAD /health/liveliness HTTP/1.1" 200 OK\n'
    )
    scan = hv.scan_log_text(text, config)
    assert scan["ok"] is True
    assert scan["failures"] == []
    leftover = [
        hit
        for hit in scan["forbidden_hits"]
        if hit["kind"] == "leftover_uvicorn"
    ]
    assert leftover == []


def test_should_flag_leftover_uvicorn_on_platform_http_suite_probes(
    hv, config
) -> None:
    required_paths = (
        "/internal/aawm/session-transfer-status",
        "/grok/v1",
        "/grok/v1/models",
        "/v1/chat/completions",
    )
    leftover_spec = config["checks"]["logs"]["leftover_uvicorn"]
    documented = leftover_spec.get("replaced_route_paths") or []
    allow_paths = leftover_spec.get("allow_paths") or []
    for path in required_paths:
        assert path in documented
        assert path not in allow_paths

    text = (
        'INFO:     172.18.0.1:33862 - "GET /internal/aawm/session-transfer-status HTTP/1.1" 403 Forbidden\n'
        'INFO:     172.18.0.1:33872 - "GET /grok/v1 HTTP/1.1" 403 Forbidden\n'
        'INFO:     172.18.0.1:33886 - "GET /grok/v1/models HTTP/1.1" 401 Unauthorized\n'
        'INFO:     172.18.0.1:33896 - "POST /v1/chat/completions HTTP/1.1" 400 Bad Request\n'
    )
    scan = hv.scan_log_text(text, config)
    assert scan["ok"] is False
    leftover = [
        hit["substring"]
        for hit in scan["forbidden_hits"]
        if hit["kind"] == "leftover_uvicorn"
    ]
    joined = " ".join(scan["failures"])
    for path in required_paths:
        assert path in joined
        assert any(path in line for line in leftover)


def test_should_flag_leftover_uvicorn_on_ohmypi_model_info_probes(hv, config) -> None:
    required_paths = (
        "/model_group/info",
        "/openai_passthrough/model_group/info",
        "/openai_passthrough/v1/model/info",
        "/openai_passthrough/model/info",
        "/openai_passthrough/v2/model/info",
    )
    leftover_spec = config["checks"]["logs"]["leftover_uvicorn"]
    documented = leftover_spec.get("replaced_route_paths") or []
    for path in required_paths:
        assert path in documented

    text = (
        'INFO:     172.18.0.1:50324 - "GET /model_group/info HTTP/1.1" 200 OK\n'
        'INFO:     172.18.0.1:50325 - "GET /openai_passthrough/model_group/info HTTP/1.1" 404 Not Found\n'
        'INFO:     172.18.0.1:50326 - "GET /openai_passthrough/v1/model/info HTTP/1.1" 404 Not Found\n'
        'INFO:     172.18.0.1:50327 - "GET /openai_passthrough/model/info HTTP/1.1" 404 Not Found\n'
        'INFO:     172.18.0.1:50328 - "GET /openai_passthrough/v2/model/info HTTP/1.1" 404 Not Found\n'
    )
    scan = hv.scan_log_text(text, config)
    assert scan["ok"] is False
    leftover = [
        hit["substring"]
        for hit in scan["forbidden_hits"]
        if hit["kind"] == "leftover_uvicorn"
    ]
    joined = " ".join(scan["failures"])
    expected_gets = tuple(f"GET {path}" for path in required_paths)
    for path in expected_gets:
        assert path in joined
        assert any(path in line for line in leftover)


def test_should_accept_clean_rollup_logs(hv, config) -> None:
    text = (_FIXTURES / "clean_rollup.txt").read_text(encoding="utf-8")
    scan = hv.scan_log_text(text, config, require_rollup=True)
    assert scan["ok"] is True
    assert scan["failures"] == []
    assert scan["rollup_hits"]


def test_should_fail_ohmypi_tui_rollup_without_client_name_version_and_repo(
    hv, config
) -> None:
    text = (_FIXTURES / "ohmypi_identity_miss.txt").read_text(encoding="utf-8")
    scan = hv.scan_log_text(
        text,
        config,
        require_rollup=True,
        tui="ohmypi",
    )
    assert scan["ok"] is False
    joined = " ".join(scan["failures"])
    assert "Ohmypi" in joined or "ohmypi" in joined.lower()
    assert "repository" in joined.lower() or "repo" in joined.lower() or "client" in joined.lower()
    assert any("Bun[" in item or "Oh@" in item for item in scan["rollup_hits"])


def test_should_accept_ohmypi_tui_rollup_with_name_version_and_repo(
    hv, config
) -> None:
    text = (_FIXTURES / "ohmypi_identity_ok.txt").read_text(encoding="utf-8")
    scan = hv.scan_log_text(
        text,
        config,
        require_rollup=True,
        tui="ohmypi",
    )
    assert scan["ok"] is True
    assert scan["failures"] == []
    assert any("Ohmypi[" in item and "#" in item for item in scan["rollup_hits"])


def test_should_accept_ohmypi_tui_rollup_when_concurrent_aawm_infrastructure_headers_are_present(
    hv, config
) -> None:
    text = (
        _FIXTURES / "ohmypi_identity_ok_with_concurrent_aawm_infrastructure.txt"
    ).read_text(encoding="utf-8")
    scan = hv.scan_log_text(
        text,
        config,
        require_rollup=True,
        tui="ohmypi",
    )
    assert scan["ok"] is True
    assert scan["failures"] == []
    assert any(
        "litellm#Ohmypi[" in item and "@" in item for item in scan["rollup_hits"]
    )
    assert any("aawm-infrastructure@" in item for item in scan["rollup_hits"])


def test_should_accept_ohmypi_tui_rollup_when_concurrent_codex_auto_review_litellm_at_host_headers_are_present(
    hv, config
) -> None:
    text = (
        _FIXTURES / "ohmypi_identity_ok_with_concurrent_codex_auto_review.txt"
    ).read_text(encoding="utf-8")
    scan = hv.scan_log_text(
        text,
        config,
        require_rollup=True,
        tui="ohmypi",
    )
    assert scan["ok"] is True
    assert scan["failures"] == []
    assert any(
        "litellm#Ohmypi[" in item and "@" in item for item in scan["rollup_hits"]
    )
    assert any(
        item.startswith("20260823 19:46:35 litellm@thoth /openai_passthrough/responses")
        for item in scan["rollup_hits"]
    )
    assert any(
        item.startswith("20260823 19:47:35 litellm@thoth /openai_passthrough/responses")
        for item in scan["rollup_hits"]
    )
    assert "codex-auto-review" in text


def test_should_still_fail_ohmypi_tui_rollup_when_unlabeled_ohmypi_mixes_with_labeled_ohmypi(
    hv, config
) -> None:
    text = (
        "20260823 17:35:57 litellm#Ohmypi[17.4.2]@thoth /v1/chat/completions\n"
        " - gpt-5.6-sol(sota-openai):max - Turns: 5\n"
        "20260823 17:36:00 Oh@thoth /openai_passthrough/v1/responses\n"
        " - gpt-5.6-sol(sota-openai):max - Turns: 1\n"
        "20260823 17:36:01 Bun[1.3.14]@thoth /v1/chat/completions\n"
        " - openrouter/qwen/qwen3.6-flash:none - Turns: 4\n"
        "20260823 17:36:02 litellm@thoth /v1/chat/completions\n"
        " - gpt-5.6-sol:xhigh - Turns: 3\n"
    )
    scan = hv.scan_log_text(
        text,
        config,
        require_rollup=True,
        tui="ohmypi",
    )
    assert scan["ok"] is False
    joined = " ".join(scan["failures"])
    assert "Ohmypi" in joined or "ohmypi" in joined.lower()
    assert any("Oh@" in item for item in scan["rollup_hits"])
    assert any("Bun[" in item for item in scan["rollup_hits"])
    assert any(
        item.startswith("20260823 17:36:02 litellm@thoth")
        for item in scan["rollup_hits"]
    )
    assert any("litellm#Ohmypi[" in item for item in scan["rollup_hits"])


def test_should_warn_not_fail_on_expected_work_miss_traceback(hv, config) -> None:
    text = (
        "LiteLLM Proxy:ERROR: You passed in model=work. There are no healthy "
        "deployments for this model.\n"
        "Traceback (most recent call last):\n"
        '  File "/app/litellm/proxy/proxy_server.py", line 7046, in chat_completion\n'
    )
    scan = hv.scan_log_text(text, config)
    assert scan["ok"] is True
    assert scan["failures"] == []
    assert any("expected log signature" in item for item in scan["warnings"])


def test_should_still_fail_unrelated_traceback_after_expected_work_miss(
    hv, config
) -> None:
    padding = "x" * 4000
    text = (
        "You passed in model=work. There are no healthy deployments for this model.\n"
        "Traceback (most recent call last):\n"
        f"{padding}\n"
        "Exception in ASGI application\n"
        "Traceback (most recent call last):\n"
        "  File boom.py, line 1\n"
    )
    scan = hv.scan_log_text(text, config)
    assert scan["ok"] is False
    joined = " ".join(scan["failures"])
    assert "Traceback (most recent call last)" in joined
    assert "Exception in ASGI application" in joined


def test_should_scrub_langfuse_and_db_secrets_from_child_env(hv, config, monkeypatch) -> None:
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("HOME", "/home/test")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-should-not-leak")
    monkeypatch.setenv("DATABASE_URL", "postgres://user:pass@localhost/db")
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-master")
    monkeypatch.setenv("LITELLM_BASE_URL", "http://127.0.0.1:4011")
    monkeypatch.setenv("PI_CONFIG_FILES", "/tmp/alpha.yml")
    monkeypatch.setenv("UNRELATED_RANDOM", "nope")
    env = hv.scrubbed_child_env(config)
    assert env["PATH"] == "/usr/bin"
    assert env["LITELLM_BASE_URL"] == "http://127.0.0.1:4011"
    assert env["PI_CONFIG_FILES"] == "/tmp/alpha.yml"
    for denied in (
        "LANGFUSE_PUBLIC_KEY",
        "DATABASE_URL",
        "LITELLM_MASTER_KEY",
        "UNRELATED_RANDOM",
    ):
        assert denied not in env


def test_should_stage_ohmypi_identity_overlay_with_repo_and_version(hv, config) -> None:
    from hv2.drivers.ohmypi import OhmypiDriver

    driver = OhmypiDriver(config)
    overlay = driver.identity_overlay_payload(version="17.4.2")
    providers = overlay["providers"]
    for name in ("litellm-alpha", "litellm-alpha-passthrough"):
        headers = providers[name]["headers"]
        assert headers["x-aawm-client"] == "Oh My Pi"
        assert headers["x-aawm-client-name"] == "omp"
        assert headers["x-aawm-client-version"] == "17.4.2"
        assert headers["x-aawm-repository"] == "litellm"
    argv = driver.launch_argv("work")
    assert "--config" in argv
    config_path = argv[argv.index("--config") + 1]
    assert "ohmypi-identity" in config_path or config_path.endswith(".yml")


def test_should_export_ohmypi_identity_overlay_on_pi_config_files_for_child_sessions(
    hv, config, tmp_path: Path
) -> None:
    from hv2.drivers.ohmypi import OhmypiDriver

    cfg = _clone_config(config)
    session_dir = tmp_path / "omp-sessions"
    cfg["tuis"]["ohmypi"]["session_dir"] = str(session_dir)
    driver = OhmypiDriver(cfg)
    identity = str(driver.write_identity_overlay())
    env = driver.child_env()
    pi_config = env["PI_CONFIG_FILES"]
    components = pi_config.split(":")
    assert components[0] == identity
    operator = str(Path.home() / ".omp" / "agent" / "litellm-alpha.yml")
    if operator and operator != identity:
        assert operator in components[1:]
        assert pi_config == f"{identity}:{operator}"
    else:
        assert pi_config == identity


def test_should_refuse_ohmypi_print_flags(hv, config) -> None:
    from hv2.drivers.ohmypi import OhmypiDriver

    driver = OhmypiDriver(config)
    argv = driver.launch_argv("work")
    assert "-p" not in argv
    assert "--print" not in argv
    assert "litellm-alpha-passthrough/work" in argv
    with pytest.raises(hv.PlanError, match="forbidden flag"):
        driver.assert_no_print_flags(["omp", "-p", "hi"])
    tmux_argv = [
        "tmux",
        "-L",
        "tmux37",
        "new-session",
        "-d",
        "-s",
        "hv2-ohmypi-work",
        "--",
        *argv,
    ]
    driver.assert_no_print_flags(tmux_argv)
    assert "-p" not in tmux_argv
    assert "--print" not in tmux_argv


def test_should_dry_run_platform_without_docker(hv, config) -> None:
    plan = hv.build_plan(
        config=config,
        kind="platform",
        instance_token="alpha",
        tui=None,
        models=None,
        orchestration_parent=None,
        orchestration_children=None,
        dry_run=True,
        write_artifact=None,
    )
    assert plan.container == "litellm-alpha"
    assert plan.requires_tui is False
    with patch("hv2.kinds.runner.run_docker") as docker_run:
        artifact = hv.run_plan(plan)
        docker_run.assert_not_called()
    assert artifact["ok"] is True
    assert artifact["dry_run"] is True
    assert artifact["plan"]["container"] == "litellm-alpha"


def test_should_plan_ohmypi_model_all_from_yaml(hv, config) -> None:
    plan = hv.build_plan(
        config=config,
        kind="model",
        instance_token="litellm-alpha",
        tui="ohmypi",
        models=["all"],
        orchestration_parent=None,
        orchestration_children=None,
        dry_run=True,
        write_artifact=None,
    )
    assert plan.tui == "ohmypi"
    assert list(plan.models) == hv.compiled_aliases(config)
    assert "auto-review" in plan.models
    assert "codex-auto-review" in plan.models
    assert "PONG" in plan.extra["pong_prompt"]


def test_should_plan_model_kind_including_auto_review_compiled_alias(hv, config) -> None:
    aliases = hv.compiled_aliases(config)
    assert "auto-review" in aliases
    assert "codex-auto-review" in aliases
    plan = hv.build_plan(
        config=config,
        kind="model",
        instance_token="alpha",
        tui="ohmypi",
        models=["auto-review"],
        orchestration_parent=None,
        orchestration_children=None,
        dry_run=True,
        write_artifact=None,
    )
    assert plan.kind == "model"
    assert plan.models == ("auto-review",)
    assert "auto-review" in hv.expand_group("all", config)


def test_should_plan_orchestration_parents_and_children(hv, config) -> None:
    plan = hv.build_plan(
        config=config,
        kind="orchestration",
        instance_token="alpha",
        tui="ohmypi",
        models=None,
        orchestration_parent="all-sota",
        orchestration_children=None,
        dry_run=True,
        write_artifact=None,
    )
    assert "sota-openai" in plan.orchestration_parents
    assert "sota-moonshot" in plan.orchestration_parents
    assert list(plan.orchestration_children) == list(_ORCH_BASELINE_CHILDREN)
    assert "work-other" not in plan.orchestration_children
    assert "sota-deepseek" not in plan.orchestration_children
    assert "codex-auto-review" not in plan.orchestration_children
    prompt = plan.extra["orchestration_prompt_template"]
    assert "agent=sota-xai" in prompt
    assert "agent=auto-review" in prompt
    assert "agent=codex-auto-review" not in prompt
    assert "model=basic" not in prompt
    assert "Spawn exactly four" not in prompt
    assert "PONG" in prompt


def test_should_refuse_redis_flush(hv) -> None:
    with pytest.raises(hv.ProtectedTargetError, match="FLUSHALL"):
        hv.assert_safe_redis(["FLUSHALL"])


def test_error_jsonl_scanner_reads_existing_file_and_does_not_write(tmp_path: Path) -> None:
    from hv2.checks.error_jsonl import scan_new_rows

    path = tmp_path / "alpha-error.jsonl"
    path.write_text(
        '{"message":"prior","traceback":null}\n',
        encoding="utf-8",
    )
    before = path.stat().st_size
    path.write_text(
        path.read_text(encoding="utf-8")
        + '{"message":"boom","traceback":"Traceback (most recent call last):\\nValueError"}\n',
        encoding="utf-8",
    )
    config = {
        "checks": {
            "error_jsonl": {
                "path": str(path),
                "traceback_null_is_warning": True,
            }
        }
    }
    result = scan_new_rows(config, before_size=before)
    assert result["ok"] is False
    assert any("traceback" in item for item in result["failures"])
    assert result["rows"][0]["message"] == "boom"
    # The harness scanner is a reader. It must not create or rewrite the sink.
    assert path.is_file()


def test_should_treat_jsonl_null_traceback_as_warning() -> None:
    from hv2.checks.error_jsonl import scan_new_rows

    config = {
        "checks": {
            "error_jsonl": {
                "path": str(_REPO / ".analysis" / "alpha-error.jsonl"),
                "traceback_null_is_warning": True,
            }
        }
    }
    # Empty new slice: cursor at EOF means no new rows.
    from hv2.checks import error_jsonl as mod

    path = mod.jsonl_path(config)
    before = path.stat().st_size if path.is_file() else 0
    result = scan_new_rows(config, before_size=before)
    assert result["ok"] is True
    assert result["rows"] == []


def test_run_py_does_not_import_local_ci() -> None:
    text = (_HV2 / "run.py").read_text(encoding="utf-8")
    assert "local-ci" not in text
    assert "run_anthropic_adapter" not in text
    python_files = list((_HV2 / "hv2").rglob("*.py"))
    joined = "\n".join(path.read_text(encoding="utf-8") for path in python_files)
    assert "scripts.local-ci" not in joined
    assert "aawm-litellm" not in joined or "protected" in joined


class _FakeHTTPResponse:
    def __init__(self, body: bytes = b"{}", status: int = 200) -> None:
        self._body = body
        self.status = status

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None


def _clone_config(config: dict[str, Any]) -> dict[str, Any]:
    import copy

    return copy.deepcopy(config)


def _request_headers(request: Any) -> dict[str, str]:
    raw: dict[str, str] = {}
    items: list[Any] = []
    if hasattr(request, "header_items"):
        items = list(request.header_items())
    if not items:
        for attr in ("headers", "unredirected_hdrs"):
            headers = getattr(request, attr, None) or {}
            if hasattr(headers, "items"):
                items.extend(list(headers.items()))
    for key, value in items:
        raw[str(key).lower()] = str(value)
    return raw


def _fake_urlopen_map(
    path_to_status_body: dict[str, tuple[int, Any]],
    captured: list[Any] | None = None,
):
    def _fake(request: Any, timeout: Any = None) -> _FakeHTTPResponse:
        if captured is not None:
            captured.append(request)
        from urllib.parse import urlparse

        path = urlparse(request.full_url).path
        if path not in path_to_status_body:
            raise AssertionError(
                f"unexpected urlopen path {path} url={request.full_url}"
            )
        status, body = path_to_status_body[path]
        if body is None:
            raw = b""
        elif isinstance(body, (bytes, bytearray)):
            raw = bytes(body)
        else:
            import json as json_mod

            raw = json_mod.dumps(body).encode("utf-8")
        return _FakeHTTPResponse(raw, status)

    return _fake


def _http_suite_config(
    config: dict[str, Any],
    *,
    names: set[str] | None = None,
    rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    cfg = _clone_config(config)
    checks = cfg.setdefault("checks", {})
    http = checks.setdefault("http", {})
    if rows is not None:
        http["suite"] = rows
        return cfg
    suite = http.get("suite") if isinstance(http.get("suite"), list) else []
    http["suite"] = [row for row in suite if row.get("name") in (names or set())]
    return cfg


def _catalog_payload(ids: list[str]) -> dict[str, Any]:
    return {"data": [{"id": name} for name in ids]}


def _catalog_id_set(hv: Any, config: dict[str, Any], *, drop: set[str], extra: list[str]) -> list[str]:
    models = config.get("models") if isinstance(config.get("models"), dict) else {}
    served = [str(item) for item in (models.get("served_concrete_ids") or [])]
    aliases = [name for name in hv.compiled_aliases(config) if name not in drop]
    out: list[str] = []
    seen: set[str] = set()
    for name in [*aliases, *served, *extra]:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def test_should_check_health_using_yaml_liveliness_path(hv, config) -> None:
    from hv2.checks.http import check_health

    captured: list[Any] = []
    with patch(
        "hv2.checks.http.urlopen",
        side_effect=_fake_urlopen_map(
            {"/health/liveliness": (200, {"status": "ok"})},
            captured,
        ),
    ):
        result = check_health(config, "http://127.0.0.1:4011")
    assert result["ok"] is True
    assert result["status"] == 200
    assert result["failures"] == []
    assert captured, "urlopen was not called"
    assert captured[0].full_url == "http://127.0.0.1:4011/health/liveliness"
    assert config["checks"]["health"]["path"] == "/health/liveliness"
    assert 200 in [int(item) for item in config["checks"]["health"]["expect_status"]]


def test_should_pass_http_suite_when_generic_work_miss_returns_404(hv, config) -> None:
    from hv2.checks.http import check_http_suite

    cfg = _http_suite_config(
        config,
        names={"generic_work_miss", "session_transfer_status_no_identity"},
    )
    with patch(
        "hv2.checks.http.urlopen",
        side_effect=_fake_urlopen_map(
            {
                "/v1/chat/completions": (404, {"error": {"message": "no healthy deployments"}}),
                "/internal/aawm/session-transfer-status": (404, {"error": "missing identity"}),
            }
        ),
    ):
        result = check_http_suite(cfg, "http://127.0.0.1:4011")
    assert result["ok"] is True
    assert result["failures"] == []
    by_name = {row["name"]: row for row in result["results"]}
    assert by_name["generic_work_miss"]["status"] == 404
    assert by_name["generic_work_miss"]["ok"] is True
    assert by_name["session_transfer_status_no_identity"]["status"] == 404
    assert by_name["session_transfer_status_no_identity"]["ok"] is True


def test_should_fail_http_suite_when_generic_work_miss_returns_500(hv, config) -> None:
    from hv2.checks.http import check_http_suite

    cfg = _http_suite_config(config, names={"generic_work_miss"})
    with patch(
        "hv2.checks.http.urlopen",
        side_effect=_fake_urlopen_map(
            {"/v1/chat/completions": (500, {"error": "upstream"})}
        ),
    ):
        result = check_http_suite(cfg, "http://127.0.0.1:4011")
    assert result["ok"] is False
    joined = " ".join(result["failures"])
    assert "generic_work_miss" in joined
    assert "500" in joined


def test_should_fail_http_suite_when_session_transfer_returns_500(hv, config) -> None:
    from hv2.checks.http import check_http_suite

    cfg = _http_suite_config(config, names={"session_transfer_status_no_identity"})
    with patch(
        "hv2.checks.http.urlopen",
        side_effect=_fake_urlopen_map(
            {"/internal/aawm/session-transfer-status": (500, {"error": "boom"})}
        ),
    ):
        result = check_http_suite(cfg, "http://127.0.0.1:4011")
    assert result["ok"] is False
    joined = " ".join(result["failures"])
    assert "session_transfer_status_no_identity" in joined
    assert "500" in joined
    assert "rejected" in joined


def test_should_fail_catalog_when_work_missing_or_aawm_sota_zai_published(
    hv, config
) -> None:
    from hv2.checks.http import check_catalog_http, check_http_suite

    ids = _catalog_id_set(hv, config, drop={"work"}, extra=["aawm-sota-zai"])
    cfg = _http_suite_config(
        config,
        rows=[
            {
                "name": "passthrough_v1_models",
                "method": "GET",
                "path": "/openai_passthrough/v1/models",
                "expect_status": [200],
                "catalog": "compiled_aliases",
            }
        ],
    )
    payload = _catalog_payload(ids)
    with patch(
        "hv2.checks.http.urlopen",
        side_effect=_fake_urlopen_map(
            {"/openai_passthrough/v1/models": (200, payload)}
        ),
    ):
        catalog = check_catalog_http(cfg, "http://127.0.0.1:4011")
        suite = check_http_suite(cfg, "http://127.0.0.1:4011")
    for result in (catalog, suite):
        assert result["ok"] is False
        joined = " ".join(result["failures"])
        assert "work" in joined
        assert "aawm-sota-zai" in joined


def test_should_not_apply_catalog_assertions_on_non_catalog_models_path(
    hv, config
) -> None:
    from hv2.checks.http import check_http_suite

    cfg = _http_suite_config(
        config,
        rows=[
            {
                "name": "vendor_models",
                "method": "GET",
                "path": "/vendor/models",
                "expect_status": [200],
            }
        ],
    )
    payload = _catalog_payload(["unrelated-model", "aawm-sota-zai"])
    with patch(
        "hv2.checks.http.urlopen",
        side_effect=_fake_urlopen_map({"/vendor/models": (200, payload)}),
    ):
        result = check_http_suite(cfg, "http://127.0.0.1:4011")
    assert result["ok"] is True
    joined = " ".join(result["failures"])
    assert "compiled aliases" not in joined
    assert "unexpectedly published" not in joined


def test_should_honor_catalog_yaml_flag_on_non_passthrough_path(hv, config) -> None:
    from hv2.checks.http import check_http_suite

    ids = _catalog_id_set(hv, config, drop={"work"}, extra=["aawm-sota-zai"])
    cfg = _http_suite_config(
        config,
        rows=[
            {
                "name": "alias_catalog",
                "method": "GET",
                "path": "/internal/alias-catalog",
                "expect_status": [200],
                "catalog": "compiled_aliases",
            }
        ],
    )
    with patch(
        "hv2.checks.http.urlopen",
        side_effect=_fake_urlopen_map(
            {"/internal/alias-catalog": (200, _catalog_payload(ids))}
        ),
    ):
        result = check_http_suite(cfg, "http://127.0.0.1:4011")
    assert result["ok"] is False
    joined = " ".join(result["failures"])
    assert "work" in joined
    assert "aawm-sota-zai" in joined


def test_should_inject_identity_headers_on_http_requests(hv, config) -> None:
    from hv2.checks.http import request_json

    assert config["checks"]["identity_headers"]
    captured: list[Any] = []
    with patch(
        "hv2.checks.http.urlopen",
        side_effect=_fake_urlopen_map(
            {"/health/liveliness": (200, {"status": "ok"})},
            captured,
        ),
    ):
        result = request_json(
            config,
            base_url="http://127.0.0.1:4011",
            method="GET",
            path="/health/liveliness",
        )
    assert result["status"] == 200
    assert captured, "urlopen was not called"
    headers = _request_headers(captured[0])
    assert "x-litellm-end-user-id" in headers
    assert "x-aawm-client" in headers
    assert headers["x-aawm-client"]


def test_should_snapshot_redis_prefix_without_flushall(hv, config) -> None:
    from hv2.checks.redis_scan import snapshot_redis

    calls: list[list[str]] = []

    def fake_run_docker(
        _config: Any, args: Any, *, container: str, **_kwargs: Any
    ) -> Any:
        argv = [str(item) for item in args]
        calls.append(argv)
        joined = " ".join(argv).upper()
        if "INFO" in joined:
            return SimpleNamespace(
                returncode=0,
                stdout="# Memory\nused_memory:4096\nused_memory_human:4.00K\n",
                stderr="",
            )
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "aawm-routing-alpha-v1:alpha\n"
                "aawm-routing-alpha-v1:beta\n"
                "aawm-routing-alpha-v1:gamma\n"
            ),
            stderr="",
        )

    with patch("hv2.checks.redis_scan.run_docker", side_effect=fake_run_docker):
        result = snapshot_redis(config)
    assert result["ok"] is True
    assert result["used_memory_bytes"] == 4096
    assert result["prefix_key_count"] == 3
    assert result["failures"] == []
    assert calls, "run_docker was not called"
    for argv in calls:
        assert "FLUSHALL" not in " ".join(argv).upper()


def test_should_exclude_skip_prefixes_from_expand_group_all(hv, config) -> None:
    all_models = hv.expand_group("all", config)
    assert all_models
    for model_id in all_models:
        assert not model_id.startswith("claude-")
        assert not model_id.startswith("aawm-")
    cfg = _clone_config(config)
    cfg["models"]["compiled_aliases"] = [
        "work",
        "expert",
        "claude-opus-5",
        "aawm-sota-zai",
    ]
    cfg["models"]["skip_prefixes"] = ["aawm-", "claude-"]
    filtered = hv.expand_group("all", cfg)
    assert "work" in filtered
    assert "expert" in filtered
    assert "claude-opus-5" not in filtered
    assert "aawm-sota-zai" not in filtered
    for model_id in filtered:
        assert not model_id.startswith("claude-")
        assert not model_id.startswith("aawm-")


def test_should_refuse_explicit_skipped_prefix_model(hv, config) -> None:
    with pytest.raises(hv.PlanError):
        hv.build_plan(
            config=config,
            kind="model",
            instance_token="alpha",
            tui="ohmypi",
            models=["claude-opus-5"],
            orchestration_parent=None,
            orchestration_children=None,
            dry_run=True,
            write_artifact=None,
        )


def test_should_require_explicit_model_when_kind_flag_set(hv, config) -> None:
    cfg = _clone_config(config)
    cfg["kinds"]["model"]["require_explicit_model"] = True
    with pytest.raises(hv.PlanError):
        hv.build_plan(
            config=cfg,
            kind="model",
            instance_token="alpha",
            tui="ohmypi",
            models=None,
            orchestration_parent=None,
            orchestration_children=None,
            dry_run=True,
            write_artifact=None,
        )


def test_should_expand_orchestration_default_parent_group(hv, config) -> None:
    cfg = _clone_config(config)
    orch = cfg["kinds"]["orchestration"]
    orch.pop("default_parent", None)
    orch["default_parent_group"] = "all-sota"
    plan = hv.build_plan(
        config=cfg,
        kind="orchestration",
        instance_token="alpha",
        tui="ohmypi",
        models=None,
        orchestration_parent=None,
        orchestration_children=None,
        dry_run=True,
        write_artifact=None,
    )
    assert list(plan.orchestration_parents) == hv.expand_group("all-sota", cfg)


def test_should_inject_harness_user_id_and_drop_anthropic_key(
    hv, config, monkeypatch
) -> None:
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("HOME", "/home/test")
    monkeypatch.delenv("AAWM_HARNESS_USER_ID", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    env = hv.scrubbed_child_env(config)
    assert env["AAWM_HARNESS_USER_ID"] == config["default_harness_user_id"]
    assert "ANTHROPIC_API_KEY" not in env


def test_should_refuse_protected_url_without_calling_urlopen(hv, config) -> None:
    from hv2.checks.http import request_json

    with patch("hv2.checks.http.urlopen") as mocked:
        with pytest.raises(hv.ProtectedTargetError, match="4000"):
            request_json(
                config,
                base_url="http://127.0.0.1:4000",
                method="GET",
                path="/health/liveliness",
            )
        mocked.assert_not_called()


def test_should_warn_on_sha_drift_without_invalidating(hv) -> None:
    from hv2.artifact import sha_drift_warning

    warning = sha_drift_warning(
        {"commit": "aaa", "dirty": "false"},
        {"commit": "bbb", "dirty": "false"},
    )
    assert warning is not None
    assert "STRONG WARNING" in warning
    assert "not invalidated" in warning
    assert (
        sha_drift_warning(
            {"commit": "aaa", "dirty": "false"},
            {"commit": "aaa", "dirty": "false"},
        )
        is None
    )
    dirty_flip = sha_drift_warning(
        {"commit": "aaa", "dirty": "false"},
        {"commit": "aaa", "dirty": "true"},
    )
    assert dirty_flip is not None


def test_should_halt_remaining_steps_on_asgi_traceback(hv, config, tmp_path: Path) -> None:
    from hv2.kinds.runner import run_plan

    cfg = _clone_config(config)
    cfg["artifact"]["durable_dir"] = str(tmp_path)
    plan = hv.build_plan(
        config=cfg,
        kind="platform",
        instance_token="alpha",
        tui=None,
        models=None,
        orchestration_parent=None,
        orchestration_children=None,
        dry_run=False,
        write_artifact=tmp_path / "summary.json",
    )
    from hv2.instance import ResolvedInstance

    resolved = ResolvedInstance(
        alias="alpha",
        container="litellm-alpha",
        host="127.0.0.1",
        host_port=4011,
        container_port=4011,
        base_url="http://127.0.0.1:4011",
        inspect_env={},
        running=True,
    )
    object.__setattr__(plan, "resolved", resolved)
    asgi = (_FIXTURES / "forbidden_asgi.txt").read_text(encoding="utf-8")

    def fake_health(_plan, **_kwargs):
        return {"ok": True, "failures": [], "status": 200}

    def fake_http(_plan, **_kwargs):
        return {"ok": True, "failures": []}

    def fake_jsonl(_plan, **_kwargs):
        return {"ok": True, "failures": []}

    def fake_redis(_plan, **_kwargs):
        return {"ok": True, "failures": []}

    def fake_logs(_plan, **_kwargs):
        return hv.scan_log_text(asgi, cfg)

    with patch.dict(
        "hv2.kinds.runner._STEP_HANDLERS",
        {
            "health": fake_health,
            "http_suite": fake_http,
            "error_jsonl": fake_jsonl,
            "redis_scan": fake_redis,
            "docker_logs": fake_logs,
        },
    ):
        artifact = run_plan(plan)
    assert artifact["ok"] is False
    assert artifact["halted"] is True
    names = [row["name"] for row in artifact["results"]]
    assert "docker_logs" in names
    docker = next(row for row in artifact["results"] if row["name"] == "docker_logs")
    joined = " ".join(docker.get("failures") or [])
    assert "Exception in ASGI application" in joined
    jsonl_path = Path(artifact["durable_jsonl"])
    assert jsonl_path.is_file()
    lines = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines[0]["event"] == "run_start"
    assert lines[0]["git"]["commit"]
    assert lines[-1]["event"] == "run_end"
    assert lines[-1]["halted"] is True
    step_results = [row["result"] for row in lines if row.get("event") == "step"]
    assert "fail" in step_results


def test_should_launch_ohmypi_session_with_tmux_env_and_selected_alias(
    hv, config, monkeypatch
) -> None:
    from hv2.drivers.ohmypi import OhmypiDriver

    driver = OhmypiDriver(config)
    calls: list[list[str]] = []

    def fake_run(args: Any, *, timeout: int = 10) -> Any:
        calls.append([str(item) for item in args])
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(driver, "_run_tmux", fake_run)
    monkeypatch.setattr(driver, "tmux_has_session", lambda name=None: False)
    monkeypatch.setattr(driver, "wait_for_pane", lambda *a, **k: True)
    monkeypatch.setattr(
        driver,
        "capture_pane",
        lambda: "Default model: litellm-alpha-passthrough/work\nπ  >",
    )
    monkeypatch.setattr(driver, "ensure_workspace", lambda: None)
    launched = driver.ensure_session("work", tools=False)
    assert launched["ok"] is True
    assert launched["selected"] is True
    assert launched["selector"] == "litellm-alpha-passthrough/work"
    assert "-p" not in launched["argv"]
    assert "--print" not in launched["argv"]
    assert "--no-tools" in launched["argv"]
    new_session = next(row for row in calls if row and row[0] == "new-session")
    assert "omp-alpha-test" not in new_session
    assert "-p" not in new_session
    assert "--print" not in new_session
    env_values = [
        new_session[index + 1]
        for index, token in enumerate(new_session)
        if token == "-e" and index + 1 < len(new_session)
    ]
    assert any(item.startswith("PI_CONFIG_FILES=") for item in env_values)
    assert "litellm-alpha-passthrough/work" in new_session
    assert "--config" in new_session


def test_should_refuse_ohmypi_operator_session_reuse(hv, config) -> None:
    from hv2.drivers.ohmypi import OhmypiDriver

    cfg = _clone_config(config)
    cfg["tuis"]["ohmypi"]["select_model"]["reuse_operator_session"] = True
    driver = OhmypiDriver(cfg)
    with pytest.raises(hv.PlanError, match="omp-alpha-test"):
        driver.ensure_session("work", tools=False)


def test_should_expand_ohmypi_catalog_find_argv(hv, config) -> None:
    from hv2.drivers.ohmypi import OhmypiDriver

    driver = OhmypiDriver(config)
    argv = driver.expand_argv("catalog_find_argv", {"pattern": "work"})
    assert argv[:3] == ["omp", "models", "find"]
    assert "work" in argv
    assert "--json" in argv
    assert "-p" not in argv


def test_should_paste_multiline_ohmypi_prompt_instead_of_send_keys(
    hv, config, monkeypatch
) -> None:
    from hv2.drivers.ohmypi import OhmypiDriver

    driver = OhmypiDriver(config)
    driver._active_session = "hv2-ohmypi-sota-openai-1"
    calls: list[tuple[list[str], str | None]] = []

    def fake_run(args: Any, *, timeout: int = 10, stdin_text: str | None = None) -> Any:
        calls.append(([str(item) for item in args], stdin_text))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(driver, "_run_tmux", fake_run)
    sent = driver.send_keys("line one\nline two\nline three")
    assert sent["ok"] is True
    assert sent["method"] == "paste-buffer"
    assert any(row[0][:1] == ["load-buffer"] for row in calls)
    assert any(row[0][:1] == ["paste-buffer"] for row in calls)
    assert any(row[0] == ["send-keys", "-t", "hv2-ohmypi-sota-openai-1", "Enter"] for row in calls)
    loaded = next(row for row in calls if row[0][:1] == ["load-buffer"])
    assert "line one" in (loaded[1] or "")
    assert "-p" not in " ".join(token for row, _stdin in calls for token in row)

    calls.clear()
    sent = driver.send_keys("Reply with exactly the word PONG.")
    assert sent["method"] == "send-keys"
    assert calls[0][0][:1] == ["send-keys"]
    assert "Enter" in calls[0][0]


def _orchestration_sent_prompt() -> str:
    path = _HV2 / "config" / "prompts" / "orchestration.txt"
    return path.read_text(encoding="utf-8").replace("{parent}", "sota-openai")


def _prompt_only_working_pane(prompt: str) -> str:
    return (
        "Default model: litellm-alpha-passthrough/sota-openai\n"
        f"{prompt}\n"
        "Working…\n"
    )


def _ohmypi_idle_pong_pane_with_launch_splash() -> str:
    """Ohmypi 17.4 pane after exact PONG: launch splash stays in scrollback."""

    return (
        "╭─── omp v17.4.2 ────────────────────────────────────────────────────────────╮\n"
        "│                          │ Tips                                            │\n"
        "│      Welcome back!       │ # for prompt actions                            │\n"
        "│                          │ / for commands                                  │\n"
        "│   AAWM alias sota-zai    │                                                 │\n"
        "│litellm-alpha-passthrough │                                                 │\n"
        "│                          │ ─────────────────────────────────────────────── │\n"
        "│                          │ Recent sessions                                 │\n"
        "│                          │ • Reply with exactly the word PONG. (just now)  │\n"
        "╰──────────────────────────┴─────────────────────────────────────────────────╯\n"
        " Connecting to MCP servers: aawm-transcript…\n"
        "\n"
        " Connected to MCP server: aawm-transcript.\n"
        "\n"
        " Reply with exactly the word PONG.\n"
        "\n"
        " PONG\n"
        "\n"
        " ※ recap: Session held only a connectivity check: you asked for the exact reply\n"
        " PONG and I answered PONG.\n"
        "\n"
        "╭── π  > ⬢ AAWM alias sota-zai > 🗑 omp-alpha-workspace > ◫ 6.4%/128K ⟲ ▶───────╮\n"
        "╰─                                                                            ─╯\n"
        "interactive\n"
    )


def _ohmypi_welcome_back_basic_chrome() -> str:
    """Welcome-back / Recent sessions chrome shared by Ohmypi basic panes."""

    return (
        "╭─── omp v17.4.2 ────────────────────────────────────────────────────────────╮\n"
        "│                          │ Tips                                            │\n"
        "│      Welcome back!       │ # for prompt actions                            │\n"
        "│                          │ / for commands                                  │\n"
        "│       ▀██████████▀       │ ! to run bash                                   │\n"
        "│        ╘██    ██         │ $ to run python                                 │\n"
        "│         ██    ██         │ ─────────────────────────────────────────────── │\n"
        "│         ██    ██         │ LSP Servers                                     │\n"
        "│        ▄██▄  ▄██▄        │ No LSP servers                                  │\n"
        "│                          │                                                 │\n"
        "│     AAWM alias basic     │                                                 │\n"
        "│litellm-alpha-passthrough │                                                 │\n"
        "│                          │ ─────────────────────────────────────────────── │\n"
        "│                          │ Recent sessions                                 │\n"
        "│                          │ • Reply with exactly the word PONG. (40m ago)   │\n"
        "│                          │ • Reply with exactly the word PONG. (46m ago)   │\n"
        "│                          │ • Reply with exactly the word PONG. (52m ago)   │\n"
        "│                          │ • Reply with exactly the word PONG. (57m ago)   │\n"
        "│                          │                                                 │\n"
        "╰──────────────────────────┴─────────────────────────────────────────────────╯\n"
        " Tip: Press shift+tab to cycle through reasoning effort levels\n"
        "\n"
        " Connecting to MCP servers: aawm-transcript…\n"
        "\n"
        "────────────────────────────────────────────────────────────────────────────────\n"
        " Update Available\n"
        " New version 18.0.3 is available. Run: omp update\n"
        "────────────────────────────────────────────────────────────────────────────────\n"
        "\n"
        " Connected to MCP server: aawm-transcript.\n"
    )


def _ohmypi_idle_footer_basic() -> str:
    return (
        "╭── π  > ⬢ AAWM alias basic > 🗑 omp-alpha-workspace > ◫ 6.4%/128K ⟲ ▶──────────╮\n"
        "╰─                                                                            ─╯\n"
        "interactive\n"
    )


def _ohmypi_idle_pong_pane_without_recap() -> str:
    """Live basic/work retry pane: exact PONG + idle footer, no recap."""

    return (
        f"{_ohmypi_welcome_back_basic_chrome()}\n"
        "\n"
        " Reply with exactly the word PONG.\n"
        "\n"
        "\n"
        " The user wants exactly the word PONG.\n"
        "\n"
        " PONG\n"
        "\n"
        f"{_ohmypi_idle_footer_basic()}"
    )


def _ohmypi_pre_send_splash_idle_pane() -> str:
    """Splash + idle footer before this send paints a standalone prompt echo."""

    return f"{_ohmypi_welcome_back_basic_chrome()}\n\n{_ohmypi_idle_footer_basic()}"


def _ohmypi_prompt_echo_idle_without_pong() -> str:
    """This turn's prompt echo is painted, but no standalone PONG follows it yet."""

    return (
        f"{_ohmypi_welcome_back_basic_chrome()}\n"
        "\n"
        " Reply with exactly the word PONG.\n"
        "\n"
        f"{_ohmypi_idle_footer_basic()}"
    )


def _ohmypi_restored_complete_pong_turn_pane() -> str:
    """Retry3 restore: one complete echo+PONG turn, no second prompt echo."""

    return (
        f"{_ohmypi_welcome_back_basic_chrome()}\n"
        "\n"
        " Reply with exactly the word PONG.\n"
        "\n"
        "\n"
        " The user asked me to reply with exactly the word PONG.\n"
        "\n"
        " PONG\n"
        "\n"
        f"{_ohmypi_idle_footer_basic()}"
    )


def _ohmypi_session_dir_leftover_pong_before_latest_prompt() -> str:
    """Restored capture-pane -S -200: leftover PONG sits before this turn's echo."""

    recent = "".join(
        f"│                          │ • Reply with exactly the word PONG. ({n}m ago)   │\n"
        for n in range(40, 72)
    )
    restored_scrollback = "\n".join(
        f" restored session-dir scrollback {index}" for index in range(140)
    )
    return (
        "╭─── omp v17.4.2 ────────────────────────────────────────────────────────────╮\n"
        "│                          │ Tips                                            │\n"
        "│      Welcome back!       │ # for prompt actions                            │\n"
        "│                          │ / for commands                                  │\n"
        "│       ▀██████████▀       │ ! to run bash                                   │\n"
        "│        ╘██    ██         │ $ to run python                                 │\n"
        "│         ██    ██         │ ─────────────────────────────────────────────── │\n"
        "│         ██    ██         │ LSP Servers                                     │\n"
        "│        ▄██▄  ▄██▄        │ No LSP servers                                  │\n"
        "│                          │                                                 │\n"
        "│     AAWM alias basic     │                                                 │\n"
        "│litellm-alpha-passthrough │                                                 │\n"
        "│                          │ ─────────────────────────────────────────────── │\n"
        "│                          │ Recent sessions                                 │\n"
        f"{recent}"
        "│                          │                                                 │\n"
        "╰──────────────────────────┴─────────────────────────────────────────────────╯\n"
        " Tip: Press shift+tab to cycle through reasoning effort levels\n"
        "\n"
        " Connecting to MCP servers: aawm-transcript…\n"
        "\n"
        " Connected to MCP server: aawm-transcript.\n"
        "\n"
        f"{restored_scrollback}\n"
        "\n"
        " Reply with exactly the word PONG.\n"
        "\n"
        " PONG\n"
        "\n"
        " Reply with exactly the word PONG.\n"
        "\n"
        "╭── π  > ⬢ AAWM alias basic > 🗑 omp-alpha-workspace > ◫ 6.4%/128K ⟲ ▶──────────╮\n"
        "╰─                                                                            ─╯\n"
        "interactive\n"
    )


def _ohmypi_session_dir_leftover_pong_plus_live_pong() -> str:
    """Same restore, but a new standalone PONG follows the latest prompt echo."""

    return _ohmypi_session_dir_leftover_pong_before_latest_prompt().replace(
        "\n╭── π  >",
        "\n PONG\n\n╭── π  >",
        1,
    )


_OHMYPI_MODEL_REPLY_NEEDLES = [
    "※ recap:",
    "No endpoints found for",
    "404 Not Found",
    "status code 404",
    "Error: 404",
]


def _ohmypi_busy_working_pane(*, idle_footer: bool) -> str:
    footer = ""
    if idle_footer:
        footer = (
            "╭── π  > ⬢ AAWM alias work > 🗑 omp-alpha-workspace > ◫ 6.4%/128K ⟲ ▶───────╮\n"
            "╰─                                                                            ─╯\n"
            "interactive\n"
        )
    return (
        " Reply with exactly the word PONG.\n"
        " Working…\n"
        " Thinking\n"
        f"{footer}"
    )


def _fast_ohmypi_tmux_float(key: str, default: float) -> float:
    if key in {"wait_idle_seconds", "wait_reply_seconds"}:
        return 0.2
    if key == "poll_interval_seconds":
        return 0.01
    return default


def _pane_has_pass_evidence(pane: str, needles: list[str], *, prompt: str) -> bool:
    from hv2.kinds.runner import _pane_has_any

    try:
        return bool(_pane_has_any(pane, needles, prompt=prompt))
    except TypeError:
        return bool(_pane_has_any(pane, needles))


def test_should_not_count_orchestration_prompt_needle_as_pass_evidence() -> None:
    prompt = _orchestration_sent_prompt()
    pane = _prompt_only_working_pane(prompt)
    needles = ["omp-alpha-fanout"]
    assert "omp-alpha-fanout" in prompt
    assert "omp-alpha-fanout" in pane
    # Needle is only in the pasted prompt, so it is not reply / pass evidence.
    assert _pane_has_pass_evidence(pane, needles, prompt=prompt) is False


def test_should_count_recap_needle_not_in_prompt_as_pass_evidence() -> None:
    prompt = _orchestration_sent_prompt()
    pane = (
        _prompt_only_working_pane(prompt).replace("Working…\n", "")
        + "※ recap:\n"
        + "basic/work/expert/sota returned date stdout and omp-alpha-fanout\n"
    )
    assert "※ recap:" not in prompt
    assert _pane_has_pass_evidence(pane, ["※ recap:"], prompt=prompt) is True


def test_should_accept_standalone_exact_pong_line() -> None:
    from hv2.kinds.runner import _pane_exact_pong

    prompt = "Reply with exactly the word PONG."
    pane = (
        "Default model: litellm-alpha-passthrough/work\n"
        f"{prompt}\n"
        "PONG\n"
        "π  > ⬢ AAWM alias / model work\n"
    )
    assert _pane_exact_pong(pane, prompt) is True


def test_should_reject_prompt_echo_and_non_exact_pong() -> None:
    from hv2.kinds.runner import _pane_exact_pong

    prompt = "Reply with exactly the word PONG."
    echo_only = f"Default model: litellm-alpha-passthrough/work\n{prompt}\n"
    assert _pane_exact_pong(echo_only, prompt) is False
    assert _pane_exact_pong("PONG\n", "PONG") is False
    assert _pane_exact_pong("PONG!\n", prompt) is False
    assert _pane_exact_pong("The word is PONG\n", prompt) is False


def test_should_not_mark_tui_replied_when_needle_is_only_in_sent_prompt(
    hv, config, monkeypatch
) -> None:
    from hv2.drivers.ohmypi import OhmypiDriver

    prompt = _orchestration_sent_prompt()
    pane = _prompt_only_working_pane(prompt)
    driver = OhmypiDriver(config)
    monkeypatch.setattr(
        driver,
        "send_keys",
        lambda text: {"ok": True, "method": "paste-buffer"},
    )
    monkeypatch.setattr(driver, "capture_pane", lambda: pane)

    def fake_tmux_float(key: str, default: float) -> float:
        if key == "wait_reply_seconds":
            return 0.2
        if key == "poll_interval_seconds":
            return 0.01
        return default

    monkeypatch.setattr(driver, "_tmux_float", fake_tmux_float)
    waited = driver.send_prompt_and_wait(
        prompt, reply_needles=["omp-alpha-fanout"]
    )
    assert waited["replied"] is False
    assert waited["ok"] is False


def test_should_mark_tui_replied_when_recap_needle_is_absent_from_prompt(
    hv, config, monkeypatch
) -> None:
    from hv2.drivers.ohmypi import OhmypiDriver

    prompt = _orchestration_sent_prompt()
    pane = (
        _prompt_only_working_pane(prompt).replace("Working…\n", "")
        + "※ recap:\n"
        + "four children returned date stdout\n"
    )
    driver = OhmypiDriver(config)
    monkeypatch.setattr(
        driver,
        "send_keys",
        lambda text: {"ok": True, "method": "paste-buffer"},
    )
    monkeypatch.setattr(driver, "capture_pane", lambda: pane)

    def fake_tmux_float(key: str, default: float) -> float:
        if key == "wait_reply_seconds":
            return 0.2
        if key == "poll_interval_seconds":
            return 0.01
        return default

    monkeypatch.setattr(driver, "_tmux_float", fake_tmux_float)
    waited = driver.send_prompt_and_wait(prompt, reply_needles=["※ recap:"])
    assert waited["replied"] is True
    assert waited["ok"] is True


def test_should_not_mark_tui_ok_when_reply_seen_but_not_idle(
    hv, config, monkeypatch
) -> None:
    from hv2.drivers.ohmypi import OhmypiDriver

    prompt = "Reply with exactly the word PONG."
    pane = (
        "Default model: litellm-alpha-passthrough/work\n"
        f"{prompt}\n"
        "※ recap:\n"
        "PONG\n"
    )
    driver = OhmypiDriver(config)
    monkeypatch.setattr(
        driver,
        "send_keys",
        lambda text: {"ok": True, "method": "paste-buffer"},
    )
    monkeypatch.setattr(driver, "capture_pane", lambda: pane)
    monkeypatch.setattr(
        driver, "wait_until_idle", lambda timeout_seconds=None: False
    )

    def fake_tmux_float(key: str, default: float) -> float:
        if key == "wait_reply_seconds":
            return 0.2
        if key == "poll_interval_seconds":
            return 0.01
        return default

    monkeypatch.setattr(driver, "_tmux_float", fake_tmux_float)
    waited = driver.send_prompt_and_wait(prompt, reply_needles=["※ recap:"])
    assert waited["replied"] is True
    assert waited["idle"] is False
    assert waited["ok"] is False


def test_should_treat_ohmypi_welcome_back_splash_as_idle_after_exact_pong(
    hv, config, monkeypatch
) -> None:
    from hv2.drivers.ohmypi import OhmypiDriver

    pane = _ohmypi_idle_pong_pane_with_launch_splash()
    assert "Welcome back!" in pane
    assert "Recent sessions" in pane
    assert "Connecting to MCP servers:" in pane
    assert "Connected to MCP server: aawm-transcript." in pane
    assert "Reply with exactly the word PONG." in pane
    assert "\n PONG\n" in pane
    assert "※ recap:" in pane
    assert "π  >" in pane
    assert "╰─" in pane
    assert "interactive" in pane

    driver = OhmypiDriver(config)
    monkeypatch.setattr(driver, "capture_pane", lambda: pane)
    monkeypatch.setattr(driver, "_tmux_float", _fast_ohmypi_tmux_float)
    monkeypatch.setattr(
        driver,
        "send_keys",
        lambda text: {"ok": True, "method": "send-keys"},
    )

    assert driver.wait_until_idle() is True

    waited = driver.send_prompt_and_wait(
        "Reply with exactly the word PONG.",
        reply_needles=["※ recap:"],
    )
    assert waited["idle"] is True
    assert waited["ok"] is True


def test_should_not_treat_ohmypi_working_or_thinking_pane_as_idle(
    hv, config, monkeypatch
) -> None:
    from hv2.drivers.ohmypi import OhmypiDriver

    driver = OhmypiDriver(config)
    monkeypatch.setattr(driver, "_tmux_float", _fast_ohmypi_tmux_float)

    busy = _ohmypi_busy_working_pane(idle_footer=False)
    assert "Working…" in busy
    assert "Thinking" in busy
    assert "π  >" not in busy
    assert "╰─" not in busy
    monkeypatch.setattr(driver, "capture_pane", lambda: busy)
    assert driver.wait_until_idle() is False

    busy_with_footer = _ohmypi_busy_working_pane(idle_footer=True)
    assert "Working…" in busy_with_footer
    assert "π  >" in busy_with_footer
    assert "╰─" in busy_with_footer
    monkeypatch.setattr(driver, "capture_pane", lambda: busy_with_footer)
    assert driver.wait_until_idle() is False


def test_should_treat_ohmypi_exact_pong_without_recap_as_idle_model_turn(
    hv, config, monkeypatch
) -> None:
    from hv2.drivers.ohmypi import OhmypiDriver
    from hv2.kinds.runner import _pane_exact_pong

    prompt = "Reply with exactly the word PONG."
    pre_send = _ohmypi_pre_send_splash_idle_pane()
    echo_only = _ohmypi_prompt_echo_idle_without_pong()
    live = _ohmypi_idle_pong_pane_without_recap()
    assert "Welcome back!" in pre_send
    assert "Recent sessions" in pre_send
    assert "Connecting to MCP servers:" in pre_send
    assert "π  >" in pre_send
    assert "╰─" in pre_send
    assert "interactive" in pre_send
    assert _pane_exact_pong(pre_send, prompt) is False
    assert _last_stripped_line_index(echo_only, prompt) >= 0
    assert _pane_exact_pong(echo_only, prompt) is False
    assert "Welcome back!" in live
    assert "Recent sessions" in live
    assert "Connecting to MCP servers:" in live
    assert "Reply with exactly the word PONG." in live
    assert "\n PONG\n" in live
    assert "※ recap:" not in live
    assert "π  >" in live
    assert "╰─" in live
    assert "interactive" in live
    assert _pane_exact_pong(live, prompt) is True

    sent = {"done": False}
    post_send_panes = iter([echo_only, live])

    def fake_send_keys(text: str) -> dict[str, Any]:
        sent["done"] = True
        return {"ok": True, "method": "send-keys"}

    def fake_capture_pane() -> str:
        if not sent["done"]:
            return pre_send
        try:
            return next(post_send_panes)
        except StopIteration:
            return live

    driver = OhmypiDriver(config)
    monkeypatch.setattr(driver, "capture_pane", fake_capture_pane)
    monkeypatch.setattr(driver, "_tmux_float", _fast_ohmypi_tmux_float)
    monkeypatch.setattr(driver, "send_keys", fake_send_keys)

    waited = driver.send_prompt_and_wait(
        prompt, reply_needles=_OHMYPI_MODEL_REPLY_NEEDLES
    )
    assert sent["done"] is True
    assert waited["replied"] is True
    assert waited["idle"] is True
    assert waited["ok"] is True
    assert _pane_exact_pong(str(waited.get("pane") or ""), prompt) is True


def test_should_not_treat_restored_complete_pong_turn_as_this_sends_live_reply(
    hv, config, monkeypatch
) -> None:
    from hv2.drivers.ohmypi import OhmypiDriver
    from hv2.kinds.runner import _pane_exact_pong

    prompt = "Reply with exactly the word PONG."
    pane = _ohmypi_restored_complete_pong_turn_pane()
    echo_indexes = [
        index
        for index, raw in enumerate(pane.splitlines())
        if raw.strip() == prompt
    ]
    leftover_pong = _last_stripped_line_index(pane, "PONG")

    assert "Welcome back!" in pane
    assert "Recent sessions" in pane
    assert "Connecting to MCP servers:" in pane
    assert "The user asked me to reply with exactly the word PONG." in pane
    assert "Reply with exactly the word PONG." in pane
    assert "\n PONG\n" in pane
    assert "※ recap:" not in pane
    assert "π  >" in pane
    assert "╰─" in pane
    assert "interactive" in pane
    assert echo_indexes == [echo_indexes[0]]
    assert leftover_pong > echo_indexes[0]
    assert prompt not in [raw.strip() for raw in pane.splitlines()[leftover_pong + 1 :]]
    # Scanner still matches: the hole is send_prompt_and_wait without a watermark.
    assert _pane_exact_pong(pane, prompt) is True

    driver = OhmypiDriver(config)
    monkeypatch.setattr(driver, "capture_pane", lambda: pane)
    monkeypatch.setattr(driver, "_tmux_float", _fast_ohmypi_tmux_float)
    monkeypatch.setattr(
        driver,
        "send_keys",
        lambda text: {"ok": True, "method": "send-keys"},
    )

    waited = driver.send_prompt_and_wait(
        prompt, reply_needles=_OHMYPI_MODEL_REPLY_NEEDLES
    )
    assert waited["replied"] is False
    assert waited["idle"] is False
    assert waited["ok"] is False


def _last_stripped_line_index(pane: str, expected: str) -> int:
    last = -1
    for index, raw_line in enumerate(pane.splitlines()):
        if raw_line.strip() == expected:
            last = index
    return last


def test_should_not_treat_leftover_session_dir_pong_before_latest_prompt_as_live_reply(
    hv, config, monkeypatch
) -> None:
    from hv2.drivers.ohmypi import OhmypiDriver
    from hv2.kinds.runner import _pane_exact_pong, _pane_has_any

    prompt = "Reply with exactly the word PONG."
    pane = _ohmypi_session_dir_leftover_pong_before_latest_prompt()
    lines = [raw.strip() for raw in pane.splitlines()]
    latest_echo = _last_stripped_line_index(pane, prompt)
    leftover_pong = _last_stripped_line_index(pane, "PONG")

    assert "Welcome back!" in pane
    assert "Recent sessions" in pane
    assert "Reply with exactly the word PONG." in pane
    assert "\n PONG\n" in pane
    assert "※ recap:" not in pane
    assert "π  >" in pane
    assert "╰─" in pane
    assert "interactive" in pane
    assert len(pane.splitlines()) >= 200
    assert leftover_pong >= 0
    assert latest_echo > leftover_pong
    assert "PONG" not in lines[latest_echo + 1 :]
    assert _pane_exact_pong(pane, prompt) is False
    assert _pane_has_any(pane, ["PONG"], prompt=prompt) is False

    driver = OhmypiDriver(config)
    monkeypatch.setattr(driver, "capture_pane", lambda: pane)
    monkeypatch.setattr(driver, "_tmux_float", _fast_ohmypi_tmux_float)
    monkeypatch.setattr(
        driver,
        "send_keys",
        lambda text: {"ok": True, "method": "send-keys"},
    )

    waited = driver.send_prompt_and_wait(
        prompt, reply_needles=_OHMYPI_MODEL_REPLY_NEEDLES
    )
    assert waited["replied"] is False
    assert waited["idle"] is False
    assert waited["ok"] is False


def test_should_treat_leftover_session_dir_pong_plus_new_pong_after_latest_prompt_as_live_reply(
    hv, config, monkeypatch
) -> None:
    from hv2.drivers.ohmypi import OhmypiDriver
    from hv2.kinds.runner import _pane_exact_pong, _pane_has_any

    prompt = "Reply with exactly the word PONG."
    leftover = _ohmypi_session_dir_leftover_pong_before_latest_prompt()
    live = leftover.replace(
        "\n╭── π  >",
        "\n Reply with exactly the word PONG.\n\n PONG\n\n╭── π  >",
        1,
    )
    leftover_echo = _last_stripped_line_index(leftover, prompt)
    live_echo = _last_stripped_line_index(live, prompt)
    live_pong = _last_stripped_line_index(live, "PONG")

    assert leftover_echo >= 0
    assert live_echo > leftover_echo
    assert live_pong > live_echo
    assert _pane_exact_pong(leftover, prompt) is False
    assert _pane_exact_pong(live, prompt) is True
    assert _pane_has_any(live, ["PONG"], prompt=prompt) is True

    sent = {"done": False}
    post_send_panes = iter([live])

    def fake_send_keys(text: str) -> dict[str, Any]:
        sent["done"] = True
        return {"ok": True, "method": "send-keys"}

    def fake_capture_pane() -> str:
        if not sent["done"]:
            return leftover
        try:
            return next(post_send_panes)
        except StopIteration:
            return live

    driver = OhmypiDriver(config)
    monkeypatch.setattr(driver, "capture_pane", fake_capture_pane)
    monkeypatch.setattr(driver, "_tmux_float", _fast_ohmypi_tmux_float)
    monkeypatch.setattr(driver, "send_keys", fake_send_keys)

    waited = driver.send_prompt_and_wait(
        prompt, reply_needles=_OHMYPI_MODEL_REPLY_NEEDLES
    )
    assert sent["done"] is True
    assert waited["replied"] is True
    assert waited["idle"] is True
    assert waited["ok"] is True
    assert _pane_exact_pong(str(waited.get("pane") or ""), prompt) is True


_ORCH_CHILDREN = ("basic", "work", "expert", "sota")


def _task_tool_result_line(agents: Sequence[str], *, status: str = "ok") -> str:
    return json.dumps(
        {
            "message": {
                "role": "toolResult",
                "toolName": "task",
                "details": {
                    "results": [
                        {"agent": name, "status": status} for name in agents
                    ]
                },
            }
        }
    )


def test_should_fail_child_spawn_evidence_on_recap_only_pane() -> None:
    from hv2.checks.orch_evidence import _HUB_IDLE_PEER, _PANE_CHILD_DATE, child_spawn_evidence
    from hv2.kinds.runner import _pane_has_any

    prompt = _orchestration_sent_prompt()
    pane = (
        _prompt_only_working_pane(prompt).replace("Working…\n", "")
        + "※ recap:\n"
        + "basic/work/expert/sota returned date stdout and omp-alpha-fanout\n"
    )
    assert _pane_has_any(pane, ["※ recap:"], prompt=prompt) is True
    evidence = child_spawn_evidence(children=_ORCH_CHILDREN, pane=pane)
    assert evidence["ok"] is False
    joined = " ".join(evidence["failures"])
    assert "recap-only" in joined
    assert "missing successful Ohmypi task results" in joined
    for name in _ORCH_CHILDREN:
        assert name in joined
    assert evidence["successful_agents"] == []
    assert evidence["saw_task_result"] is False
    # Recap remains non-evidence; pane/hub regexes must still see hyphenated
    # baseline children when those names later appear as real spawn rows.
    assert _PANE_CHILD_DATE.search(" - sota-xai\n   - date: Sat 22 Aug 2026\n")
    assert _HUB_IDLE_PEER.search("AutoReviewDate [auto-review · sub · idle]")


def test_should_fail_child_spawn_evidence_on_recap_only_pane_for_baseline_children() -> None:
    from hv2.checks.orch_evidence import _HUB_IDLE_PEER, _PANE_CHILD_DATE, child_spawn_evidence
    from hv2.kinds.runner import _pane_has_any

    prompt = _orchestration_sent_prompt()
    pane = (
        _prompt_only_working_pane(prompt).replace("Working…\n", "")
        + "※ recap:\n"
        + "/".join(_ORCH_BASELINE_CHILDREN)
        + " returned PONG, date stdout and omp-alpha-fanout\n"
    )
    assert _pane_has_any(pane, ["※ recap:"], prompt=prompt) is True
    evidence = child_spawn_evidence(children=_ORCH_BASELINE_CHILDREN, pane=pane)
    assert evidence["ok"] is False
    joined = " ".join(evidence["failures"])
    assert "recap-only" in joined
    assert "missing successful Ohmypi task results" in joined
    for name in _ORCH_BASELINE_CHILDREN:
        assert name in joined
    assert evidence["successful_agents"] == []
    assert evidence["saw_task_result"] is False
    assert evidence["children"] == list(_ORCH_BASELINE_CHILDREN)
    assert _PANE_CHILD_DATE.search(" - auto-review\n   - date: Sat 22 Aug 2026\n")
    assert _HUB_IDLE_PEER.search("SotaXaiDate [sota-xai · sub · idle]")


def test_should_pass_child_spawn_evidence_on_hyphenated_pane_date_and_hub_idle_peers() -> None:
    from hv2.checks.orch_evidence import child_spawn_evidence

    date_lines = "".join(
        f" - {name}\n   - date: Sat 22 Aug 2026 01:31:{index:02d} AM EDT\n"
        for index, name in enumerate(_ORCH_BASELINE_CHILDREN)
    )
    hub_lines = "".join(
        f"{name.title().replace('-', '')}Date [{name} · sub · idle] — parent Main\n"
        for name in _ORCH_BASELINE_CHILDREN
    )
    pane = (
        "Default model: litellm-alpha-passthrough/sota-openai\n"
        f"{date_lines}"
        f"{hub_lines}"
        "π  > ⬢ AAWM alias sota-openai\n"
    )
    evidence = child_spawn_evidence(children=_ORCH_BASELINE_CHILDREN, pane=pane)
    assert "sota-xai" in evidence["successful_agents"]
    assert "auto-review" in evidence["successful_agents"]
    assert evidence["successful_agents"] == sorted(_ORCH_BASELINE_CHILDREN)
    assert evidence["ok"] is True
    assert evidence["failures"] == []


def test_should_fail_child_spawn_evidence_on_unknown_agent() -> None:
    from hv2.checks.orch_evidence import child_spawn_evidence

    pane = (
        'Unknown agent "basic"\n'
        "failed preflight\n"
        "unavailable. The spawn tool reported available agents: scout\n"
    )
    evidence = child_spawn_evidence(children=_ORCH_CHILDREN, pane=pane)
    assert evidence["ok"] is False
    joined = " ".join(evidence["failures"])
    assert "Unknown agent" in pane
    assert "preflight" in joined or "unknown=" in joined
    assert "basic" in evidence["unknown_agents"] or "basic" in joined


def test_should_pass_child_spawn_evidence_on_ohmypi_task_tool_results(
    tmp_path: Path,
) -> None:
    from hv2.checks.orch_evidence import child_spawn_evidence

    session_dir = tmp_path / "omp-sessions"
    session_dir.mkdir()
    (session_dir / "parent.jsonl").write_text(
        _task_tool_result_line(_ORCH_CHILDREN) + "\n",
        encoding="utf-8",
    )
    pane = "※ recap:\nfour children returned date stdout\n"
    evidence = child_spawn_evidence(
        children=_ORCH_CHILDREN,
        pane=pane,
        session_dir=str(session_dir),
    )
    assert evidence["ok"] is True
    assert evidence["failures"] == []
    assert evidence["saw_task_result"] is True
    assert evidence["successful_agents"] == sorted(_ORCH_CHILDREN)


def test_should_pass_child_spawn_evidence_on_ohmypi_hub_job_completions(
    tmp_path: Path,
) -> None:
    from hv2.checks.orch_evidence import child_spawn_evidence

    session_dir = tmp_path / "omp-sessions"
    session_dir.mkdir()
    jobs = [
        {
            "id": f"{name.title()}Date",
            "type": "task",
            "status": "completed",
            "resolvedModel": f"litellm-alpha-passthrough/{name}",
            "resultText": (
                f'<task-result id="{name.title()}Date" agent="{name}" '
                'status="completed" duration="11.3s"></task-result>'
            ),
        }
        for name in _ORCH_CHILDREN
    ]
    (session_dir / "parent.jsonl").write_text(
        json.dumps(
            {
                "message": {
                    "role": "toolResult",
                    "toolName": "hub",
                    "content": [{"type": "text", "text": "Spawned 4 background agents"}],
                    "details": {"op": "wait", "jobs": jobs},
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    pane = (
        " - basic\n   - date: Sat 22 Aug 2026 01:31:11 AM EDT\n"
        " - work\n   - date: Sat 22 Aug 2026 01:31:08 AM EDT\n"
        " - expert\n   - date: Sat 22 Aug 2026 01:31:07 AM EDT\n"
        " - sota\n   - date: Sat 22 Aug 2026 01:31:09 AM EDT\n"
        "π  > ⬢ AAWM alias sota-openai\n"
    )
    evidence = child_spawn_evidence(
        children=_ORCH_CHILDREN,
        pane=pane,
        session_dir=str(session_dir),
    )
    assert evidence["ok"] is True
    assert evidence["failures"] == []
    assert evidence["saw_task_result"] is True
    assert evidence["successful_agents"] == sorted(_ORCH_CHILDREN)


def test_should_count_idle_hub_peer_and_nested_child_session_as_spawn(
    tmp_path: Path,
) -> None:
    from hv2.checks.orch_evidence import child_spawn_evidence

    session_dir = tmp_path / "omp-sessions"
    nested = session_dir / "parent-id"
    nested.mkdir(parents=True)
    (session_dir / "parent.jsonl").write_text(
        json.dumps(
            {
                "message": {
                    "role": "toolResult",
                    "toolName": "hub",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "WorkDate [work · sub · idle] — parent Main\n"
                                "BasicDate [basic · sub · idle] — parent Main\n"
                                "SotaDate [sota · sub · idle] — parent Main\n"
                                "ExpertDate [expert · sub · idle] — parent Main\n"
                            ),
                        }
                    ],
                    "details": {"op": "list"},
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (nested / "SotaDate.jsonl").write_text(
        json.dumps(
            {
                "type": "model_change",
                "model": "litellm-alpha-passthrough/sota",
            }
        )
        + "\n"
        + json.dumps(
            {
                "type": "session_init",
                "agent": "sota",
                "resolvedModel": "litellm-alpha-passthrough/sota",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    evidence = child_spawn_evidence(
        children=_ORCH_CHILDREN,
        pane="π  > ⬢ AAWM alias sota-openai\n",
        session_dir=str(session_dir),
    )
    assert evidence["ok"] is True
    assert evidence["successful_agents"] == sorted(_ORCH_CHILDREN)


def test_should_not_count_spawned_using_announcement_as_child_completion(
    tmp_path: Path,
) -> None:
    from hv2.checks.orch_evidence import child_spawn_evidence

    session_dir = tmp_path / "omp-sessions"
    session_dir.mkdir()
    (session_dir / "parent.jsonl").write_text(
        json.dumps(
            {
                "message": {
                    "role": "toolResult",
                    "toolName": "task",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Spawned 4 background agents using "
                                "basic, work, expert, sota. Each result "
                                "auto-delivers on yield."
                            ),
                        }
                    ],
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    evidence = child_spawn_evidence(
        children=_ORCH_CHILDREN,
        pane="π  > ⬢ AAWM alias sota-openai\n",
        session_dir=str(session_dir),
    )
    assert evidence["ok"] is False
    assert evidence["successful_agents"] == []
    joined = " ".join(evidence["failures"])
    assert "recap-only" in joined
    for name in _ORCH_CHILDREN:
        assert name in joined


def test_should_count_async_result_task_result_as_spawn(tmp_path: Path) -> None:
    from hv2.checks.orch_evidence import child_spawn_evidence

    session_dir = tmp_path / "omp-sessions"
    session_dir.mkdir()
    jobs = [
        {
            "id": f"{name.title()}Date",
            "type": "task",
            "status": "completed",
            "resolvedModel": f"litellm-alpha-passthrough/{name}",
            "resultText": (
                f'<task-result id="{name.title()}Date" agent="{name}" '
                'status="completed"></task-result>'
            ),
        }
        for name in ("basic", "work", "expert")
    ]
    (session_dir / "parent.jsonl").write_text(
        json.dumps(
            {
                "message": {
                    "role": "toolResult",
                    "toolName": "hub",
                    "details": {"op": "wait", "jobs": jobs},
                }
            }
        )
        + "\n"
        + json.dumps(
            {
                "type": "custom_message",
                "customType": "async-result",
                "content": (
                    '<system-notice>\nBackground job SotaDate has completed.\n'
                    '<task-result id="SotaDate" agent="sota" '
                    'status="completed" duration="30.6s"></task-result>\n'
                    "</system-notice>"
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    evidence = child_spawn_evidence(
        children=_ORCH_CHILDREN,
        pane="π  > ⬢ AAWM alias sota-openai\n",
        session_dir=str(session_dir),
    )
    assert evidence["ok"] is True
    assert evidence["successful_agents"] == sorted(_ORCH_CHILDREN)
    assert evidence["saw_task_result"] is True


def test_should_not_count_nested_bash_without_yield_as_child_completion(
    tmp_path: Path,
) -> None:
    from hv2.checks.orch_evidence import child_spawn_evidence

    session_dir = tmp_path / "omp-sessions"
    nested = session_dir / "parent-id"
    nested.mkdir(parents=True)
    (session_dir / "parent.jsonl").write_text(
        json.dumps(
            {
                "message": {
                    "role": "toolResult",
                    "toolName": "task",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Spawned 4 background agents using "
                                "basic, work, expert, sota."
                            ),
                        }
                    ],
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    for name in _ORCH_CHILDREN:
        (nested / f"{name.title()}Date.jsonl").write_text(
            json.dumps(
                {
                    "type": "session_init",
                    "agent": name,
                    "resolvedModel": f"litellm-alpha-passthrough/{name}",
                }
            )
            + "\n"
            + json.dumps(
                {
                    "message": {
                        "role": "toolResult",
                        "toolName": "bash",
                        "content": [
                            {
                                "type": "text",
                                "text": "Sat 22 Aug 2026 02:18:16 AM EDT\n",
                            }
                        ],
                        "isError": False,
                    }
                }
            )
            + "\n",
            encoding="utf-8",
        )
    evidence = child_spawn_evidence(
        children=_ORCH_CHILDREN,
        pane="waiting on 4 jobs\nWaiting For Date Results\n",
        session_dir=str(session_dir),
    )
    assert evidence["ok"] is False
    assert evidence["successful_agents"] == []
    joined = " ".join(evidence["failures"])
    assert "recap-only" in joined


def test_should_count_nested_yield_as_child_completion(tmp_path: Path) -> None:
    from hv2.checks.orch_evidence import child_spawn_evidence

    session_dir = tmp_path / "omp-sessions"
    nested = session_dir / "parent-id"
    nested.mkdir(parents=True)
    (session_dir / "parent.jsonl").write_text("{}\n", encoding="utf-8")
    for name in _ORCH_CHILDREN:
        (nested / f"{name.title()}Date.jsonl").write_text(
            json.dumps(
                {
                    "type": "session_init",
                    "agent": name,
                    "resolvedModel": f"litellm-alpha-passthrough/{name}",
                }
            )
            + "\n"
            + json.dumps(
                {
                    "message": {
                        "role": "toolResult",
                        "toolName": "yield",
                        "content": [{"type": "text", "text": "Result submitted."}],
                        "details": {
                            "data": {"stdout": "Sat 22 Aug 2026 02:18:16 AM EDT\n"},
                            "status": "success",
                            "type": "result",
                        },
                        "isError": False,
                    }
                }
            )
            + "\n",
            encoding="utf-8",
        )
    evidence = child_spawn_evidence(
        children=_ORCH_CHILDREN,
        pane="π  > ⬢ AAWM alias sota-openai\n",
        session_dir=str(session_dir),
    )
    assert evidence["ok"] is True
    assert evidence["successful_agents"] == sorted(_ORCH_CHILDREN)


def test_should_ignore_stale_unknown_agent_when_current_hub_jobs_succeed(
    tmp_path: Path,
) -> None:
    from hv2.checks.orch_evidence import child_spawn_evidence

    session_dir = tmp_path / "omp-sessions"
    session_dir.mkdir()
    stale = session_dir / "stale.jsonl"
    stale.write_text(
        json.dumps(
            {
                "message": {
                    "role": "toolResult",
                    "toolName": "task",
                    "content": [
                        {
                            "type": "text",
                            "text": 'Unknown agent "sota"\nfailed preflight\n',
                        }
                    ],
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    jobs = [
        {
            "id": f"{name.title()}Date",
            "type": "task",
            "status": "completed",
            "resolvedModel": f"litellm-alpha-passthrough/{name}",
            "resultText": (
                f'<task-result id="{name.title()}Date" agent="{name}" '
                'status="completed"></task-result>'
            ),
        }
        for name in _ORCH_CHILDREN
    ]
    current = session_dir / "current.jsonl"
    current.write_text(
        json.dumps(
            {
                "message": {
                    "role": "toolResult",
                    "toolName": "hub",
                    "details": {"op": "wait", "jobs": jobs},
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    stale_mtime = current.stat().st_mtime - 60
    import os

    os.utime(stale, (stale_mtime, stale_mtime))
    evidence = child_spawn_evidence(
        children=_ORCH_CHILDREN,
        pane="π  > ⬢ AAWM alias sota-openai\n",
        session_dir=str(session_dir),
        since_mtime=current.stat().st_mtime,
    )
    assert evidence["ok"] is True
    assert evidence["successful_agents"] == sorted(_ORCH_CHILDREN)
    assert "sota" not in evidence["unknown_agents"]


def test_should_use_agent_equals_not_model_equals_in_orchestration_prompt() -> None:
    prompt = _orchestration_sent_prompt()
    for name in _ORCH_BASELINE_CHILDREN:
        assert f"agent={name}" in prompt
        assert f"model={name}" not in prompt
    assert "agent=codex-auto-review" not in prompt
    assert "model=basic" not in prompt
    assert "model=work" not in prompt
    assert "Spawn exactly four" not in prompt
    lowered = prompt.lower()
    assert "pong" in lowered
    first_pong = lowered.index("pong")
    first_date = lowered.index("date")
    assert first_pong < first_date
    assert "pwd" in prompt
    assert "uname -s" in prompt
    assert "echo omp-alpha-fanout" in prompt


def test_should_stage_orchestration_child_agent_profiles(
    hv, config, tmp_path: Path
) -> None:
    from hv2.drivers.ohmypi import OhmypiDriver

    cfg = _clone_config(config)
    cwd = tmp_path / "omp-alpha-workspace"
    dest = cwd / ".omp" / "agents"
    cfg["tuis"]["ohmypi"]["cwd"] = str(cwd)
    cfg["tuis"]["ohmypi"]["project_agents_dir"] = str(dest)
    cfg["tuis"]["ohmypi"]["orchestration_child_agents"] = list(_ORCH_BASELINE_CHILDREN)
    driver = OhmypiDriver(cfg)
    staged = driver.stage_orchestration_agents()
    assert staged["ok"] is True
    assert staged["missing"] == []
    written_names = {Path(path).name for path in staged["written"]}
    assert written_names == {f"{name}.md" for name in _ORCH_BASELINE_CHILDREN}
    for name in _ORCH_BASELINE_CHILDREN:
        target = dest / f"{name}.md"
        assert target.is_file()
        source = Path(staged["src"]) / f"{name}.md"
        assert source.is_file()
        assert target.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")

    cfg_fallback = _clone_config(config)
    cfg_fallback["tuis"]["ohmypi"]["cwd"] = str(cwd)
    cfg_fallback["tuis"]["ohmypi"]["project_agents_dir"] = str(dest)
    cfg_fallback["tuis"]["ohmypi"].pop("orchestration_child_agents", None)
    fallback = OhmypiDriver(cfg_fallback).stage_orchestration_agents()
    assert {Path(path).name for path in fallback["written"]} == {
        f"{name}.md" for name in _ORCH_BASELINE_CHILDREN
    }


def test_should_expand_cwd_placeholder_when_staging_ohmypi_agents(
    hv, config, tmp_path: Path
) -> None:
    from hv2.drivers.ohmypi import OhmypiDriver

    cfg = _clone_config(config)
    cwd = tmp_path / "omp-alpha-workspace"
    cfg["tuis"]["ohmypi"]["cwd"] = str(cwd)
    cfg["tuis"]["ohmypi"]["project_agents_dir"] = "{cwd}/.omp/agents"
    driver = OhmypiDriver(cfg)
    staged = driver.stage_orchestration_agents()
    dest = cwd / ".omp" / "agents"
    assert staged["ok"] is True
    assert "{" not in str(staged["dest"])
    assert Path(staged["dest"]) == dest
    assert (dest / "sota.md").is_file()
    assert not (Path("{cwd}") / ".omp" / "agents" / "sota.md").is_file()


def test_should_require_passthrough_selector_on_ohmypi_catalog_find_for_work(
    hv, config, monkeypatch
) -> None:
    from hv2.drivers.ohmypi import OhmypiDriver
    from hv2.kinds.runner import _step_tui_catalog

    plan = hv.build_plan(
        config=config,
        kind="catalog",
        instance_token="alpha",
        tui="ohmypi",
        models=["work"],
        orchestration_parent=None,
        orchestration_children=None,
        dry_run=True,
        write_artifact=None,
    )
    assert plan.models == ("work",)
    driver = OhmypiDriver(config)
    assert driver.model_selector("work") == "litellm-alpha-passthrough/work"

    monkeypatch.setattr(
        "hv2.kinds.runner.driver_for",
        lambda tui, _config: driver,
    )
    monkeypatch.setattr(
        driver,
        "catalog_json",
        lambda: {
            "ok": True,
            "stdout": json.dumps(
                {"data": [{"id": "litellm-alpha-passthrough/work"}]}
            ),
            "stderr": "",
            "returncode": 0,
        },
    )

    monkeypatch.setattr(
        driver,
        "catalog_find",
        lambda pattern: {
            "ok": True,
            "pattern": pattern,
            "stdout": json.dumps({"id": "other-provider/work"}),
            "stderr": "",
            "returncode": 0,
        },
    )
    missing = _step_tui_catalog(plan)
    assert missing["ok"] is False
    joined = " ".join(missing["failures"])
    assert "work" in joined
    assert "litellm-alpha-passthrough/work" in joined

    monkeypatch.setattr(
        driver,
        "catalog_find",
        lambda pattern: {
            "ok": True,
            "pattern": pattern,
            "stdout": json.dumps({"id": "litellm-alpha-passthrough/work"}),
            "stderr": "",
            "returncode": 0,
        },
    )
    found = _step_tui_catalog(plan)
    assert found["ok"] is True
    assert found["failures"] == []
    assert found["finds"][0]["selector"] == "litellm-alpha-passthrough/work"
    assert found["finds"][0]["found"] is True


def test_should_leave_ohmypi_model_tmux_session_open_after_tui_model_step(
    hv, config, monkeypatch
) -> None:
    from hv2.drivers.ohmypi import OhmypiDriver
    from hv2.kinds.runner import _step_tui_model

    plan = hv.build_plan(
        config=config,
        kind="model",
        instance_token="alpha",
        tui="ohmypi",
        models=["work"],
        orchestration_parent=None,
        orchestration_children=None,
        dry_run=True,
        write_artifact=None,
    )
    driver = OhmypiDriver(config)
    closes: list[str] = []
    pane = _ohmypi_idle_pong_pane_without_recap()
    monkeypatch.setattr("hv2.kinds.runner.driver_for", lambda tui, _config: driver)
    monkeypatch.setattr(
        driver,
        "launch_argv",
        lambda model, **k: ["omp", "--model", f"litellm-alpha-passthrough/{model}"],
    )
    monkeypatch.setattr(driver, "assert_no_print_flags", lambda argv: None)
    monkeypatch.setattr(
        driver,
        "ensure_session",
        lambda model, tools=False: {
            "ok": True,
            "session": "hv2-ohmypi-work",
            "selector": driver.model_selector(model),
            "selected": True,
        },
    )
    monkeypatch.setattr(
        driver,
        "send_prompt_and_wait",
        lambda prompt, reply_needles=None: {
            "ok": True,
            "send": {"ok": True, "method": "send-keys"},
            "idle": True,
            "replied": True,
            "pane": pane,
        },
    )
    monkeypatch.setattr(driver, "pane_has_selector", lambda model, pane=None: True)
    monkeypatch.setattr(driver, "close_session", lambda: closes.append("close_session"))
    payload = _step_tui_model(plan)
    assert closes == []
    assert payload.get("ok") is True


def test_should_leave_ohmypi_orchestration_tmux_session_open_after_tui_orchestration_step(
    hv, config, tmp_path: Path, monkeypatch
) -> None:
    from hv2.drivers.ohmypi import OhmypiDriver
    from hv2.kinds.runner import _step_tui_orchestration

    session_dir = tmp_path / "omp-sessions"
    session_dir.mkdir()
    (session_dir / "parent.jsonl").write_text(
        _task_tool_result_line(_ORCH_BASELINE_CHILDREN) + "\n",
        encoding="utf-8",
    )
    cfg = _clone_config(config)
    cfg["tuis"]["ohmypi"]["session_dir"] = str(session_dir)
    plan = hv.build_plan(
        config=cfg,
        kind="orchestration",
        instance_token="alpha",
        tui="ohmypi",
        models=None,
        orchestration_parent="sota-openai",
        orchestration_children=None,
        dry_run=True,
        write_artifact=None,
    )
    driver = OhmypiDriver(cfg)
    closes: list[str] = []
    pane = (
        "Default model: litellm-alpha-passthrough/sota-openai\n"
        "π  > ⬢ AAWM alias sota-openai\n"
    )
    monkeypatch.setattr("hv2.kinds.runner.driver_for", lambda tui, _config: driver)
    monkeypatch.setattr(
        driver,
        "launch_argv",
        lambda model, **k: ["omp", "--model", f"litellm-alpha-passthrough/{model}"],
    )
    monkeypatch.setattr(driver, "assert_no_print_flags", lambda argv: None)
    monkeypatch.setattr(
        driver,
        "ensure_session",
        lambda model, tools=True: {
            "ok": True,
            "session": "hv2-ohmypi-sota-openai",
            "selector": driver.model_selector(model),
            "selected": True,
            "staged_agents": {"ok": True, "missing": []},
        },
    )
    monkeypatch.setattr(
        driver, "send_keys", lambda text: {"ok": True, "method": "paste-buffer"}
    )
    monkeypatch.setattr(driver, "capture_pane", lambda: pane)
    monkeypatch.setattr(driver, "pane_has_selector", lambda model, pane=None: True)
    monkeypatch.setattr(driver, "tmux_has_session", lambda name=None: False)
    monkeypatch.setattr(driver, "_tmux_float", _fast_ohmypi_tmux_float)
    monkeypatch.setattr(driver, "close_session", lambda: closes.append("close_session"))
    payload = _step_tui_orchestration(plan)
    assert closes == []
    assert payload.get("ok") is True


def test_should_treat_truncated_ohmypi_alias_chrome_as_selected(hv, config) -> None:
    from hv2.drivers.ohmypi import OhmypiDriver

    driver = OhmypiDriver(config)
    pane = "π  > ⬢ AAWM alias / model codex-auto-review\n"
    assert "π" in pane
    assert "AAWM alias / model codex-auto-review" in pane
    assert "litellm-alpha-passthrough/codex-auto-review" not in pane
    assert "AAWM alias codex-auto-review" not in pane
    assert driver.pane_has_selector("codex-auto-review", pane) is True


def test_should_not_treat_work_other_alias_chrome_as_selected_work(
    hv, config
) -> None:
    from hv2.drivers.ohmypi import OhmypiDriver

    driver = OhmypiDriver(config)
    pane = "π  > ⬢ AAWM alias / model work-other\n"
    assert driver.pane_has_selector("work", pane) is False


def test_should_wait_again_for_ohmypi_selected_needles_after_mcp(
    hv, config, monkeypatch
) -> None:
    from hv2.drivers.ohmypi import OhmypiDriver

    driver = OhmypiDriver(config)
    monkeypatch.setattr(
        driver,
        "_run_tmux",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    monkeypatch.setattr(driver, "tmux_has_session", lambda name=None: False)
    monkeypatch.setattr(driver, "ensure_workspace", lambda: None)
    monkeypatch.setattr(
        driver,
        "capture_pane",
        lambda: "π  > ⬢ AAWM alias / model work\nConnected to MCP\n",
    )
    selected_waits = {"n": 0}

    def fake_wait(
        needle: str | Sequence[str],
        timeout_seconds: float | None = None,
        *,
        prompt: str | None = None,
    ) -> bool:
        needles = [needle] if isinstance(needle, str) else [str(item) for item in needle]
        if any("Connected to MCP" in item or "No MCP" in item for item in needles):
            return True
        if any(
            "AAWM alias" in item
            or "Default model:" in item
            or "litellm-alpha-passthrough/" in item
            for item in needles
        ):
            selected_waits["n"] += 1
            return selected_waits["n"] >= 2
        return True

    monkeypatch.setattr(driver, "wait_for_pane", fake_wait)
    launched = driver.ensure_session("work", tools=False)
    assert launched["ok"] is True
    assert launched["selected"] is True
