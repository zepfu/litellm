"""CFG-038: Cursor, then native OIDC, then managed OAuth for Grok 4.6."""

from __future__ import annotations

from pathlib import Path

from litellm.llms.xai.route_descriptors import (
    GROK_NATIVE_OAUTH_CREDENTIAL_FAMILY,
    XAI_OAUTH_CREDENTIAL_FAMILY,
    get_grok_native_route_descriptor,
    get_oa_xai_route_descriptor,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.adapter_config import (
    CODEX_CURSOR_AGENT_AISERVER,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_compiler import (
    compile_yaml,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_schema import (
    resolve_anthropic_route_family,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_snapshot import (
    RoutingCandidate,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_startup import (
    DEFAULT_CONFIG_DIR,
    compile_directory,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.error_signals import (
    _get_codex_auto_agent_grok_account_quota_lane_cooldown_key,
    _is_codex_auto_agent_cursor_agent_candidate,
    _is_codex_auto_agent_grok_account_quota_candidate,
    _is_codex_auto_agent_xai_candidate,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.lane_keys import (
    _codex_auto_agent_candidate_key,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.policy import (
    CODEX_AUTO_AGENT_CURSOR_AGENT_LANE_KEY,
    CODEX_AUTO_AGENT_XAI_LANE_KEY,
    CODEX_AUTO_AGENT_XAI_OAUTH_LANE_KEY,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import snapshot_select


_REPO_ROOT = Path(__file__).resolve().parents[4]
_SOTA_XAI_YAML = (
    _REPO_ROOT / "litellm" / "proxy" / "aawm_alias_config" / "sota-xai.yaml"
)
_PROVIDER_XAI_YAML = (
    _REPO_ROOT / "litellm" / "proxy" / "aawm_alias_config" / "provider-xai.yaml"
)

_CURSOR_MODEL = "cursor_agent/cursor-grok-4.6-high"
_OIDC_MODEL = "xai/grok-4.6"
_OAUTH_MODEL = "oa_xai/grok-4.6"
_CURSOR_ROUTE = "codex_cursor_agent_aiserver_adapter"
_OIDC_ROUTE = "codex_grok_native_responses_adapter"
_OAUTH_ROUTE = "codex_xai_oauth_responses_adapter"
_CURSOR_ANTHROPIC_ROUTE = "anthropic_cursor_agent_aiserver_adapter"
_OIDC_ANTHROPIC_ROUTE = "anthropic_grok_native_responses_adapter"
_OAUTH_ANTHROPIC_ROUTE = "anthropic_xai_oauth_responses_adapter"

_SOTA_XAI_ORDER = (
    ("cursor_agent", _CURSOR_MODEL, _CURSOR_ROUTE, 110),
    ("xai", _OIDC_MODEL, _OIDC_ROUTE, 100),
    ("xai", _OAUTH_MODEL, _OAUTH_ROUTE, 90),
)
_PROVIDER_XAI_ORDER = (
    ("xai", _OIDC_MODEL, _OIDC_ROUTE, 100),
    ("xai", _OAUTH_MODEL, _OAUTH_ROUTE, 90),
)


def _identity(candidate: RoutingCandidate) -> tuple[str, str, str | None, int]:
    return (
        candidate.provider,
        candidate.model,
        candidate.route_family,
        candidate.priority,
    )


def _lane_key(provider: str, route_family: str) -> str:
    if provider == "cursor_agent":
        return CODEX_AUTO_AGENT_CURSOR_AGENT_LANE_KEY
    if route_family in {_OAUTH_ROUTE, _OAUTH_ANTHROPIC_ROUTE}:
        return CODEX_AUTO_AGENT_XAI_OAUTH_LANE_KEY
    return CODEX_AUTO_AGENT_XAI_LANE_KEY


def _credential_family(provider: str, model: str) -> str:
    if provider == "cursor_agent":
        return CODEX_CURSOR_AGENT_AISERVER.credential_family
    oauth = get_oa_xai_route_descriptor(model)
    if oauth is not None:
        return oauth.credential_family
    native = get_grok_native_route_descriptor(model)
    assert native is not None
    return native.credential_family


def _public_candidate(
    candidate: RoutingCandidate,
    *,
    route_family: str,
) -> dict[str, str]:
    return {
        "provider": candidate.provider,
        "model": candidate.model,
        "route_family": route_family,
    }


def _attribution(
    candidate: RoutingCandidate,
    *,
    owning_alias: str,
    route_family: str,
) -> dict[str, str | None]:
    public = _public_candidate(candidate, route_family=route_family)
    lane_key = _lane_key(candidate.provider, route_family)
    cooldown_identity_tag = snapshot_select._snapshot_cooldown_identity_tag(
        owning_alias=owning_alias,
        candidate=public,
    )
    return {
        "provider": candidate.provider,
        "model": candidate.model,
        "route_family": route_family,
        "lane_key": lane_key,
        "credential_family": _credential_family(candidate.provider, candidate.model),
        "cooldown_identity_tag": cooldown_identity_tag,
        "cooldown_key": _codex_auto_agent_candidate_key(
            public,
            lane_key,
            cooldown_identity_tag=cooldown_identity_tag,
        ),
        "quota_cooldown_key": _get_codex_auto_agent_grok_account_quota_lane_cooldown_key(
            public,
            lane_key,
        ),
    }


def test_sota_xai_yaml_compiles_cursor_then_oidc_then_oauth() -> None:
    snapshot = compile_yaml(_SOTA_XAI_YAML.read_text(encoding="utf-8"))
    alias = snapshot.aliases["sota-xai"]
    assert alias.dispatch is None
    assert len(alias.candidates) == 3
    assert all(isinstance(entry, RoutingCandidate) for entry in alias.candidates)
    assert [_identity(entry) for entry in alias.candidates] == list(_SOTA_XAI_ORDER)

    cursor, oidc, oauth = alias.candidates
    assert cursor.anthropic_route_family == _CURSOR_ANTHROPIC_ROUTE
    assert oidc.anthropic_route_family == _OIDC_ANTHROPIC_ROUTE
    assert oauth.anthropic_route_family == _OAUTH_ANTHROPIC_ROUTE
    assert resolve_anthropic_route_family(cursor.route_family, None) == (
        _CURSOR_ANTHROPIC_ROUTE
    )
    assert resolve_anthropic_route_family(oidc.route_family, None) == (
        _OIDC_ANTHROPIC_ROUTE
    )
    assert resolve_anthropic_route_family(oauth.route_family, None) == (
        _OAUTH_ANTHROPIC_ROUTE
    )


def test_directory_sota_xai_matches_file_compile_order() -> None:
    file_snapshot = compile_yaml(_SOTA_XAI_YAML.read_text(encoding="utf-8"))
    directory = compile_directory(DEFAULT_CONFIG_DIR)
    assert [
        _identity(entry) for entry in directory.aliases["sota-xai"].candidates
    ] == [_identity(entry) for entry in file_snapshot.aliases["sota-xai"].candidates]
    assert [
        _identity(entry) for entry in directory.aliases["sota-xai"].candidates
    ] == list(_SOTA_XAI_ORDER)


def test_provider_xai_stays_xai_only_with_oidc_before_oauth() -> None:
    snapshot = compile_directory(DEFAULT_CONFIG_DIR)
    alias = snapshot.aliases["provider-xai"]
    assert alias.dispatch is None
    assert _PROVIDER_XAI_YAML.read_text(encoding="utf-8")
    assert len(alias.candidates) == 2
    assert all(isinstance(entry, RoutingCandidate) for entry in alias.candidates)
    assert [_identity(entry) for entry in alias.candidates] == list(_PROVIDER_XAI_ORDER)
    assert all(entry.provider == "xai" for entry in alias.candidates)
    assert all(entry.model != _CURSOR_MODEL for entry in alias.candidates)
    oidc, oauth = alias.candidates
    assert oidc.anthropic_route_family == _OIDC_ANTHROPIC_ROUTE
    assert oauth.anthropic_route_family == _OAUTH_ANTHROPIC_ROUTE


def test_sota_xai_compiled_candidates_keep_distinct_route_lane_credential_failure_attribution() -> None:
    snapshot = compile_directory(DEFAULT_CONFIG_DIR)
    previous = snapshot_select.get_active_routing_snapshot()
    snapshot_select.set_active_routing_snapshot(snapshot)
    try:
        compiled = snapshot.aliases["sota-xai"].candidates
        assert [_identity(entry) for entry in compiled] == list(_SOTA_XAI_ORDER)
        cursor, oidc, oauth = compiled

        cursor_attr = _attribution(
            cursor, owning_alias="sota-xai", route_family=_CURSOR_ROUTE
        )
        oidc_attr = _attribution(
            oidc, owning_alias="sota-xai", route_family=_OIDC_ROUTE
        )
        oauth_attr = _attribution(
            oauth, owning_alias="sota-xai", route_family=_OAUTH_ROUTE
        )

        assert cursor_attr["lane_key"] == CODEX_AUTO_AGENT_CURSOR_AGENT_LANE_KEY
        assert oidc_attr["lane_key"] == CODEX_AUTO_AGENT_XAI_LANE_KEY
        assert oauth_attr["lane_key"] == CODEX_AUTO_AGENT_XAI_OAUTH_LANE_KEY
        assert len({cursor_attr["lane_key"], oidc_attr["lane_key"], oauth_attr["lane_key"]}) == 3

        assert cursor_attr["credential_family"] == "cursor_agent"
        assert oidc_attr["credential_family"] == GROK_NATIVE_OAUTH_CREDENTIAL_FAMILY
        assert oauth_attr["credential_family"] == XAI_OAUTH_CREDENTIAL_FAMILY
        assert oidc_attr["credential_family"] == "xai_grok_oidc"
        assert oauth_attr["credential_family"] == "xai_oauth"
        assert len(
            {
                cursor_attr["credential_family"],
                oidc_attr["credential_family"],
                oauth_attr["credential_family"],
            }
        ) == 3

        assert cursor_attr["route_family"] != oidc_attr["route_family"]
        assert oidc_attr["route_family"] != oauth_attr["route_family"]
        assert cursor_attr["cooldown_identity_tag"] != oidc_attr["cooldown_identity_tag"]
        assert oidc_attr["cooldown_identity_tag"] != oauth_attr["cooldown_identity_tag"]
        assert cursor_attr["cooldown_key"] != oidc_attr["cooldown_key"]
        assert oidc_attr["cooldown_key"] != oauth_attr["cooldown_key"]
        assert cursor_attr["quota_cooldown_key"] is None
        assert oidc_attr["quota_cooldown_key"] == (
            f"xai:__account_quota__:{CODEX_AUTO_AGENT_XAI_LANE_KEY}"
        )
        assert oauth_attr["quota_cooldown_key"] == (
            f"xai:__account_quota__:{CODEX_AUTO_AGENT_XAI_OAUTH_LANE_KEY}"
        )
        assert oidc_attr["quota_cooldown_key"] != oauth_attr["quota_cooldown_key"]

        cursor_public = _public_candidate(cursor, route_family=_CURSOR_ROUTE)
        oidc_public = _public_candidate(oidc, route_family=_OIDC_ROUTE)
        oauth_public = _public_candidate(oauth, route_family=_OAUTH_ROUTE)
        assert _is_codex_auto_agent_cursor_agent_candidate(cursor_public)
        assert not _is_codex_auto_agent_xai_candidate(cursor_public)
        assert not _is_codex_auto_agent_grok_account_quota_candidate(cursor_public)
        assert _is_codex_auto_agent_xai_candidate(oidc_public)
        assert _is_codex_auto_agent_xai_candidate(oauth_public)
        assert _is_codex_auto_agent_grok_account_quota_candidate(oidc_public)
        assert _is_codex_auto_agent_grok_account_quota_candidate(oauth_public)
        assert not _is_codex_auto_agent_cursor_agent_candidate(oidc_public)
        assert not _is_codex_auto_agent_cursor_agent_candidate(oauth_public)

        selected = snapshot_select._select_snapshot_candidates(
            "sota-xai",
            ingress="codex",
        )
        assert [
            (row["provider"], row["model"], row["route_family"], row["selection_priority"])
            for row in selected
        ] == list(_SOTA_XAI_ORDER)
        assert [row["cooldown_identity_tag"] for row in selected] == [
            cursor_attr["cooldown_identity_tag"],
            oidc_attr["cooldown_identity_tag"],
            oauth_attr["cooldown_identity_tag"],
        ]

        anthropic = snapshot_select._select_snapshot_candidates(
            "sota-xai",
            ingress="anthropic",
        )
        assert [
            (row["provider"], row["model"], row["route_family"], row["selection_priority"])
            for row in anthropic
        ] == [
            ("cursor_agent", _CURSOR_MODEL, _CURSOR_ANTHROPIC_ROUTE, 110),
            ("xai", _OIDC_MODEL, _OIDC_ANTHROPIC_ROUTE, 100),
            ("xai", _OAUTH_MODEL, _OAUTH_ANTHROPIC_ROUTE, 90),
        ]
        assert len({row["cooldown_identity_tag"] for row in anthropic}) == 3
        assert [row["cooldown_identity_tag"] for row in anthropic] != [
            row["cooldown_identity_tag"] for row in selected
        ]
    finally:
        snapshot_select.set_active_routing_snapshot(previous)


def test_provider_xai_compiled_candidates_keep_distinct_oidc_and_oauth_attribution() -> None:
    snapshot = compile_directory(DEFAULT_CONFIG_DIR)
    previous = snapshot_select.get_active_routing_snapshot()
    snapshot_select.set_active_routing_snapshot(snapshot)
    try:
        compiled = snapshot.aliases["provider-xai"].candidates
        oidc, oauth = compiled
        oidc_attr = _attribution(
            oidc, owning_alias="provider-xai", route_family=_OIDC_ROUTE
        )
        oauth_attr = _attribution(
            oauth, owning_alias="provider-xai", route_family=_OAUTH_ROUTE
        )
        assert oidc_attr["lane_key"] == CODEX_AUTO_AGENT_XAI_LANE_KEY
        assert oauth_attr["lane_key"] == CODEX_AUTO_AGENT_XAI_OAUTH_LANE_KEY
        assert oidc_attr["credential_family"] == "xai_grok_oidc"
        assert oauth_attr["credential_family"] == "xai_oauth"
        assert oidc_attr["cooldown_identity_tag"] != oauth_attr["cooldown_identity_tag"]
        assert oidc_attr["cooldown_key"] != oauth_attr["cooldown_key"]
        assert oidc_attr["quota_cooldown_key"] != oauth_attr["quota_cooldown_key"]
        selected = snapshot_select._select_snapshot_candidates(
            "provider-xai",
            ingress="codex",
        )
        assert [
            (row["provider"], row["model"], row["route_family"], row["selection_priority"])
            for row in selected
        ] == list(_PROVIDER_XAI_ORDER)
        assert all(row["provider"] == "xai" for row in selected)
        assert all(row["model"] != _CURSOR_MODEL for row in selected)
    finally:
        snapshot_select.set_active_routing_snapshot(previous)
