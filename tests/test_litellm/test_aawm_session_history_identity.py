"""Focused session-history identity tests for route-rollup group headers."""

from litellm.integrations import aawm_agent_identity as identity


def test_route_rollup_group_header_label_extracts_repository_prefix() -> None:
    assert (
        identity._extract_repository_identity_from_route_rollup_group_header_label(
            "aegis@Claude[2.1.199]"
        )
        == "aegis"
    )


def test_route_rollup_group_header_label_rejects_trace_user_style_values() -> None:
    assert (
        identity._extract_repository_identity_from_route_rollup_group_header_label(
            "user-019f0f8f-4e35-7fd2-81ad-f8f6c9794845"
        )
        is None
    )


def test_route_rollup_group_header_label_rejects_non_workspace_owner_values() -> None:
    assert (
        identity._extract_repository_identity_from_route_rollup_group_header_label(
            "zepfu@Claude[2.1.199]"
        )
        is None
    )


def test_metadata_sources_prefer_explicit_repository_over_route_rollup() -> None:
    metadata = {
        "repository": "litellm",
        "aawm_route_rollup_context": {
            "group_header_label": "aegis@Claude[2.1.199]",
        },
    }
    repository, source = identity._extract_repository_identity_from_metadata_sources_with_source(
        ("metadata", metadata),
    )
    assert repository == "litellm"
    assert source == "metadata.repository"


def test_metadata_sources_recover_repository_from_route_rollup_context() -> None:
    metadata = {
        "aawm_route_rollup_context": {
            "group_header_label": "aegis@Claude[2.1.199]",
        },
    }
    repository, source = identity._extract_repository_identity_from_metadata_sources_with_source(
        ("observation.metadata", metadata),
    )
    assert repository == "aegis"
    assert (
        source
        == "observation.metadata.aawm_route_rollup_context.group_header_label"
    )


def test_tenant_metadata_sources_use_route_rollup_after_explicit_tenant_keys() -> None:
    metadata = {
        "tenant_id": "dashboard-shell",
        "aawm_route_rollup_context": {
            "group_header_label": "aegis@Claude[2.1.199]",
        },
    }
    tenant_id, source = identity._extract_tenant_identity_from_metadata_sources(
        ("metadata", metadata),
    )
    assert tenant_id == "dashboard-shell"
    assert source == "metadata.tenant_id"


def test_tenant_metadata_sources_fallback_to_route_rollup_repository_prefix() -> None:
    metadata = {
        "aawm_route_rollup_context": {
            "group_header_label": "aegis@Claude[2.1.199]",
        },
    }
    tenant_id, source = identity._extract_tenant_identity_from_metadata_sources(
        ("observation.metadata", metadata),
    )
    assert tenant_id == "aegis"
    assert (
        source
        == "observation.metadata.aawm_route_rollup_context.group_header_label"
    )


def test_codex_trace_user_is_not_promoted_when_it_matches_stale_metadata() -> None:
    kwargs = {
        "litellm_params": {
            "metadata": {
                "trace_user_id": "stale-worktree",
                "tenant_id": "stale-worktree",
                "passthrough_route_family": "codex_responses",
                "client_name": "codex-cli",
            }
        },
        "standard_logging_object": {
            "metadata": {
                "trace_name": "codex",
                "client_user_agent": "codex-cli/1.0",
            }
        },
    }
    tenant_id, source = identity._extract_tenant_identity_from_kwargs(kwargs)
    assert tenant_id is None
    assert source is None


def test_codex_route_rollup_source_does_not_bypass_tenant_trust_guard() -> None:
    source = "litellm_params.metadata.aawm_route_rollup_context.group_header_label"
    record = {
        "provider": "openrouter",
        "model": "openrouter/cohere/north-mini-code:free",
        "client_name": "codex-tui",
        "tenant_id": "stale-worktree",
        "repository": "stale-worktree",
        "metadata": {
            "client_name": "codex-tui",
            "trace_name": "codex",
            "trace_user_id": "stale-worktree",
            "tenant_id_source": source,
            "repository_source": source,
            "passthrough_route_family": "codex_responses",
        },
    }

    identity._normalize_session_tenant_on_record(record)

    assert record["tenant_id"] is None
    assert record["metadata"]["tenant_id_source"] == "repository_untrusted"
