"""RR-008 #2: ignored-path prefix matching must not hardcode one developer home."""

from __future__ import annotations

from litellm.integrations.aawm_agent_quality_rules import _path_has_common_ignored_prefix

PREFIXES = [".analysis/", ".claude/", ".git/"]


def test_relative_ignored_prefix_matches() -> None:
    assert _path_has_common_ignored_prefix(".analysis/foo.md", PREFIXES)
    assert _path_has_common_ignored_prefix(".claude/settings.json", PREFIXES)


def test_absolute_path_on_any_host_matches_ignored_prefix() -> None:
    # Different operator home than /home/zepfu must still match.
    assert _path_has_common_ignored_prefix(
        "/home/other/projects/litellm/.analysis/review.todo.md",
        PREFIXES,
    )
    assert _path_has_common_ignored_prefix(
        "/Users/dev/projects/myfork/.claude/agents/x.md",
        PREFIXES,
    )


def test_non_ignored_paths_do_not_match() -> None:
    assert not _path_has_common_ignored_prefix(
        "litellm/integrations/aawm_agent_quality_rules.py",
        PREFIXES,
    )
    assert not _path_has_common_ignored_prefix(
        "/home/other/projects/litellm/litellm/main.py",
        PREFIXES,
    )


def test_no_hardcoded_zepfu_home_required() -> None:
    import inspect
    from litellm.integrations import aawm_agent_quality_rules as mod

    src = inspect.getsource(mod._path_has_common_ignored_prefix)
    assert "/home/zepfu" not in src
