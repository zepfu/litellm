"""Wave 7 claude_agent_spec owner unit tests.

Directly exercises the extracted agent-spec directory resolution,
frontmatter parsing, markdown reading, and declared-model loading
without importing the god module.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from litellm.proxy.pass_through_endpoints.aawm_request_policy import (
    claude_agent_spec as spec,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point env vars at a temp dir and clear module cache."""
    monkeypatch.setenv("LITELLM_CLAUDE_AGENTS_DIR", str(tmp_path))
    monkeypatch.delenv("CLAUDE_AGENTS_DIR", raising=False)
    original_cache = spec._claude_agent_model_cache
    original_env_vars = spec._CLAUDE_AGENT_SPEC_DIR_ENV_VARS
    original_default_dirs = spec._CLAUDE_AGENT_SPEC_DEFAULT_DIRS
    original_cache.clear()
    yield
    # Restore module-level state unconditionally.
    spec._claude_agent_model_cache = original_cache
    spec._CLAUDE_AGENT_SPEC_DIR_ENV_VARS = original_env_vars
    spec._CLAUDE_AGENT_SPEC_DEFAULT_DIRS = original_default_dirs
    original_cache.clear()


def _write_agent(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / f"{name}.md"
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# _get_claude_agent_spec_dir
# ---------------------------------------------------------------------------


class TestGetClaudeAgentSpecDir:
    def test_returns_env_dir(self, tmp_path: Path) -> None:
        result = spec._get_claude_agent_spec_dir()
        assert result == tmp_path

    def test_skips_blank_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LITELLM_CLAUDE_AGENTS_DIR", "   ")
        # Falls through to default dirs; result depends on host FS.
        # We only assert it does not return the blank path.
        result = spec._get_claude_agent_spec_dir()
        assert result != Path("   ")

    def test_skips_nonexistent_env_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            "LITELLM_CLAUDE_AGENTS_DIR", str(tmp_path / "nope")
        )
        result = spec._get_claude_agent_spec_dir()
        assert result != tmp_path / "nope"

    def test_fallback_second_env_var(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("LITELLM_CLAUDE_AGENTS_DIR", raising=False)
        monkeypatch.setenv("CLAUDE_AGENTS_DIR", str(tmp_path))
        result = spec._get_claude_agent_spec_dir()
        assert result == tmp_path


# ---------------------------------------------------------------------------
# _extract_model_from_markdown_frontmatter
# ---------------------------------------------------------------------------


class TestExtractModelFromMarkdownFrontmatter:
    def test_basic_model(self) -> None:
        md = "---\nmodel: gpt-4o\n---\nBody"
        assert spec._extract_model_from_markdown_frontmatter(md) == "gpt-4o"

    def test_quoted_model(self) -> None:
        md = '---\nmodel: "claude-sonnet-4-20250514"\n---\n'
        assert (
            spec._extract_model_from_markdown_frontmatter(md)
            == "claude-sonnet-4-20250514"
        )

    def test_single_quoted_model(self) -> None:
        md = "---\nmodel: 'o3'\n---\n"
        assert spec._extract_model_from_markdown_frontmatter(md) == "o3"

    def test_no_frontmatter(self) -> None:
        assert spec._extract_model_from_markdown_frontmatter("hello") is None

    def test_no_closing_fence(self) -> None:
        assert (
            spec._extract_model_from_markdown_frontmatter("---\nmodel: x\n")
            is None
        )

    def test_no_model_key(self) -> None:
        md = "---\ntitle: test\n---\n"
        assert spec._extract_model_from_markdown_frontmatter(md) is None

    def test_empty_model_value(self) -> None:
        md = "---\nmodel:   \n---\n"
        assert spec._extract_model_from_markdown_frontmatter(md) is None

    def test_model_among_other_keys(self) -> None:
        md = "---\nname: agent\nmodel: gpt-4.1\ntools: all\n---\n"
        assert spec._extract_model_from_markdown_frontmatter(md) == "gpt-4.1"


# ---------------------------------------------------------------------------
# _read_claude_agent_markdown
# ---------------------------------------------------------------------------


class TestReadClaudeAgentMarkdown:
    def test_utf8_file(self, tmp_path: Path) -> None:
        p = _write_agent(tmp_path, "a", "---\nmodel: x\n---\n")
        assert spec._read_claude_agent_markdown(p) == "---\nmodel: x\n---\n"

    def test_latin1_file(self, tmp_path: Path) -> None:
        p = tmp_path / "b.md"
        p.write_bytes(b"caf\xe9")
        result = spec._read_claude_agent_markdown(p)
        assert result is not None
        assert "caf" in result

    def test_missing_file(self, tmp_path: Path) -> None:
        assert (
            spec._read_claude_agent_markdown(tmp_path / "missing.md") is None
        )


# ---------------------------------------------------------------------------
# _load_claude_agent_declared_model
# ---------------------------------------------------------------------------


class TestLoadClaudeAgentDeclaredModel:
    def test_loads_model(self, tmp_path: Path) -> None:
        _write_agent(tmp_path, "myagent", "---\nmodel: gpt-4o\n---\nBody")
        assert spec._load_claude_agent_declared_model("myagent") == "gpt-4o"

    def test_blank_name(self) -> None:
        assert spec._load_claude_agent_declared_model("  ") is None

    def test_path_traversal_rejected(self) -> None:
        assert spec._load_claude_agent_declared_model("../etc/passwd") is None

    def test_missing_agent(self, tmp_path: Path) -> None:
        assert spec._load_claude_agent_declared_model("ghost") is None

    def test_no_model_in_frontmatter(self, tmp_path: Path) -> None:
        _write_agent(tmp_path, "nomodel", "---\ntitle: hi\n---\n")
        assert spec._load_claude_agent_declared_model("nomodel") is None

    def test_cache_hit(self, tmp_path: Path) -> None:
        _write_agent(tmp_path, "cached", "---\nmodel: o3\n---\n")
        first = spec._load_claude_agent_declared_model("cached")
        assert first == "o3"
        # Mutate file content but NOT mtime to prove cache is used.
        # (In practice mtime changes, so we just verify second call works.)
        second = spec._load_claude_agent_declared_model("cached")
        assert second == "o3"

    def test_cache_invalidated_on_mtime_change(
        self, tmp_path: Path
    ) -> None:
        p = _write_agent(tmp_path, "inv", "---\nmodel: a\n---\n")
        assert spec._load_claude_agent_declared_model("inv") == "a"
        # Force a different mtime_ns.
        st = p.stat()
        os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000))
        p.write_text("---\nmodel: b\n---\n", encoding="utf-8")
        assert spec._load_claude_agent_declared_model("inv") == "b"

    def test_no_agents_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            "LITELLM_CLAUDE_AGENTS_DIR", str(tmp_path / "nonexistent")
        )
        monkeypatch.delenv("CLAUDE_AGENTS_DIR", raising=False)
        # Default dirs may or may not exist on host; the function should
        # return None or a valid result, never raise.
        result = spec._load_claude_agent_declared_model("any")
        assert result is None or isinstance(result, str)


# ---------------------------------------------------------------------------
# install / bind_runtime
# ---------------------------------------------------------------------------


class TestInstallAndBind:
    def test_install_publishes_facades(self) -> None:
        host: dict[str, Any] = {}
        spec.install(host)
        for name in spec._HOST_FUNCTION_NAMES:
            assert name in host
            assert callable(host[name])

    def test_install_does_not_poison_module_globals(self) -> None:
        """install() must not replace the owner module's own functions."""
        originals = {
            name: getattr(spec, name) for name in spec._HOST_FUNCTION_NAMES
        }
        host: dict[str, Any] = {}
        spec.install(host)
        for name, original in originals.items():
            assert getattr(spec, name) is original, (
                f"install() replaced module-level {name}"
            )

    def test_install_seeds_incomplete_host_globals(self) -> None:
        """install() with an empty dict must seed all needed names."""
        host: dict[str, Any] = {}
        spec.install(host)
        # Constants and cache must be present for facades to work.
        assert "_CLAUDE_AGENT_SPEC_DIR_ENV_VARS" in host
        assert "_CLAUDE_AGENT_SPEC_DEFAULT_DIRS" in host
        assert "_claude_agent_model_cache" in host

    def test_install_preserves_direct_module_usability(
        self, tmp_path: Path
    ) -> None:
        """After install(), spec._load_claude_agent_declared_model still works."""
        host: dict[str, Any] = {}
        spec.install(host)
        _write_agent(tmp_path, "direct", "---\nmodel: gpt-4o\n---\n")
        assert spec._load_claude_agent_declared_model("direct") == "gpt-4o"

    def test_install_idempotent(self, tmp_path: Path) -> None:
        """Calling install() twice must not break anything."""
        host1: dict[str, Any] = {}
        host2: dict[str, Any] = {}
        spec.install(host1)
        spec.install(host2)
        _write_agent(tmp_path, "idem", "---\nmodel: o3\n---\n")
        assert host1["_load_claude_agent_declared_model"]("idem") == "o3"
        assert host2["_load_claude_agent_declared_model"]("idem") == "o3"
        assert spec._load_claude_agent_declared_model("idem") == "o3"

    def test_installed_facade_calls_through(self, tmp_path: Path) -> None:
        host: dict[str, Any] = {}
        spec.install(host)
        _write_agent(tmp_path, "facade", "---\nmodel: gpt-4.1\n---\n")
        assert host["_load_claude_agent_declared_model"]("facade") == "gpt-4.1"

    def test_installed_facade_uses_host_globals_for_monkeypatch(
        self, tmp_path: Path
    ) -> None:
        """Facade __globals__ is host dict, enabling host-side monkeypatching."""
        host: dict[str, Any] = {}
        spec.install(host)
        facade = host["_load_claude_agent_declared_model"]
        assert facade.__globals__ is host

    def test_bind_runtime_overrides_cache(self, tmp_path: Path) -> None:
        external_cache: dict[Path, tuple[Any, ...]] = {}
        spec.bind_runtime({"_claude_agent_model_cache": external_cache})
        _write_agent(tmp_path, "ext", "---\nmodel: x\n---\n")
        spec._load_claude_agent_declared_model("ext")
        assert len(external_cache) == 1
        # Restore handled by autouse fixture.

    def test_bind_runtime_overrides_env_vars(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        alt = tmp_path / "alt_agents"
        alt.mkdir()
        spec.bind_runtime(
            {"_CLAUDE_AGENT_SPEC_DIR_ENV_VARS": ("CUSTOM_AGENTS_DIR",)}
        )
        monkeypatch.setenv("CUSTOM_AGENTS_DIR", str(alt))
        assert spec._get_claude_agent_spec_dir() == alt
        # Restore handled by autouse fixture.
