"""Tests for scripts/generate_openrouter_capability_table.py.

Covers determinism, filtering, sorting, markers, date/hash derivation,
absent-vs-false capability semantics, stale --check failure, no-mutation
guarantee, malformed input, and duplicate normalized IDs.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "generate_openrouter_capability_table.py"


def _load_module():
    name = "generate_openrouter_capability_table"
    spec = importlib.util.spec_from_file_location(name, _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gen():
    return _load_module()


# ── Fixture helpers ──────────────────────────────────────────────────────


def _write_source(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "model_prices_and_context_window.json"
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return p


MINIMAL_DATA = {
    "openrouter/alpha/model-a": {
        "mode": "chat",
        "max_input_tokens": 128000,
        "max_output_tokens": 4096,
        "supports_function_calling": True,
        "supports_vision": False,
        "litellm_provider": "openrouter",
    },
    "openrouter/beta/model-b": {
        "mode": "embedding",
        "max_tokens": 8192,
        "supports_reasoning": True,
        "litellm_provider": "openrouter",
    },
    "gpt-4": {
        "mode": "chat",
        "litellm_provider": "openai",
    },
}


# ── Determinism ──────────────────────────────────────────────────────────


class TestDeterminism:
    def test_same_input_same_output(self, gen, tmp_path):
        src = _write_source(tmp_path, MINIMAL_DATA)
        entries = gen.filter_openrouter(gen.load_source(src))
        sha = gen.compute_sha256(src)
        out1 = gen.render_markdown(entries, "2026-01-01", sha, src.name)
        out2 = gen.render_markdown(entries, "2026-01-01", sha, src.name)
        assert out1 == out2

    def test_key_insertion_order_irrelevant(self, gen, tmp_path):
        """JSON key order must not affect output (we sort)."""
        from collections import OrderedDict

        d1 = OrderedDict(
            [
                ("openrouter/z/last", {"mode": "chat"}),
                ("openrouter/a/first", {"mode": "chat"}),
            ]
        )
        d2 = OrderedDict(
            [
                ("openrouter/a/first", {"mode": "chat"}),
                ("openrouter/z/last", {"mode": "chat"}),
            ]
        )
        s1 = _write_source(tmp_path, d1)
        entries1 = gen.filter_openrouter(gen.load_source(s1))
        out1 = gen.render_markdown(entries1, "2026-01-01", "abc", "f.json")

        s2 = tmp_path / "alt.json"
        s2.write_text(json.dumps(d2, indent=2), encoding="utf-8")
        entries2 = gen.filter_openrouter(gen.load_source(s2))
        out2 = gen.render_markdown(entries2, "2026-01-01", "abc", "f.json")
        assert out1 == out2


# ── Filtering ────────────────────────────────────────────────────────────


class TestFiltering:
    def test_only_openrouter_prefix(self, gen, tmp_path):
        src = _write_source(tmp_path, MINIMAL_DATA)
        entries = gen.filter_openrouter(gen.load_source(src))
        keys = [k for k, _ in entries]
        assert all(k.startswith("openrouter/") for k in keys)
        assert "gpt-4" not in keys
        assert len(keys) == 2

    def test_empty_source(self, gen, tmp_path):
        src = _write_source(tmp_path, {"gpt-4": {"mode": "chat"}})
        entries = gen.filter_openrouter(gen.load_source(src))
        assert entries == []


# ── Sorting ──────────────────────────────────────────────────────────────


class TestSorting:
    def test_sorted_by_key(self, gen, tmp_path):
        data = {
            "openrouter/z/zeta": {"mode": "chat"},
            "openrouter/a/alpha": {"mode": "chat"},
            "openrouter/m/mid": {"mode": "chat"},
        }
        src = _write_source(tmp_path, data)
        entries = gen.filter_openrouter(gen.load_source(src))
        keys = [k for k, _ in entries]
        assert keys == sorted(keys)


# ── Markers ──────────────────────────────────────────────────────────────


class TestMarkers:
    def test_begin_and_end_present(self, gen, tmp_path):
        src = _write_source(tmp_path, MINIMAL_DATA)
        entries = gen.filter_openrouter(gen.load_source(src))
        out = gen.render_markdown(entries, "2026-01-01", "abc", src.name)
        assert gen.BEGIN_MARKER in out
        assert gen.END_MARKER in out
        assert out.index(gen.BEGIN_MARKER) < out.index(gen.END_MARKER)

    def test_generated_do_not_edit_comment(self, gen, tmp_path):
        src = _write_source(tmp_path, MINIMAL_DATA)
        entries = gen.filter_openrouter(gen.load_source(src))
        out = gen.render_markdown(entries, "2026-01-01", "abc", src.name)
        assert "GENERATED FILE -- DO NOT EDIT MANUALLY" in out


# ── Date / Hash ──────────────────────────────────────────────────────────


class TestDateHash:
    def test_explicit_source_date_used(self, gen, tmp_path):
        src = _write_source(tmp_path, MINIMAL_DATA)
        date = gen.resolve_source_date(src, "2025-12-25")
        assert date == "2025-12-25"

    def test_sha256_deterministic(self, gen, tmp_path):
        src = _write_source(tmp_path, MINIMAL_DATA)
        h1 = gen.compute_sha256(src)
        h2 = gen.compute_sha256(src)
        assert h1 == h2
        assert len(h1) == 64

    def test_sha256_in_output(self, gen, tmp_path):
        src = _write_source(tmp_path, MINIMAL_DATA)
        entries = gen.filter_openrouter(gen.load_source(src))
        sha = gen.compute_sha256(src)
        out = gen.render_markdown(entries, "2026-01-01", sha, src.name)
        assert sha in out

    def test_date_in_output(self, gen, tmp_path):
        src = _write_source(tmp_path, MINIMAL_DATA)
        entries = gen.filter_openrouter(gen.load_source(src))
        out = gen.render_markdown(entries, "2026-07-20", "abc", src.name)
        assert "2026-07-20" in out

    def test_git_date_fallback_error(self, gen, tmp_path):
        """Without Git and no explicit date, resolve_source_date exits."""
        src = _write_source(tmp_path, MINIMAL_DATA)
        # tmp_path is not a git repo; without explicit date this should fail
        with pytest.raises(SystemExit, match="cannot determine source date"):
            gen.resolve_source_date(src, None)


# ── Absent vs False semantics ────────────────────────────────────────────


class TestAbsentVsFalse:
    def test_true_renders_Y(self, gen):
        assert gen._cap_cell({"supports_function_calling": True}, "supports_function_calling") == "Y"

    def test_false_renders_N(self, gen):
        assert gen._cap_cell({"supports_vision": False}, "supports_vision") == "N"

    def test_absent_renders_dash(self, gen):
        assert gen._cap_cell({}, "supports_reasoning") == "-"

    def test_row_semantics_in_output(self, gen, tmp_path):
        data = {
            "openrouter/test/semantics": {
                "mode": "chat",
                "supports_function_calling": True,
                "supports_vision": False,
                # supports_reasoning absent
            }
        }
        src = _write_source(tmp_path, data)
        entries = gen.filter_openrouter(gen.load_source(src))
        out = gen.render_markdown(entries, "2026-01-01", "abc", src.name)
        # Find the data row
        data_rows = [
            ln for ln in out.splitlines() if ln.startswith("| openrouter/test/semantics")
        ]
        assert len(data_rows) == 1
        cells = [c.strip() for c in data_rows[0].split("|")[1:-1]]
        # Model, Mode, Ctx, MaxOut, then capability flags in order
        fc_idx = 4 + gen.CAPABILITY_FLAGS.index("supports_function_calling")
        vis_idx = 4 + gen.CAPABILITY_FLAGS.index("supports_vision")
        reas_idx = 4 + gen.CAPABILITY_FLAGS.index("supports_reasoning")
        assert cells[fc_idx] == "Y"
        assert cells[vis_idx] == "N"
        assert cells[reas_idx] == "-"


# ── Stale check / no mutation ────────────────────────────────────────────


class TestCheckMode:
    def test_check_passes_when_current(self, gen, tmp_path):
        src = _write_source(tmp_path, MINIMAL_DATA)
        out_path = tmp_path / "out.md"
        rc = gen.main(
            ["--source", str(src), "--output", str(out_path), "--source-date", "2026-01-01"]
        )
        assert rc == 0
        rc = gen.main(
            [
                "--source", str(src),
                "--output", str(out_path),
                "--source-date", "2026-01-01",
                "--check",
            ]
        )
        assert rc == 0

    def test_check_fails_when_stale(self, gen, tmp_path):
        src = _write_source(tmp_path, MINIMAL_DATA)
        out_path = tmp_path / "out.md"
        gen.main(
            ["--source", str(src), "--output", str(out_path), "--source-date", "2026-01-01"]
        )
        # Mutate the source to make output stale
        data = json.loads(src.read_text())
        data["openrouter/new/model"] = {"mode": "chat"}
        src.write_text(json.dumps(data), encoding="utf-8")
        rc = gen.main(
            [
                "--source", str(src),
                "--output", str(out_path),
                "--source-date", "2026-01-01",
                "--check",
            ]
        )
        assert rc == 1

    def test_check_does_not_write(self, gen, tmp_path):
        src = _write_source(tmp_path, MINIMAL_DATA)
        out_path = tmp_path / "out.md"
        gen.main(
            ["--source", str(src), "--output", str(out_path), "--source-date", "2026-01-01"]
        )
        before = out_path.read_bytes()
        # Make stale
        data = json.loads(src.read_text())
        data["openrouter/extra/x"] = {"mode": "chat"}
        src.write_text(json.dumps(data), encoding="utf-8")
        gen.main(
            [
                "--source", str(src),
                "--output", str(out_path),
                "--source-date", "2026-01-01",
                "--check",
            ]
        )
        after = out_path.read_bytes()
        assert before == after

    def test_check_missing_output(self, gen, tmp_path):
        src = _write_source(tmp_path, MINIMAL_DATA)
        out_path = tmp_path / "nonexistent.md"
        rc = gen.main(
            [
                "--source", str(src),
                "--output", str(out_path),
                "--source-date", "2026-01-01",
                "--check",
            ]
        )
        assert rc == 1


# ── Malformed input ──────────────────────────────────────────────────────


class TestMalformedInput:
    def test_invalid_json(self, gen, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        with pytest.raises(SystemExit, match="invalid JSON"):
            gen.load_source(bad)

    def test_top_level_not_object(self, gen, tmp_path):
        bad = tmp_path / "arr.json"
        bad.write_text("[1,2,3]", encoding="utf-8")
        with pytest.raises(SystemExit, match="must be an object"):
            gen.load_source(bad)

    def test_entry_not_dict(self, gen, tmp_path):
        data = {"openrouter/bad/entry": "not-a-dict"}
        src = _write_source(tmp_path, data)
        with pytest.raises(SystemExit, match="not a JSON object"):
            gen.filter_openrouter(gen.load_source(src))


# ── Duplicate normalized IDs ─────────────────────────────────────────────


class TestDuplicateIDs:
    def test_exact_duplicate(self, gen, tmp_path):
        # JSON spec allows duplicate keys; Python json keeps last.
        # We test the filter's own detection with a crafted dict.
        # Since Python dicts can't have duplicate keys, simulate via
        # case-variant keys that normalize to the same ID.
        data = {
            "openrouter/Case/Model": {"mode": "chat"},
            "openrouter/case/model": {"mode": "chat"},
        }
        src = _write_source(tmp_path, data)
        with pytest.raises(SystemExit, match="duplicate normalized"):
            gen.filter_openrouter(gen.load_source(src))


# ── Endpoints / modalities rendering ─────────────────────────────────────


class TestEndpointsModalities:
    def test_present_rendered(self, gen, tmp_path):
        data = {
            "openrouter/test/mod": {
                "mode": "chat",
                "supported_modalities": ["text", "image"],
                "supported_output_modalities": ["text"],
                "supported_endpoints": ["/v1/chat/completions"],
            }
        }
        src = _write_source(tmp_path, data)
        entries = gen.filter_openrouter(gen.load_source(src))
        out = gen.render_markdown(entries, "2026-01-01", "abc", src.name)
        assert "image, text" in out  # sorted
        assert "/v1/chat/completions" in out

    def test_absent_renders_dash(self, gen, tmp_path):
        data = {"openrouter/test/plain": {"mode": "chat"}}
        src = _write_source(tmp_path, data)
        entries = gen.filter_openrouter(gen.load_source(src))
        out = gen.render_markdown(entries, "2026-01-01", "abc", src.name)
        row = [ln for ln in out.splitlines() if "openrouter/test/plain" in ln][0]
        cells = [c.strip() for c in row.split("|")[1:-1]]
        # Last three columns: InMod, OutMod, Endpoints
        assert cells[-3] == "-"
        assert cells[-2] == "-"
        assert cells[-1] == "-"


# ── Row count in header ──────────────────────────────────────────────────


class TestRowCount:
    def test_count_in_output(self, gen, tmp_path):
        src = _write_source(tmp_path, MINIMAL_DATA)
        entries = gen.filter_openrouter(gen.load_source(src))
        out = gen.render_markdown(entries, "2026-01-01", "abc", src.name)
        assert "Model count:** 2" in out


# ── Snapshot disclaimer ──────────────────────────────────────────────────


class TestDisclaimer:
    def test_not_live_catalog(self, gen, tmp_path):
        src = _write_source(tmp_path, MINIMAL_DATA)
        entries = gen.filter_openrouter(gen.load_source(src))
        out = gen.render_markdown(entries, "2026-01-01", "abc", src.name)
        assert "not the live OpenRouter model catalog" in out
