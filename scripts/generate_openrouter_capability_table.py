#!/usr/bin/env python3
"""Deterministic OpenRouter capability table generator.

Reads ``model_prices_and_context_window.json``, filters entries whose key
starts with ``openrouter/``, and renders a Markdown capability snapshot at
``docs/my-website/docs/providers/openrouter-model-metadata.md``.

Modes
-----
write (default)
    Generate the Markdown file, overwriting any previous artifact.
--check
    Regenerate in memory and compare byte-for-byte against the on-disk
    artifact.  Exit 0 if identical, 1 if stale.  Never writes.

The source snapshot date is derived from the Git commit date of the source
file.  Pass ``--source-date YYYY-MM-DD`` to override (useful in CI or when
the source is not in a Git checkout).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = _REPO_ROOT / "model_prices_and_context_window.json"
DEFAULT_OUTPUT = (
    _REPO_ROOT
    / "docs"
    / "my-website"
    / "docs"
    / "providers"
    / "openrouter-model-metadata.md"
)

PROVIDER_PREFIX = "openrouter/"

BEGIN_MARKER = "<!-- BEGIN openrouter-capability-table (generated) -->"
END_MARKER = "<!-- END openrouter-capability-table (generated) -->"

# ── Fixed capability columns (canonical order) ──────────────────────────

CAPABILITY_FLAGS: list[str] = [
    "supports_function_calling",
    "supports_vision",
    "supports_reasoning",
    "supports_prompt_caching",
    "supports_tool_choice",
    "supports_audio_input",
    "supports_audio_output",
    "supports_pdf_input",
    "supports_video_input",
    "supports_computer_use",
    "supports_response_schema",
    "supports_system_messages",
    "supports_parallel_function_calling",
    "supports_native_cache_control",
    "supports_assistant_prefill",
    "supports_url_context",
    "supports_web_search",
]

CAP_ABBREV: dict[str, str] = {
    "supports_function_calling": "FC",
    "supports_vision": "Vis",
    "supports_reasoning": "Reas",
    "supports_prompt_caching": "Cch",
    "supports_tool_choice": "TC",
    "supports_audio_input": "AI",
    "supports_audio_output": "AO",
    "supports_pdf_input": "PDF",
    "supports_video_input": "VI",
    "supports_computer_use": "CU",
    "supports_response_schema": "RS",
    "supports_system_messages": "SM",
    "supports_parallel_function_calling": "PF",
    "supports_native_cache_control": "NC",
    "supports_assistant_prefill": "AP",
    "supports_url_context": "UC",
    "supports_web_search": "WS",
}

# ── Helpers ──────────────────────────────────────────────────────────────


def load_source(path: Path) -> dict[str, Any]:
    """Load and validate the source JSON mapping."""
    raw = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"error: invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(
            f"error: top-level JSON value in {path} must be an object"
        )
    return data


def filter_openrouter(
    data: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    """Return sorted ``(key, entry)`` pairs for ``openrouter/`` keys.

    Raises ``SystemExit`` on duplicate normalized (lower-cased) IDs.
    """
    entries: list[tuple[str, dict[str, Any]]] = []
    seen_normalized: dict[str, str] = {}
    for key in sorted(data):
        if not key.startswith(PROVIDER_PREFIX):
            continue
        norm = key.lower()
        if norm in seen_normalized:
            raise SystemExit(
                f"error: duplicate normalized openrouter ID {norm!r} "
                f"(keys: {seen_normalized[norm]!r}, {key!r})"
            )
        seen_normalized[norm] = key
        val = data[key]
        if not isinstance(val, dict):
            raise SystemExit(f"error: entry {key!r} is not a JSON object")
        entries.append((key, val))
    return entries


def resolve_source_date(source_path: Path, explicit: str | None) -> str:
    """Return ``YYYY-MM-DD`` from Git or the explicit override."""
    if explicit:
        return explicit
    try:
        proc = subprocess.run(
            ["git", "log", "-1", "--format=%cs", "--", str(source_path)],
            capture_output=True,
            text=True,
            cwd=str(source_path.parent),
            check=True,
        )
        date_str = proc.stdout.strip()
        if not date_str:
            raise ValueError("empty git output")
        return date_str
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        ValueError,
    ) as exc:
        raise SystemExit(
            f"error: cannot determine source date from Git ({exc}); "
            "pass --source-date YYYY-MM-DD"
        ) from exc


def compute_sha256(path: Path) -> str:
    """SHA-256 hex digest of the raw source bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cap_cell(entry: dict[str, Any], flag: str) -> str:
    """Render a capability flag: ``Y`` / ``N`` / ``-`` (absent)."""
    val = entry.get(flag)
    if val is True:
        return "Y"
    if val is False:
        return "N"
    return "-"


def _int_or_dash(entry: dict[str, Any], field: str) -> str:
    val = entry.get(field)
    if val is None:
        return "-"
    try:
        return str(int(val))
    except (ValueError, TypeError):
        return str(val)


def _context_cell(entry: dict[str, Any]) -> str:
    """Context window: prefer ``max_input_tokens``, fall back to ``max_tokens``."""
    for field in ("max_input_tokens", "max_tokens"):
        val = entry.get(field)
        if val is not None:
            try:
                return str(int(val))
            except (ValueError, TypeError):
                return str(val)
    return "-"


def _list_cell(entry: dict[str, Any], field: str) -> str:
    val = entry.get(field)
    if not val or not isinstance(val, list):
        return "-"
    return ", ".join(sorted(str(v) for v in val))


# ── Rendering ────────────────────────────────────────────────────────────


def render_markdown(  # noqa: PLR0915
    entries: list[tuple[str, dict[str, Any]]],
    source_date: str,
    source_sha: str,
    source_name: str,
) -> str:
    """Render the full Markdown artifact as a string."""
    lines: list[str] = []
    a = lines.append

    a(BEGIN_MARKER)
    a("<!--")
    a("  GENERATED FILE -- DO NOT EDIT MANUALLY")
    a("  Generator: scripts/generate_openrouter_capability_table.py")
    a(f"  Source: {source_name}")
    a(f"  Source SHA256: {source_sha}")
    a(f"  Source snapshot date: {source_date}")
    a(f"  OpenRouter model count: {len(entries)}")
    a("-->")
    a("")
    a("# OpenRouter Model Metadata")
    a("")
    a(
        "> **LiteLLM metadata snapshot** -- not the live OpenRouter model"
        " catalog."
    )
    a(f"> Generated from `{source_name}` by")
    a("> `scripts/generate_openrouter_capability_table.py`.")
    a("> Regenerate: `python scripts/generate_openrouter_capability_table.py`")
    a("")
    a(f"- **Source snapshot date:** {source_date}")
    a(f"- **Source SHA256:** `{source_sha}`")
    a(f"- **Model count:** {len(entries)}")
    a("")
    a("## Capability Legend")
    a("")
    a("| Symbol | Meaning |")
    a("|--------|---------|")
    a("| Y | Explicitly declared as supported |")
    a("| N | Explicitly declared as not supported |")
    a("| - | Not declared in metadata |")
    a("")
    a("### Column Abbreviations")
    a("")
    a("| Abbrev | Capability |")
    a("|--------|------------|")
    for flag in CAPABILITY_FLAGS:
        a(f"| {CAP_ABBREV[flag]} | {flag} |")
    a("")
    a("## Models")
    a("")

    # Table header
    cap_hdrs = [CAP_ABBREV[f] for f in CAPABILITY_FLAGS]
    header = (
        ["Model", "Mode", "Ctx", "MaxOut"]
        + cap_hdrs
        + ["InMod", "OutMod", "Endpoints"]
    )
    a("| " + " | ".join(header) + " |")
    a("|" + "|".join(["---"] * len(header)) + "|")

    for key, entry in entries:
        mode = entry.get("mode")
        if not isinstance(mode, str) or not mode:
            mode = "-"
        row = [
            key,
            mode,
            _context_cell(entry),
            _int_or_dash(entry, "max_output_tokens"),
        ]
        for flag in CAPABILITY_FLAGS:
            row.append(_cap_cell(entry, flag))
        row.append(_list_cell(entry, "supported_modalities"))
        row.append(_list_cell(entry, "supported_output_modalities"))
        row.append(_list_cell(entry, "supported_endpoints"))
        a("| " + " | ".join(row) + " |")

    a("")
    a(END_MARKER)
    a("")  # trailing newline
    return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate OpenRouter capability table from LiteLLM model metadata."
        ),
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Path to model_prices_and_context_window.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to the generated Markdown file",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare generated output against on-disk file; exit 1 if stale",
    )
    parser.add_argument(
        "--source-date",
        default=None,
        help="Explicit source snapshot date (YYYY-MM-DD); overrides Git lookup",
    )
    args = parser.parse_args(argv)

    source: Path = args.source
    output: Path = args.output

    data = load_source(source)
    entries = filter_openrouter(data)
    source_date = resolve_source_date(source, args.source_date)
    source_sha = compute_sha256(source)
    content = render_markdown(entries, source_date, source_sha, source.name)

    if args.check:
        if not output.exists():
            print(  # noqa: T201
                f"error: {output} does not exist; run without --check first",
                file=sys.stderr,
            )
            return 1
        existing = output.read_text(encoding="utf-8")
        if existing == content:
            print(f"ok: {output} is up to date")  # noqa: T201
            return 0
        print(  # noqa: T201
            f"error: {output} is stale; regenerate without --check",
            file=sys.stderr,
        )
        return 1

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    print(f"wrote {output} ({len(entries)} models)")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
