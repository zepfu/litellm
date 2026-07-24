"""Wave 0 guardrail: immutable upstream-owned source-byte baseline.

Guards Waves 4-6F (god-module decomposition) against accidentally rewriting
upstream-owned bytes in ``llm_passthrough_endpoints.py`` while AAWM-owned
bands move out into packages, per
``.analysis/plan-godmodule-decomposition-r3-remediation-2026-07-23.md``
Wave 0.

The fixture
``fixtures/upstream_owned_baseline_e3dc89f634.json`` was generated once from
develop commit ``e3dc89f634a61e89aeaab98c7fbf91b7bdae896c`` via:

    git show e3dc89f634a61e89aeaab98c7fbf91b7bdae896c:\
litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py \
      | <AST-address + SHA-256-of-exact-span extraction, see landing note>

For each covered upstream-owned function/decorated endpoint it records a
unique semantic AST address (``function:<qualified_name>`` or
``decorated_function:<qualified_name>``), a ``source_kind`` field, and the
full SHA-256 of the exact UTF-8 source span (first decorator line, when
present, through ``end_lineno``, preserving original line endings) -- never
a mutable line-range anchor.

``test_upstream_band_byte_stable`` reads ONLY checked-in source + fixture
data (no ``.git`` at runtime) and must PASS now: the working tree is that
same commit for upstream-owned functions on develop.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[4]
_FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "upstream_owned_baseline_e3dc89f634.json"
_EXPECTED_COMMIT = "e3dc89f634a61e89aeaab98c7fbf91b7bdae896c"
_EXPECTED_PATH = "litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py"


def _load_manifest() -> dict[str, Any]:
    with _FIXTURE_PATH.open("r", encoding="utf-8") as fixture_file:
        return json.load(fixture_file)


def test_manifest_metadata_and_addresses_are_valid() -> None:
    manifest = _load_manifest()

    assert manifest["commit"] == _EXPECTED_COMMIT
    assert manifest["path"] == _EXPECTED_PATH

    entries = manifest["entries"]
    assert entries, "manifest must cover at least one upstream-owned symbol"

    addresses = [entry["address"] for entry in entries]
    assert len(addresses) == len(set(addresses)), "manifest addresses must be unique"

    for entry in entries:
        address = entry["address"]
        assert address.startswith("function:") or address.startswith(
            "decorated_function:"
        ), f"unexpected address shape: {address!r}"
        assert entry["source_kind"] in ("function", "decorated_function")
        assert address == f"{entry['source_kind']}:{entry['qualified_name']}"

        digest = entry["sha256"]
        assert isinstance(digest, str)
        assert len(digest) == 64
        int(digest, 16)  # full hex SHA-256, raises ValueError if not hex


def _resolve_working_tree_source_digests() -> dict[str, str]:
    """Recompute SHA-256 spans for every manifest-addressed function in the
    CURRENT working-tree source, keyed by semantic AST address.

    Reads only the checked-in source file (no ``.git`` at runtime).
    """
    source_file = _REPO_ROOT / _EXPECTED_PATH
    source_text = source_file.read_text(encoding="utf-8")
    lines = source_text.splitlines(keepends=True)
    tree = ast.parse(source_text, filename=str(source_file))

    digests_by_address: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        has_decorators = bool(node.decorator_list)
        source_kind = "decorated_function" if has_decorators else "function"
        address = f"{source_kind}:{node.name}"
        if address in digests_by_address:
            # Duplicate top-level function names are out of scope for this
            # baseline; the manifest only ever addresses unique names, so a
            # collision here just means this address is not a candidate.
            continue
        start_line = node.decorator_list[0].lineno if has_decorators else node.lineno
        end_line = node.end_lineno
        assert end_line is not None
        span_text = "".join(lines[start_line - 1 : end_line])
        digests_by_address[address] = hashlib.sha256(span_text.encode("utf-8")).hexdigest()
    return digests_by_address


def test_upstream_band_byte_stable() -> None:
    manifest = _load_manifest()
    working_tree_digests = _resolve_working_tree_source_digests()

    mismatches: list[str] = []
    missing: list[str] = []
    for entry in manifest["entries"]:
        address = entry["address"]
        working_tree_digest = working_tree_digests.get(address)
        if working_tree_digest is None:
            missing.append(address)
            continue
        if working_tree_digest != entry["sha256"]:
            mismatches.append(address)

    assert not missing, f"manifest addresses missing from working-tree source: {missing}"
    assert not mismatches, (
        "upstream-owned source bytes changed for: "
        f"{mismatches}. If this is an approved upstream-sync update, land a "
        "separate reviewed baseline-replacement change; do not edit expected "
        "hashes inline in an extraction wave."
    )
