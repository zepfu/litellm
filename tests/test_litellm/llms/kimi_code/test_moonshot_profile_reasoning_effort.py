"""MS-033 regression: moonshot profile reasoning effort contract.

The moonshot agent profile must advertise a reasoning effort that is:
1. Present in the Codex model catalog's supported_reasoning_levels for
   sota-moonshot (low, medium, high, xhigh -- NOT max).
2. Correctly mapped by the kimi_code adapter to the native upstream effort.
3. Not silently overwritten when explicitly requested by the caller.

The pre-egress rejection was: "Reasoning effort max is not supported for
model sota-moonshot. Supported reasoning efforts: low, medium, high,
xhigh" because the profile pinned ``max`` which is absent from the catalog.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

import pytest

from litellm.llms.kimi_code.adapters.adapter import (
    _K3_REASONING_EFFORT_MAP,
    _K3_VARIANT_EFFORTS,
    _apply_kimi_reasoning_effort,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[4]
_MODEL_CATALOG_PATH = _REPO_ROOT / "scripts" / "local-ci" / "codex_moonshot_model_catalog.json"
_MOONSHOT_PROFILE_PATH = Path.home() / ".codex" / "agents" / "moonshot.toml"


def _load_model_catalog() -> dict[str, Any]:
    with open(_MODEL_CATALOG_PATH) as f:
        return json.load(f)


def _catalog_supported_efforts() -> set[str]:
    catalog = _load_model_catalog()
    for model_entry in catalog["models"]:
        if model_entry["slug"] == "sota-moonshot":
            return {
                level["effort"]
                for level in model_entry["supported_reasoning_levels"]
            }
    raise AssertionError("sota-moonshot not found in model catalog")


# ---------------------------------------------------------------------------
# 1. Profile default resolution
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _MOONSHOT_PROFILE_PATH.exists(),
    reason="moonshot.toml profile not present on this host",
)
def test_moonshot_profile_parses_and_uses_catalog_supported_effort() -> None:
    """The profile file must parse as valid TOML and use a catalog-supported effort."""
    with open(_MOONSHOT_PROFILE_PATH, "rb") as f:
        profile = tomllib.load(f)

    assert profile["name"] == "moonshot"
    assert profile["model"] == "sota-moonshot"

    effort = profile["model_reasoning_effort"]
    supported = _catalog_supported_efforts()
    assert effort in supported, (
        f"Profile pins reasoning effort {effort!r} but the model catalog "
        f"only supports {sorted(supported)}. This causes a pre-egress "
        f"rejection and zero children spawn."
    )
    assert effort == "xhigh", (
        f"Expected xhigh (the highest catalog-supported tier), got {effort!r}"
    )


# ---------------------------------------------------------------------------
# 2. Model catalog contract: xhigh supported, max rejected
# ---------------------------------------------------------------------------


def test_catalog_supports_xhigh_but_not_max() -> None:
    """sota-moonshot must advertise xhigh and must NOT advertise max."""
    supported = _catalog_supported_efforts()
    assert "xhigh" in supported
    assert "high" in supported
    assert "max" not in supported, (
        "max must not be in the catalog; it is a native kimi_code tier "
        "that the adapter maps from xhigh, not a Codex-facing effort."
    )


def test_catalog_default_reasoning_level_is_supported() -> None:
    catalog = _load_model_catalog()
    for model_entry in catalog["models"]:
        if model_entry["slug"] == "sota-moonshot":
            default = model_entry["default_reasoning_level"]
            supported = {
                level["effort"]
                for level in model_entry["supported_reasoning_levels"]
            }
            assert default in supported
            return
    raise AssertionError("sota-moonshot not found")


# ---------------------------------------------------------------------------
# 3. Adapter effort mapping: xhigh -> native max, explicit not overwritten
# ---------------------------------------------------------------------------


def test_xhigh_maps_to_native_max_for_k3() -> None:
    """xhigh (Codex-facing) must map to max (kimi_code native) for k3."""
    assert _K3_REASONING_EFFORT_MAP["xhigh"] == "max"


def test_high_maps_to_native_high_for_k3() -> None:
    assert _K3_REASONING_EFFORT_MAP["high"] == "high"


@pytest.mark.parametrize("explicit_effort", ["high", "xhigh"])
def test_explicit_supported_effort_is_preserved_not_overwritten(
    explicit_effort: str,
) -> None:
    """When the caller explicitly requests a supported effort, the adapter
    must map it through without replacing it with a different value."""
    request_body: dict[str, Any] = {"reasoning_effort": explicit_effort}
    completion_kwargs: dict[str, Any] = {}

    _apply_kimi_reasoning_effort(
        request_body=request_body,
        upstream_model="k3",
        forced_effort=None,
        completion_kwargs=completion_kwargs,
    )

    expected_native = _K3_REASONING_EFFORT_MAP[explicit_effort]
    assert completion_kwargs["reasoning_effort"] == expected_native


def test_explicit_effort_via_reasoning_dict_is_preserved() -> None:
    """Effort nested in reasoning.effort must also be extracted and mapped."""
    request_body: dict[str, Any] = {"reasoning": {"effort": "xhigh"}}
    completion_kwargs: dict[str, Any] = {}

    _apply_kimi_reasoning_effort(
        request_body=request_body,
        upstream_model="k3",
        forced_effort=None,
        completion_kwargs=completion_kwargs,
    )

    assert completion_kwargs["reasoning_effort"] == "max"


# ---------------------------------------------------------------------------
# 4. Fail-closed: max is not a valid Codex-facing effort for this alias
# ---------------------------------------------------------------------------


def test_max_is_not_extractable_as_supported_catalog_effort() -> None:
    """max must not appear in the catalog supported set, so any profile
    or request pinning max would be rejected pre-egress by the Codex
    client's model catalog validation."""
    supported = _catalog_supported_efforts()
    assert "max" not in supported


def test_variant_forced_effort_k3_max_is_native_not_codex_facing() -> None:
    """kimi_code/k3-max forces native max effort at the adapter level.
    This is correct: the adapter speaks native kimi_code protocol.
    The Codex-facing catalog must NOT expose max as a user-selectable tier."""
    assert _K3_VARIANT_EFFORTS["kimi_code/k3-max"] == "max"
    # But the catalog does not advertise max:
    assert "max" not in _catalog_supported_efforts()
