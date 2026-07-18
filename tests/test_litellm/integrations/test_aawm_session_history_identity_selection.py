"""Shared session_history ordered identity-selection helper (RR-066)."""

from __future__ import annotations

from litellm.integrations.aawm_session_history.identity_selection import (
    iter_identity_candidates,
    select_first_identity,
    select_first_identity_value,
)


def test_select_first_identity_respects_priority_order() -> None:
    calls: list[str] = []

    def source_a() -> str:
        calls.append("a")
        return ""

    def source_b() -> str:
        calls.append("b")
        return "repo-b"

    def source_c() -> str:
        calls.append("c")
        return "repo-c"

    selected = select_first_identity(
        [
            ("a", source_a),
            ("b", source_b),
            ("c", source_c),
        ]
    )
    assert selected == ("b", "repo-b")
    # Short-circuit: later sources are not evaluated after first hit.
    assert calls == ["a", "b"]


def test_select_first_identity_value_and_normalize() -> None:
    value = select_first_identity_value(
        [
            ("raw", lambda: "  litellm  "),
            ("fallback", lambda: "other"),
        ],
        normalize=lambda raw: raw.strip() if isinstance(raw, str) else None,
    )
    assert value == "litellm"


def test_iter_identity_candidates_skips_absent() -> None:
    candidates = list(
        iter_identity_candidates(
            [
                ("none", lambda: None),
                ("blank", lambda: "   "),
                ("ok", lambda: "dashboard-shell"),
            ]
        )
    )
    assert candidates == [("ok", "dashboard-shell")]


def test_package_exports_identity_selection_helpers() -> None:
    from litellm.integrations import aawm_session_history as package

    assert package.select_first_identity is select_first_identity
    assert package.select_first_identity_value is select_first_identity_value
