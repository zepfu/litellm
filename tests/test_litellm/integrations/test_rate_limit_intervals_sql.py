from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
D1_190_SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "apply_rate_limit_intervals_mview_2026_06_03_antigravity.sql"
)
LEGACY_SCRIPT = REPO_ROOT / "scripts" / "apply_rate_limit_intervals_mview_2026_05_23.sql"

XAI_WEEKLY_100_PCT_EXCEPTION = (
    "provider = 'xai'\n"
    "              AND quota_key = 'xai_grok_build_weekly_credits:credits'"
)


def test_antigravity_rate_limit_intervals_script_projects_pool_rows() -> None:
    sql = D1_190_SCRIPT.read_text(encoding="utf-8")

    assert "'antigravity'" in sql
    assert "'antigravity_code_assist:gemini_pool'" in sql
    assert "'antigravity_code_assist:vertex_pool'" in sql
    assert "provider = 'antigravity'" in sql
    assert "WHEN rate_limit_observations.provider = 'antigravity'::text" in sql
    assert "THEN NULL::text" in sql
    assert "remaining_pct < 100" in sql
    assert "provider <> 'antigravity'" in sql
    assert "antigravity_code_assist:gemini_pool', 'antigravity_code_assist:vertex_pool']) THEN 'short'" not in sql
    assert "COALESCE(model, ''::text), quota_key, quota_type" in sql
    assert "COALESCE(model, ''::text)," in sql


def test_xai_grok_weekly_credits_quota_key_allowed_and_mapped_weekly() -> None:
    sql = D1_190_SCRIPT.read_text(encoding="utf-8")

    assert "'xai_grok_build_weekly_credits:credits'" in sql
    assert (
        "WHEN quota_key = ANY (ARRAY['anthropic_unified_7d:7d', 'codex:secondary', 'xai_grok_build_weekly_credits:credits']) THEN 'weekly'"
        in sql
    )


def test_xai_grok_weekly_credits_allows_hundred_pct_remaining_antigravity_script() -> None:
    sql = D1_190_SCRIPT.read_text(encoding="utf-8")

    assert XAI_WEEKLY_100_PCT_EXCEPTION in sql
    assert "remaining_pct < 100" in sql


def test_legacy_rate_limit_intervals_script_includes_weekly_credits_key() -> None:
    sql = LEGACY_SCRIPT.read_text(encoding="utf-8")

    assert "'xai_grok_build_weekly_credits:credits'" in sql
    assert "xai_grok_build_weekly_credits:credits']) THEN 'weekly'" in sql


def test_xai_grok_weekly_credits_allows_hundred_pct_remaining_legacy_script() -> None:
    sql = LEGACY_SCRIPT.read_text(encoding="utf-8")

    assert XAI_WEEKLY_100_PCT_EXCEPTION in sql
    assert "remaining_pct < 100" in sql
