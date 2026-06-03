from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
D1_190_SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "apply_rate_limit_intervals_mview_2026_06_03_antigravity.sql"
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
