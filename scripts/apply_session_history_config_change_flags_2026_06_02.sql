-- Add and backfill session_history flags for sensitive/config file changes.
-- Intended target: aawm_tristore.public.session_history.

BEGIN;

ALTER TABLE public.session_history
    ADD COLUMN IF NOT EXISTS changed_pre_commit_config BOOLEAN,
    ADD COLUMN IF NOT EXISTS changed_env_file BOOLEAN,
    ADD COLUMN IF NOT EXISTS changed_pyproject_toml BOOLEAN,
    ADD COLUMN IF NOT EXISTS changed_gitignore BOOLEAN;

WITH modified_path_rows AS (
    SELECT
        activity.litellm_call_id,
        lower(
            regexp_replace(
                replace(
                    trim(both '"' from trim(both '''' from trim(path_value))),
                    chr(92),
                    '/'
                ),
                '^.*/',
                ''
            )
        ) AS basename
    FROM public.session_history_tool_activity AS activity
    CROSS JOIN LATERAL jsonb_array_elements_text(
        CASE
            WHEN jsonb_typeof(activity.file_paths_modified) = 'array'
                THEN activity.file_paths_modified
            ELSE '[]'::jsonb
        END
    ) AS modified_paths(path_value)
    WHERE NULLIF(activity.litellm_call_id, '') IS NOT NULL
),
config_change_flags AS (
    SELECT
        litellm_call_id,
        bool_or(basename IN ('.pre-commit-config.yaml', '.pre-commit-config.yml'))
            AS changed_pre_commit_config,
        bool_or(basename LIKE '.env%') AS changed_env_file,
        bool_or(basename = 'pyproject.toml') AS changed_pyproject_toml,
        bool_or(basename = '.gitignore') AS changed_gitignore
    FROM modified_path_rows
    GROUP BY litellm_call_id
)
UPDATE public.session_history AS history
SET
    changed_pre_commit_config = flags.changed_pre_commit_config,
    changed_env_file = flags.changed_env_file,
    changed_pyproject_toml = flags.changed_pyproject_toml,
    changed_gitignore = flags.changed_gitignore
FROM config_change_flags AS flags
WHERE history.litellm_call_id = flags.litellm_call_id;

COMMIT;
