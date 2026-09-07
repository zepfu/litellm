ALTER TABLE model_mapping_versions ADD COLUMN change_kind TEXT NOT NULL DEFAULT 'prospective';
ALTER TABLE model_mapping_versions ADD COLUMN valid_from TEXT;
ALTER TABLE model_mapping_versions ADD COLUMN valid_until TEXT;
ALTER TABLE model_mapping_versions ADD COLUMN published_at TEXT;
ALTER TABLE model_mapping_versions ADD COLUMN supersedes_version TEXT;
ALTER TABLE model_mapping_versions ADD COLUMN correction_of_version TEXT;
ALTER TABLE model_mapping_versions ADD COLUMN provenance_json TEXT NOT NULL DEFAULT '{}';
ALTER TABLE model_mapping_versions ADD COLUMN warnings_json TEXT NOT NULL DEFAULT '[]';

ALTER TABLE attempt_mapping_history ADD COLUMN change_kind TEXT NOT NULL DEFAULT 'prospective';
ALTER TABLE attempt_mapping_history ADD COLUMN valid_from TEXT;
ALTER TABLE attempt_mapping_history ADD COLUMN valid_until TEXT;
ALTER TABLE attempt_mapping_history ADD COLUMN warnings_json TEXT NOT NULL DEFAULT '[]';
ALTER TABLE attempt_mapping_history ADD COLUMN provenance_json TEXT NOT NULL DEFAULT '{}';

CREATE INDEX IF NOT EXISTS idx_model_mapping_validity
  ON model_mapping_versions(review_status, change_kind, valid_from, valid_until);
