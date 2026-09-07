ALTER TABLE observations ADD COLUMN supersedes_observation_id TEXT;

CREATE TABLE IF NOT EXISTS message_revisions (
  revision_id TEXT PRIMARY KEY,
  scope_key TEXT NOT NULL,
  collector_account_id TEXT NOT NULL,
  provider TEXT NOT NULL,
  provider_user_id TEXT,
  workspace_id TEXT,
  quota_owner_id TEXT,
  conversation_id TEXT NOT NULL,
  message_id TEXT NOT NULL,
  revision INTEGER NOT NULL,
  revision_fingerprint TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  observed_at TEXT NOT NULL,
  run_id TEXT NOT NULL,
  UNIQUE (scope_key, conversation_id, message_id, revision_fingerprint)
);

CREATE TABLE IF NOT EXISTS attempt_revisions (
  attempt_id TEXT NOT NULL,
  revision INTEGER NOT NULL,
  scope_key TEXT NOT NULL,
  projection_fingerprint TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  recorded_at TEXT NOT NULL,
  source TEXT NOT NULL,
  PRIMARY KEY (attempt_id, revision),
  UNIQUE (attempt_id, projection_fingerprint),
  FOREIGN KEY (attempt_id) REFERENCES attempts(attempt_id)
);

CREATE TABLE IF NOT EXISTS attempt_mapping_history (
  history_id TEXT PRIMARY KEY,
  attempt_id TEXT NOT NULL,
  scope_key TEXT NOT NULL,
  mapping_version TEXT NOT NULL,
  requested_family TEXT,
  recorded_final_family TEXT,
  resolved_family TEXT,
  recorded_at TEXT NOT NULL,
  source TEXT NOT NULL,
  UNIQUE (attempt_id, mapping_version),
  FOREIGN KEY (attempt_id) REFERENCES attempts(attempt_id)
);

CREATE INDEX IF NOT EXISTS idx_message_revisions_lookup
  ON message_revisions(scope_key, conversation_id, message_id, revision);
CREATE INDEX IF NOT EXISTS idx_attempt_revisions_lookup
  ON attempt_revisions(scope_key, attempt_id, revision);
CREATE INDEX IF NOT EXISTS idx_mapping_history_lookup
  ON attempt_mapping_history(scope_key, attempt_id, recorded_at);
