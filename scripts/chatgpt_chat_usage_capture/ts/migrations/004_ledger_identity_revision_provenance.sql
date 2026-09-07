CREATE TABLE observations_v4 (
  observation_id TEXT PRIMARY KEY,
  scope_key TEXT NOT NULL,
  collector_account_id TEXT NOT NULL,
  provider TEXT NOT NULL,
  provider_user_id TEXT,
  workspace_id TEXT,
  quota_owner_id TEXT,
  source_kind TEXT NOT NULL,
  source_id TEXT NOT NULL,
  revision_fingerprint TEXT NOT NULL,
  revision_number INTEGER NOT NULL,
  surface TEXT NOT NULL,
  conversation_id TEXT,
  payload_json TEXT NOT NULL,
  observed_at TEXT NOT NULL,
  run_id TEXT NOT NULL,
  schema_version TEXT NOT NULL,
  provenance_json TEXT NOT NULL,
  supersedes_observation_id TEXT
);

INSERT INTO observations_v4(
  observation_id, scope_key, collector_account_id, provider,
  provider_user_id, workspace_id, quota_owner_id, source_kind, source_id,
  revision_fingerprint, revision_number, surface, conversation_id,
  payload_json, observed_at, run_id, schema_version, provenance_json,
  supersedes_observation_id
)
SELECT observation_id, scope_key, collector_account_id, provider,
       provider_user_id, workspace_id, quota_owner_id, source_kind, source_id,
       revision_fingerprint, revision_number, surface, conversation_id,
       payload_json, observed_at, run_id, schema_version, provenance_json,
       supersedes_observation_id
FROM observations;

DROP TABLE observations;
ALTER TABLE observations_v4 RENAME TO observations;

CREATE INDEX IF NOT EXISTS idx_observations_source
  ON observations(scope_key, source_kind, source_id, revision_number);

CREATE TABLE message_revisions_v4 (
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
  run_id TEXT NOT NULL
);

INSERT INTO message_revisions_v4(
  revision_id, scope_key, collector_account_id, provider,
  provider_user_id, workspace_id, quota_owner_id, conversation_id,
  message_id, revision, revision_fingerprint, payload_json, observed_at, run_id
)
SELECT revision_id, scope_key, collector_account_id, provider,
       provider_user_id, workspace_id, quota_owner_id, conversation_id,
       message_id, revision, revision_fingerprint, payload_json, observed_at, run_id
FROM message_revisions;

DROP TABLE message_revisions;
ALTER TABLE message_revisions_v4 RENAME TO message_revisions;

CREATE INDEX IF NOT EXISTS idx_message_revisions_lookup
  ON message_revisions(scope_key, conversation_id, message_id, revision);

CREATE TABLE attempt_revisions_v4 (
  attempt_id TEXT NOT NULL,
  revision INTEGER NOT NULL,
  scope_key TEXT NOT NULL,
  collector_account_id TEXT NOT NULL,
  projection_fingerprint TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  recorded_at TEXT NOT NULL,
  source TEXT NOT NULL,
  PRIMARY KEY (attempt_id, revision),
  FOREIGN KEY (attempt_id) REFERENCES attempts(attempt_id)
);

INSERT INTO attempt_revisions_v4(
  attempt_id, revision, scope_key, collector_account_id,
  projection_fingerprint, payload_json, recorded_at, source
)
SELECT revisions.attempt_id, revisions.revision, revisions.scope_key,
       COALESCE(attempts.collector_account_id, 'unknown-collector'),
       revisions.projection_fingerprint, revisions.payload_json,
       revisions.recorded_at, revisions.source
FROM attempt_revisions AS revisions
LEFT JOIN attempts
  ON attempts.attempt_id = revisions.attempt_id;

DROP TABLE attempt_revisions;
ALTER TABLE attempt_revisions_v4 RENAME TO attempt_revisions;

CREATE INDEX IF NOT EXISTS idx_attempt_revisions_lookup
  ON attempt_revisions(scope_key, attempt_id, revision);

CREATE TABLE IF NOT EXISTS activity_provenance (
  scope_key TEXT NOT NULL,
  activity_kind TEXT NOT NULL,
  activity_id TEXT NOT NULL,
  collector_account_id TEXT NOT NULL,
  first_seen_at TEXT NOT NULL,
  last_seen_at TEXT NOT NULL,
  PRIMARY KEY (scope_key, activity_kind, activity_id, collector_account_id)
);

CREATE INDEX IF NOT EXISTS idx_activity_provenance_lookup
  ON activity_provenance(scope_key, activity_kind, activity_id, last_seen_at);

INSERT OR IGNORE INTO activity_provenance(
  scope_key, activity_kind, activity_id, collector_account_id,
  first_seen_at, last_seen_at
)
SELECT scope_key, 'observation', observation_id, collector_account_id,
       observed_at, observed_at
FROM observations;

INSERT OR IGNORE INTO activity_provenance(
  scope_key, activity_kind, activity_id, collector_account_id,
  first_seen_at, last_seen_at
)
SELECT scope_key, 'message',
       conversation_id || ':' || message_id,
       collector_account_id,
       updated_at, updated_at
FROM message_records;

INSERT OR IGNORE INTO activity_provenance(
  scope_key, activity_kind, activity_id, collector_account_id,
  first_seen_at, last_seen_at
)
SELECT scope_key, 'attempt', attempt_id, collector_account_id,
       updated_at, updated_at
FROM attempts;
