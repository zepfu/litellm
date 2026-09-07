CREATE TABLE IF NOT EXISTS accounts (
  collector_account_id TEXT PRIMARY KEY,
  scope_key TEXT NOT NULL UNIQUE,
  provider TEXT NOT NULL,
  provider_user_id TEXT,
  workspace_id TEXT,
  quota_owner_id TEXT,
  surface TEXT NOT NULL,
  auth_state TEXT NOT NULL,
  plan_policy_id TEXT,
  enabled INTEGER NOT NULL DEFAULT 1 CHECK (enabled IN (0, 1)),
  profile_path TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS collector_runs (
  run_id TEXT PRIMARY KEY,
  scope_key TEXT NOT NULL,
  collector_account_id TEXT NOT NULL,
  mode TEXT NOT NULL,
  started_at TEXT NOT NULL,
  ended_at TEXT,
  result TEXT,
  details_json TEXT NOT NULL DEFAULT '{}',
  FOREIGN KEY (collector_account_id) REFERENCES accounts(collector_account_id)
);

CREATE TABLE IF NOT EXISTS conversation_state (
  scope_key TEXT NOT NULL,
  collector_account_id TEXT NOT NULL,
  provider TEXT NOT NULL,
  provider_user_id TEXT,
  workspace_id TEXT,
  quota_owner_id TEXT,
  conversation_id TEXT NOT NULL,
  created_at TEXT,
  updated_at TEXT,
  is_archived INTEGER NOT NULL DEFAULT 0 CHECK (is_archived IN (0, 1)),
  surface TEXT NOT NULL,
  origin TEXT,
  current_node TEXT,
  page_coverage TEXT NOT NULL,
  warnings_json TEXT NOT NULL DEFAULT '[]',
  last_seen_run_id TEXT,
  PRIMARY KEY (scope_key, conversation_id),
  FOREIGN KEY (collector_account_id) REFERENCES accounts(collector_account_id)
);

CREATE TABLE IF NOT EXISTS observations (
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
  UNIQUE (scope_key, source_kind, source_id, revision_fingerprint)
);

CREATE TABLE IF NOT EXISTS message_records (
  scope_key TEXT NOT NULL,
  collector_account_id TEXT NOT NULL,
  provider TEXT NOT NULL,
  provider_user_id TEXT,
  workspace_id TEXT,
  quota_owner_id TEXT,
  conversation_id TEXT NOT NULL,
  message_id TEXT NOT NULL,
  node_id TEXT,
  parent_id TEXT,
  children_json TEXT NOT NULL DEFAULT '[]',
  role TEXT,
  channel TEXT,
  created_at TEXT,
  status TEXT,
  end_turn INTEGER CHECK (end_turn IN (0, 1)),
  requested_model_raw TEXT,
  requested_mode_raw TEXT,
  requested_reasoning_effort_raw TEXT,
  recorded_final_model_raw TEXT,
  generation_id TEXT,
  request_id TEXT,
  surface TEXT NOT NULL,
  origin TEXT,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  revision INTEGER NOT NULL DEFAULT 1,
  revision_fingerprint TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (scope_key, conversation_id, message_id)
);

CREATE TABLE IF NOT EXISTS attempts (
  attempt_id TEXT PRIMARY KEY,
  scope_key TEXT NOT NULL,
  collector_account_id TEXT NOT NULL,
  provider TEXT NOT NULL,
  provider_user_id TEXT,
  workspace_id TEXT,
  quota_owner_id TEXT,
  conversation_id TEXT NOT NULL,
  identity_basis TEXT NOT NULL,
  time_basis TEXT NOT NULL,
  attempt_time TEXT,
  earliest_possible_at TEXT,
  latest_possible_at TEXT,
  requested_model_raw TEXT,
  requested_mode_raw TEXT,
  requested_reasoning_effort_raw TEXT,
  recorded_final_model_raw TEXT,
  resolved_model_raw TEXT,
  requested_family TEXT,
  recorded_final_family TEXT,
  resolved_family TEXT,
  mapping_version TEXT NOT NULL,
  outcome TEXT NOT NULL,
  completed_answer INTEGER NOT NULL CHECK (completed_answer IN (0, 1)),
  generation_started INTEGER NOT NULL CHECK (generation_started IN (0, 1)),
  surface TEXT NOT NULL,
  origin TEXT,
  revision INTEGER NOT NULL DEFAULT 1,
  projection_fingerprint TEXT NOT NULL,
  warnings_json TEXT NOT NULL DEFAULT '[]',
  tombstone INTEGER NOT NULL DEFAULT 0 CHECK (tombstone IN (0, 1)),
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS attempt_aliases (
  scope_key TEXT NOT NULL,
  collector_account_id TEXT NOT NULL,
  provider TEXT NOT NULL,
  provider_user_id TEXT,
  workspace_id TEXT,
  quota_owner_id TEXT,
  alias_kind TEXT NOT NULL,
  alias_value TEXT NOT NULL,
  attempt_id TEXT NOT NULL,
  ambiguous INTEGER NOT NULL DEFAULT 0 CHECK (ambiguous IN (0, 1)),
  first_seen_at TEXT NOT NULL,
  last_seen_at TEXT NOT NULL,
  PRIMARY KEY (scope_key, alias_kind, alias_value),
  FOREIGN KEY (attempt_id) REFERENCES attempts(attempt_id)
);

CREATE TABLE IF NOT EXISTS attempt_evidence (
  attempt_id TEXT NOT NULL,
  evidence_kind TEXT NOT NULL,
  evidence_id TEXT NOT NULL,
  scope_key TEXT NOT NULL,
  PRIMARY KEY (attempt_id, evidence_kind, evidence_id),
  FOREIGN KEY (attempt_id) REFERENCES attempts(attempt_id)
);

CREATE TABLE IF NOT EXISTS model_mapping_versions (
  version TEXT PRIMARY KEY,
  canonical_families_json TEXT NOT NULL,
  rules_json TEXT NOT NULL,
  review_status TEXT NOT NULL,
  source TEXT NOT NULL,
  created_at TEXT NOT NULL,
  reviewed_at TEXT,
  reviewed_by TEXT
);

CREATE TABLE IF NOT EXISTS coverage_gaps (
  gap_id TEXT PRIMARY KEY,
  scope_key TEXT NOT NULL,
  collector_account_id TEXT NOT NULL,
  provider TEXT NOT NULL,
  provider_user_id TEXT,
  workspace_id TEXT,
  quota_owner_id TEXT,
  source_kind TEXT NOT NULL,
  source_id TEXT NOT NULL,
  reason TEXT NOT NULL,
  state TEXT NOT NULL,
  first_seen_at TEXT NOT NULL,
  last_seen_at TEXT NOT NULL,
  details_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS aggregate_revisions (
  revision_id TEXT PRIMARY KEY,
  scope_key TEXT NOT NULL,
  collector_account_id TEXT NOT NULL,
  mapping_version TEXT NOT NULL,
  input_fingerprint TEXT NOT NULL,
  evaluated_at TEXT NOT NULL,
  created_at TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  UNIQUE (scope_key, mapping_version, input_fingerprint, evaluated_at)
);

CREATE TABLE IF NOT EXISTS schema_migrations (
  version INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  checksum TEXT NOT NULL,
  applied_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_observations_source
  ON observations(scope_key, source_kind, source_id, revision_number);
CREATE INDEX IF NOT EXISTS idx_messages_conversation
  ON message_records(scope_key, conversation_id, created_at, message_id);
CREATE INDEX IF NOT EXISTS idx_attempts_scope_time
  ON attempts(scope_key, attempt_time, earliest_possible_at, attempt_id);
CREATE INDEX IF NOT EXISTS idx_attempts_raw_models
  ON attempts(scope_key, requested_model_raw, recorded_final_model_raw, resolved_model_raw);
CREATE INDEX IF NOT EXISTS idx_coverage_gaps_scope
  ON coverage_gaps(scope_key, state, last_seen_at);
