CREATE TABLE IF NOT EXISTS history_state (
  scope_key TEXT PRIMARY KEY,
  collector_account_id TEXT NOT NULL,
  state_json TEXT NOT NULL,
  FOREIGN KEY (collector_account_id) REFERENCES accounts(collector_account_id)
);
