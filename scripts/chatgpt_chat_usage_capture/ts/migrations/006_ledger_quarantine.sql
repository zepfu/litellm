ALTER TABLE observations
  ADD COLUMN quarantine_json TEXT NOT NULL
  DEFAULT '{"state":"clear","warnings":[],"timestamps":[]}';

ALTER TABLE conversation_state
  ADD COLUMN quarantine_json TEXT NOT NULL
  DEFAULT '{"state":"clear","warnings":[],"timestamps":[]}';

ALTER TABLE message_records
  ADD COLUMN quarantine_json TEXT NOT NULL
  DEFAULT '{"state":"clear","warnings":[],"timestamps":[]}';

ALTER TABLE message_revisions
  ADD COLUMN quarantine_json TEXT NOT NULL
  DEFAULT '{"state":"clear","warnings":[],"timestamps":[]}';

ALTER TABLE attempts
  ADD COLUMN quarantine_json TEXT NOT NULL
  DEFAULT '{"state":"clear","warnings":[],"timestamps":[]}';

ALTER TABLE attempt_revisions
  ADD COLUMN quarantine_json TEXT NOT NULL
  DEFAULT '{"state":"clear","warnings":[],"timestamps":[]}';
