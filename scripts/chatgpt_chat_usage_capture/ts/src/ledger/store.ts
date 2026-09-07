import Database from "better-sqlite3";
import {
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
} from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import type {
  ConversationDetailProjection,
  ConversationSummary,
  MessageRecord,
} from "../contracts/records.js";
import { reconstructAttempts } from "../normalize/reconstruct.js";
import {
  mapModelEvidence,
  suggestModelMappings,
  validateMappingVersion,
} from "../normalize/model-mapping.js";
import {
  assertNoSecrets,
  observationProjection,
  sanitizeMetadata,
  sanitizeToken,
} from "../security/sanitizer.js";
import { canonicalJson, fingerprint, scopeKey, stableId } from "./identity.js";
import type {
  IngestContext,
  IngestResult,
  LedgerScope,
  ModelMappingVersion,
  ReconstructedAttempt,
  StoredMessage,
} from "./types.js";

type SqlRow = Record<string, unknown>;
type SqliteDatabase = InstanceType<typeof Database>;

export interface AccountRow extends LedgerScope {
  authState: string;
  planPolicyId: string | null;
  enabled: boolean;
  profilePath: string | null;
}

export interface AggregateRevision {
  revisionId: string;
  mappingVersion: string;
  inputFingerprint: string;
  evaluatedAt: string;
  createdAt: string;
  payload: Record<string, unknown>;
}

export class LedgerError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "LedgerError";
  }
}

export class AliasCollisionError extends LedgerError {
  constructor(message: string) {
    super(message);
    this.name = "AliasCollisionError";
  }
}

export class Ledger {
  readonly db: SqliteDatabase;
  readonly path: string;

  private savepointCounter = 0;

  constructor(path: string) {
    this.path = path;
    if (path !== ":memory:") {
      mkdirSync(dirname(resolve(path)), { recursive: true, mode: 0o700 });
    }
    this.db = new Database(path);
    this.db.pragma("journal_mode = WAL");
    this.db.pragma("synchronous = NORMAL");
    this.db.pragma("foreign_keys = ON");
    this.db.pragma("busy_timeout = 5000");
    this.migrate();
  }

  close(): void {
    this.db.close();
  }

  get schemaVersion(): number {
    const row = this.db
      .prepare("SELECT COALESCE(MAX(version), 0) AS version FROM schema_migrations")
      .get() as SqlRow | undefined;
    return Number(row?.version ?? 0);
  }

  transaction<T>(callback: () => T): T {
    if (!this.db.inTransaction) {
      this.db.exec("BEGIN IMMEDIATE");
      try {
        const result = callback();
        this.db.exec("COMMIT");
        return result;
      } catch (error) {
        this.db.exec("ROLLBACK");
        throw error;
      }
    }

    const savepoint = `stage2b_sp_${this.savepointCounter++}`;
    this.db.exec(`SAVEPOINT ${savepoint}`);
    try {
      const result = callback();
      this.db.exec(`RELEASE ${savepoint}`);
      return result;
    } catch (error) {
      this.db.exec(`ROLLBACK TO ${savepoint}`);
      this.db.exec(`RELEASE ${savepoint}`);
      throw error;
    }
  }

  migrate(): void {
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS schema_migrations (
        version INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        checksum TEXT NOT NULL,
        applied_at TEXT NOT NULL
      )
    `);
    const migrations = loadMigrations();
    const applied = new Set(
      this.db
        .prepare("SELECT version FROM schema_migrations ORDER BY version")
        .all()
        .map((row: unknown) => Number((row as SqlRow).version)),
    );
    for (const migration of migrations) {
      if (applied.has(migration.version)) {
        continue;
      }
      this.transaction(() => {
        this.db.exec(migration.sql);
        this.db
          .prepare(
            "INSERT INTO schema_migrations(version, name, checksum, applied_at) VALUES (?, ?, ?, ?)",
          )
          .run(
            migration.version,
            migration.name,
            fingerprint(migration.sql),
            new Date().toISOString(),
          );
      });
    }
  }

  upsertAccount(
    scope: LedgerScope,
    options: {
      authState?: string;
      planPolicyId?: string | null;
      enabled?: boolean;
      profilePath?: string | null;
    } = {},
  ): void {
    const key = scopeKey(scope);
    const now = new Date().toISOString();
    this.db
      .prepare(
        `
        INSERT INTO accounts (
          collector_account_id, scope_key, provider, provider_user_id,
          workspace_id, quota_owner_id, surface, auth_state, plan_policy_id,
          enabled, profile_path, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(collector_account_id) DO UPDATE SET
          scope_key=excluded.scope_key,
          provider=excluded.provider,
          provider_user_id=excluded.provider_user_id,
          workspace_id=excluded.workspace_id,
          quota_owner_id=excluded.quota_owner_id,
          surface=excluded.surface,
          auth_state=excluded.auth_state,
          plan_policy_id=excluded.plan_policy_id,
          enabled=excluded.enabled,
          profile_path=excluded.profile_path,
          updated_at=excluded.updated_at
        `,
      )
      .run(
        scope.collectorAccountId,
        key,
        scope.provider,
        scope.providerUserId,
        scope.workspaceId,
        scope.quotaOwnerId,
        scope.surface,
        options.authState ?? "ready",
        options.planPolicyId ?? null,
        options.enabled === false ? 0 : 1,
        options.profilePath ?? null,
        now,
        now,
      );
  }

  accountScope(collectorAccountId: string): LedgerScope {
    const row = this.db
      .prepare("SELECT * FROM accounts WHERE collector_account_id=?")
      .get(collectorAccountId) as SqlRow | undefined;
    if (!row) {
      throw new LedgerError(`account not found: ${collectorAccountId}`);
    }
    return scopeFromRow(row);
  }

  listAccounts(): AccountRow[] {
    return this.db
      .prepare("SELECT * FROM accounts ORDER BY collector_account_id")
      .all()
      .map((row: unknown) => {
        const item = row as SqlRow;
        return {
          ...scopeFromRow(item),
          authState: String(item.auth_state),
          planPolicyId: nullableString(item.plan_policy_id),
          enabled: Number(item.enabled) === 1,
          profilePath: nullableString(item.profile_path),
        };
      });
  }

  startRun(
    scope: LedgerScope,
    run: { runId: string; mode: string; startedAt: string; details?: Record<string, unknown> },
  ): void {
    this.db
      .prepare(
        `
        INSERT INTO collector_runs(
          run_id, scope_key, collector_account_id, mode, started_at, details_json
        ) VALUES (?, ?, ?, ?, ?, ?)
        `,
      )
      .run(
        run.runId,
        scopeKey(scope),
        scope.collectorAccountId,
        run.mode,
        run.startedAt,
        JSON.stringify(run.details ?? {}),
      );
  }

  finishRun(
    runId: string,
    result: string,
    endedAt: string,
    details: Record<string, unknown> = {},
  ): void {
    this.db
      .prepare(
        "UPDATE collector_runs SET ended_at=?, result=?, details_json=? WHERE run_id=?",
      )
      .run(endedAt, result, JSON.stringify(details), runId);
  }

  insertObservation(
    scope: LedgerScope,
    context: IngestContext,
    payload: Record<string, unknown>,
  ): { observationId: string; inserted: boolean } {
    const sanitized = observationProjection(payload, {
      sourceKind: context.sourceKind,
      runId: context.runId,
      evidenceId: context.sourceId,
    });
    assertNoSecrets(sanitized);
    const revisionFingerprint = fingerprint(stableObservationPayload(sanitized));
    const key = scopeKey(scope);
    const existing = this.db
      .prepare(
        `
        SELECT observation_id
        FROM observations
        WHERE scope_key=? AND source_kind=? AND source_id=? AND revision_fingerprint=?
        `,
      )
      .get(key, context.sourceKind, context.sourceId, revisionFingerprint) as
      | SqlRow
      | undefined;
    if (existing) {
      return { observationId: String(existing.observation_id), inserted: false };
    }
    const previous = this.db
      .prepare(
        `
        SELECT observation_id, revision_number
        FROM observations
        WHERE scope_key=? AND source_kind=? AND source_id=?
        ORDER BY revision_number DESC
        LIMIT 1
        `,
      )
      .get(key, context.sourceKind, context.sourceId) as SqlRow | undefined;
    const revisionNumber = Number(previous?.revision_number ?? 0) + 1;
    const observationId = stableId(
      key,
      context.sourceKind,
      context.sourceId,
      revisionFingerprint,
    );
    this.db
      .prepare(
        `
        INSERT INTO observations(
          observation_id, scope_key, collector_account_id, provider,
          provider_user_id, workspace_id, quota_owner_id, source_kind, source_id,
          revision_fingerprint, revision_number, surface, conversation_id,
          payload_json, observed_at, run_id, schema_version, provenance_json,
          supersedes_observation_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        `,
      )
      .run(
        observationId,
        key,
        scope.collectorAccountId,
        scope.provider,
        scope.providerUserId,
        scope.workspaceId,
        scope.quotaOwnerId,
        context.sourceKind,
        context.sourceId,
        revisionFingerprint,
        revisionNumber,
        scope.surface,
        typeof payload.conversationId === "string"
          ? payload.conversationId
          : typeof payload.conversation_id === "string"
            ? payload.conversation_id
            : null,
        JSON.stringify(sanitized),
        context.observedAt,
        context.runId,
        context.schemaVersion,
        JSON.stringify(context.provenance ?? {}),
        previous?.observation_id ?? null,
      );
    return { observationId, inserted: true };
  }

  ingestConversation(
    scope: LedgerScope,
    detail: ConversationDetailProjection,
    mapping: ModelMappingVersion,
    context: IngestContext,
    summary?: ConversationSummary,
  ): IngestResult {
    validateMappingVersion(mapping);
    return this.transaction(() => {
      const observation = this.insertObservation(
        scope,
        context,
        {
          conversation_id: detail.conversationId,
          created_at: detail.createdAt,
          updated_at: detail.updatedAt,
          current_node: detail.currentNode,
          surface: detail.surface,
          messages: detail.messages.map((message) =>
            messagePayload(sanitizeMessage(message)),
          ),
          coverage: detail.coverage,
          warnings: detail.warnings,
        },
      );
      if (summary) {
        this.upsertConversation(scope, summary, context.runId);
      } else {
        this.upsertConversation(
          scope,
          {
            conversationId: detail.conversationId,
            createdAt: detail.createdAt,
            updatedAt: detail.updatedAt,
            isArchived: false,
            workspaceId: scope.workspaceId,
            projectId: null,
            surface: detail.surface,
            origin: null,
            hasVersions: null,
            currentNode: detail.currentNode,
            coverage: detail.coverage,
          },
          context.runId,
        );
      }

      let messageInserted = 0;
      let messageDeduplicated = 0;
      for (const message of detail.messages) {
        const result = this.upsertMessage(scope, message, context);
        if (result.inserted) {
          messageInserted += 1;
        } else {
          messageDeduplicated += 1;
        }
      }

      const conversationOrigin =
        summary?.origin ?? this.conversationOrigin(scope, detail.conversationId);
      const messages = this.messagesFor(scope, detail.conversationId).map((message) =>
        conversationOrigin !== null && message.origin === null
          ? { ...message, origin: conversationOrigin }
          : message,
      );
      const attempts = reconstructAttempts(messages, {
        scope,
        conversationId: detail.conversationId,
        mapping,
      });
      let attemptInserted = 0;
      let attemptUpdated = 0;
      let attemptDeduplicated = 0;
      let aliasConflicts = 0;
      for (const attempt of attempts) {
        const result = this.upsertAttempt(scope, attempt, context);
        attemptInserted += result.status === "inserted" ? 1 : 0;
        attemptUpdated += result.status === "updated" ? 1 : 0;
        attemptDeduplicated += result.status === "deduplicated" ? 1 : 0;
        aliasConflicts += result.aliasConflicts;
        this.attachEvidence(
          attempt.attemptId,
          "observation",
          observation.observationId,
          scope,
        );
      }

      const warnings = [...detail.warnings];
      if (detail.coverage !== "validated_page" || warnings.length > 0) {
        this.recordCoverageGap(
          scope,
          {
            sourceKind: context.sourceKind,
            sourceId: context.sourceId,
            reason: detail.coverage === "unrecognized"
              ? "unrecognized_conversation_detail"
              : "partial_conversation_detail",
            details: { warnings, coverage: detail.coverage },
          },
          context.observedAt,
        );
      }
      if (aliasConflicts > 0) {
        warnings.push(`alias_conflicts:${aliasConflicts}`);
      }
      return {
        observationId: observation.observationId,
        observationInserted: observation.inserted,
        messageInserted,
        messageDeduplicated,
        attemptInserted,
        attemptUpdated,
        attemptDeduplicated,
        aliasConflicts,
        coverage: detail.coverage,
        warnings,
      };
    });
  }

  upsertConversation(
    scope: LedgerScope,
    summary: ConversationSummary,
    runId: string,
  ): void {
    const key = scopeKey(scope);
    this.db
      .prepare(
        `
        INSERT INTO conversation_state(
          scope_key, collector_account_id, provider, provider_user_id,
          workspace_id, quota_owner_id, conversation_id, created_at, updated_at,
          is_archived, surface, origin, current_node, page_coverage,
          warnings_json, last_seen_run_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(scope_key, conversation_id) DO UPDATE SET
          created_at=COALESCE(excluded.created_at, conversation_state.created_at),
          updated_at=COALESCE(excluded.updated_at, conversation_state.updated_at),
          is_archived=excluded.is_archived,
          surface=excluded.surface,
          origin=COALESCE(excluded.origin, conversation_state.origin),
          current_node=COALESCE(excluded.current_node, conversation_state.current_node),
          page_coverage=excluded.page_coverage,
          warnings_json=excluded.warnings_json,
          last_seen_run_id=excluded.last_seen_run_id
        `,
      )
      .run(
        key,
        scope.collectorAccountId,
        scope.provider,
        scope.providerUserId,
        scope.workspaceId,
        scope.quotaOwnerId,
        summary.conversationId,
        summary.createdAt,
        summary.updatedAt,
        summary.isArchived ? 1 : 0,
        summary.surface,
        summary.origin,
        summary.currentNode,
        summary.coverage,
        JSON.stringify([]),
        runId,
      );
  }

  upsertMessage(
    scope: LedgerScope,
    record: MessageRecord,
    context: IngestContext,
  ): { inserted: boolean; revision: number; revisionFingerprint: string } {
    const clean = sanitizeMessage(record);
    assertNoSecrets(clean);
    const key = scopeKey(scope);
    const payload = messagePayload(clean);
    const revisionFingerprint = fingerprint(payload);
    const current = this.db
      .prepare(
        `
        SELECT revision, revision_fingerprint
        FROM message_records
        WHERE scope_key=? AND conversation_id=? AND message_id=?
        `,
      )
      .get(key, clean.conversationId, clean.messageId) as SqlRow | undefined;
    if (current && current.revision_fingerprint === revisionFingerprint) {
      return {
        inserted: false,
        revision: Number(current.revision),
        revisionFingerprint,
      };
    }
    const revision = Number(current?.revision ?? 0) + 1;
    const revisionId = stableId(
      key,
      clean.conversationId,
      clean.messageId,
      revisionFingerprint,
    );
    this.db
      .prepare(
        `
        INSERT OR IGNORE INTO message_revisions(
          revision_id, scope_key, collector_account_id, provider,
          provider_user_id, workspace_id, quota_owner_id, conversation_id,
          message_id, revision, revision_fingerprint, payload_json,
          observed_at, run_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        `,
      )
      .run(
        revisionId,
        key,
        scope.collectorAccountId,
        scope.provider,
        scope.providerUserId,
        scope.workspaceId,
        scope.quotaOwnerId,
        clean.conversationId,
        clean.messageId,
        revision,
        revisionFingerprint,
        JSON.stringify(payload),
        context.observedAt,
        context.runId,
      );
    this.db
      .prepare(
        `
        INSERT INTO message_records(
          scope_key, collector_account_id, provider, provider_user_id,
          workspace_id, quota_owner_id, conversation_id, message_id, node_id,
          parent_id, children_json, role, channel, created_at, status, end_turn,
          requested_model_raw, requested_mode_raw, requested_reasoning_effort_raw,
          recorded_final_model_raw, generation_id, request_id, surface, origin,
          metadata_json, revision, revision_fingerprint, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(scope_key, conversation_id, message_id) DO UPDATE SET
          node_id=excluded.node_id,
          parent_id=excluded.parent_id,
          children_json=excluded.children_json,
          role=excluded.role,
          channel=excluded.channel,
          created_at=excluded.created_at,
          status=excluded.status,
          end_turn=excluded.end_turn,
          requested_model_raw=excluded.requested_model_raw,
          requested_mode_raw=excluded.requested_mode_raw,
          requested_reasoning_effort_raw=excluded.requested_reasoning_effort_raw,
          recorded_final_model_raw=excluded.recorded_final_model_raw,
          generation_id=excluded.generation_id,
          request_id=excluded.request_id,
          surface=excluded.surface,
          origin=excluded.origin,
          metadata_json=excluded.metadata_json,
          revision=excluded.revision,
          revision_fingerprint=excluded.revision_fingerprint,
          updated_at=excluded.updated_at
        `,
      )
      .run(
        key,
        scope.collectorAccountId,
        scope.provider,
        scope.providerUserId,
        scope.workspaceId,
        scope.quotaOwnerId,
        clean.conversationId,
        clean.messageId,
        clean.nodeId,
        clean.parentId,
        JSON.stringify(clean.children),
        clean.role,
        clean.channel,
        clean.createdAt,
        clean.status,
        clean.endTurn === null ? null : clean.endTurn ? 1 : 0,
        clean.requestedModelRaw,
        clean.requestedModeRaw,
        clean.requestedReasoningEffortRaw,
        clean.recordedFinalModelRaw,
        clean.generationId,
        clean.requestId,
        clean.surface,
        clean.origin,
        JSON.stringify(clean.metadata),
        revision,
        revisionFingerprint,
        context.observedAt,
      );
    return { inserted: true, revision, revisionFingerprint };
  }

  messagesFor(scope: LedgerScope, conversationId: string): MessageRecord[] {
    const rows = this.db
      .prepare(
        `
        SELECT *
        FROM message_records
        WHERE scope_key=? AND conversation_id=?
        ORDER BY COALESCE(created_at, ''), message_id
        `,
      )
      .all(scopeKey(scope), conversationId) as SqlRow[];
    return rows.map(messageFromRow);
  }

  messageRevisions(
    scope: LedgerScope,
    conversationId: string,
    messageId: string,
  ): Array<Record<string, unknown>> {
    return this.db
      .prepare(
        `
        SELECT *
        FROM message_revisions
        WHERE scope_key=? AND conversation_id=? AND message_id=?
        ORDER BY revision
        `,
      )
      .all(scopeKey(scope), conversationId, messageId)
      .map((row: unknown) => {
        const item = row as SqlRow;
        return { ...item, payload: parseJsonObject(item.payload_json) };
      });
  }

  upsertAttempt(
    scope: LedgerScope,
    attempt: ReconstructedAttempt,
    context: IngestContext,
  ): { status: "inserted" | "updated" | "deduplicated"; aliasConflicts: number } {
    const key = scopeKey(scope);
    const payload = attemptPayload(attempt);
    const projectionFingerprint = fingerprint(payload);
    const current = this.db
      .prepare("SELECT * FROM attempts WHERE attempt_id=? AND scope_key=?")
      .get(attempt.attemptId, key) as SqlRow | undefined;
    let status: "inserted" | "updated" | "deduplicated";
    let revision: number;
    if (!current) {
      revision = 1;
      this.insertAttemptRow(scope, attempt, revision, projectionFingerprint, context.observedAt);
      this.insertAttemptRevision(scope, attempt, revision, projectionFingerprint, payload, context, "ingest");
      status = "inserted";
    } else if (current.projection_fingerprint === projectionFingerprint) {
      revision = Number(current.revision);
      status = "deduplicated";
    } else {
      revision = Number(current.revision) + 1;
      const priorAttempt = rowToReconstructedAttempt(
        rowToAttemptPayload(current),
        scope,
      );
      this.insertAttemptRevision(
        scope,
        priorAttempt,
        Number(current.revision),
        String(current.projection_fingerprint),
        attemptPayload(priorAttempt),
        context,
        "prior_projection",
      );
      this.updateAttemptRow(scope, attempt, revision, projectionFingerprint, context.observedAt);
      this.insertAttemptRevision(scope, attempt, revision, projectionFingerprint, payload, context, "ingest");
      status = "updated";
    }
    let aliasConflicts = 0;
    for (const [kind, value] of attempt.aliases) {
      if (!value) {
        continue;
      }
      const existing = this.db
        .prepare(
          `
          SELECT attempt_id
          FROM attempt_aliases
          WHERE scope_key=? AND alias_kind=? AND alias_value=?
          `,
        )
        .get(key, kind, value) as SqlRow | undefined;
      if (existing && existing.attempt_id !== attempt.attemptId) {
        aliasConflicts += 1;
        this.recordCoverageGap(
          scope,
          {
            sourceKind: "attempt_alias",
            sourceId: `${kind}:${value}`,
            reason: "alias_collision",
            details: {
              existingAttemptId: existing.attempt_id,
              incomingAttemptId: attempt.attemptId,
              aliasKind: kind,
            },
          },
          context.observedAt,
        );
        continue;
      }
      this.db
        .prepare(
          `
          INSERT INTO attempt_aliases(
            scope_key, collector_account_id, provider, provider_user_id,
            workspace_id, quota_owner_id, alias_kind, alias_value, attempt_id,
            ambiguous, first_seen_at, last_seen_at
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
          ON CONFLICT(scope_key, alias_kind, alias_value) DO UPDATE SET
            last_seen_at=excluded.last_seen_at
          `,
        )
        .run(
          key,
          scope.collectorAccountId,
          scope.provider,
          scope.providerUserId,
          scope.workspaceId,
          scope.quotaOwnerId,
          kind,
          value,
          attempt.attemptId,
          context.observedAt,
          context.observedAt,
        );
    }
    for (const messageId of attempt.evidenceMessageIds) {
      this.attachEvidence(attempt.attemptId, "message", messageId, scope);
    }
    if (status === "inserted" || status === "updated") {
      this.recordMappingHistory(scope, attempt, context.observedAt, "ingest");
    }
    return { status, aliasConflicts };
  }

  listAttempts(scope: LedgerScope, includeTombstones = false): Array<Record<string, unknown>> {
    const sql = `
      SELECT *
      FROM attempts
      WHERE scope_key=? ${includeTombstones ? "" : "AND tombstone=0"}
      ORDER BY COALESCE(attempt_time, earliest_possible_at, latest_possible_at, ''), attempt_id
    `;
    return this.db
      .prepare(sql)
      .all(scopeKey(scope))
      .map((row: unknown) => rowToAttemptPayload(row as SqlRow));
  }

  attemptRevisions(scope: LedgerScope, attemptId: string): Array<Record<string, unknown>> {
    return this.db
      .prepare(
        `
        SELECT *
        FROM attempt_revisions
        WHERE scope_key=? AND attempt_id=?
        ORDER BY revision
        `,
      )
      .all(scopeKey(scope), attemptId)
      .map((row: unknown) => {
        const item = row as SqlRow;
        return { ...item, payload: parseJsonObject(item.payload_json) };
      });
  }

  attemptMappingHistory(
    scope: LedgerScope,
    attemptId: string,
  ): Array<Record<string, unknown>> {
    return this.db
      .prepare(
        `
        SELECT *
        FROM attempt_mapping_history
        WHERE scope_key=? AND attempt_id=?
        ORDER BY rowid
        `,
      )
      .all(scopeKey(scope), attemptId)
      .map((row: unknown) => row as SqlRow);
  }

  saveModelMapping(mapping: ModelMappingVersion): void {
    validateMappingVersion(mapping);
    assertNoSecrets(mapping);
    this.db
      .prepare(
        `
        INSERT INTO model_mapping_versions(
          version, canonical_families_json, rules_json, review_status,
          source, created_at, reviewed_at, reviewed_by
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(version) DO UPDATE SET
          canonical_families_json=excluded.canonical_families_json,
          rules_json=excluded.rules_json,
          review_status=excluded.review_status,
          source=excluded.source,
          reviewed_at=excluded.reviewed_at,
          reviewed_by=excluded.reviewed_by
        `,
      )
      .run(
        mapping.version,
        JSON.stringify(mapping.canonicalFamilies),
        JSON.stringify(mapping.rules),
        mapping.reviewStatus,
        mapping.source,
        mapping.createdAt,
        mapping.reviewedAt ?? null,
        mapping.reviewedBy ?? null,
      );
  }

  modelMapping(version: string): ModelMappingVersion {
    const row = this.db
      .prepare("SELECT * FROM model_mapping_versions WHERE version=?")
      .get(version) as SqlRow | undefined;
    if (!row) {
      throw new LedgerError(`model mapping not found: ${version}`);
    }
    return {
      version: String(row.version),
      canonicalFamilies: parseJsonArray(row.canonical_families_json),
      rules: parseJsonUnknownArray(row.rules_json) as ModelMappingVersion["rules"],
      reviewStatus: row.review_status as ModelMappingVersion["reviewStatus"],
      source: String(row.source),
      createdAt: String(row.created_at),
      reviewedAt: nullableString(row.reviewed_at),
      reviewedBy: nullableString(row.reviewed_by),
    };
  }

  modelMappings(): ModelMappingVersion[] {
    return this.db
      .prepare("SELECT version FROM model_mapping_versions ORDER BY created_at, version")
      .all()
      .map((row: unknown) => this.modelMapping(String((row as SqlRow).version)));
  }

  modelMappingSuggestions(
    scope: LedgerScope,
    mapping: ModelMappingVersion,
  ) {
    return suggestModelMappings(
      this.listAttempts(scope).map((attempt) => ({
        requestedModelRaw: nullableString(attempt.requestedModelRaw),
        requestedModeRaw: nullableString(attempt.requestedModeRaw),
        requestedReasoningEffortRaw: nullableString(
          attempt.requestedReasoningEffortRaw,
        ),
        recordedFinalModelRaw: nullableString(attempt.recordedFinalModelRaw),
        resolvedModelRaw: nullableString(attempt.resolvedModelRaw),
        completedAnswer: Boolean(attempt.completedAnswer),
      })),
      mapping,
      scope.collectorAccountId,
    );
  }

  reclassifyAttempts(
    scope: LedgerScope,
    mapping: ModelMappingVersion,
    recordedAt: string,
    source = "aggregate_rebuild",
  ): number {
    validateMappingVersion(mapping);
    let changed = 0;
    for (const row of this.listAttempts(scope, true)) {
      const requestedFamily = mapModelEvidence(
        {
          slug: nullableString(row.requestedModelRaw),
          mode: nullableString(row.requestedModeRaw),
          reasoningEffort: nullableString(row.requestedReasoningEffortRaw),
        },
        mapping,
        scope.collectorAccountId,
      );
      const recordedFinalFamily = mapModelEvidence(
        {
          slug: nullableString(row.recordedFinalModelRaw),
          mode: null,
          reasoningEffort: null,
        },
        mapping,
        scope.collectorAccountId,
      );
      const resolvedFamily = mapModelEvidence(
        {
          slug: nullableString(row.resolvedModelRaw),
          mode: null,
          reasoningEffort: null,
        },
        mapping,
        scope.collectorAccountId,
      );
      const current = [
        nullableString(row.requestedFamily),
        nullableString(row.recordedFinalFamily),
        nullableString(row.resolvedFamily),
        String(row.mappingVersion),
      ];
      const next = [
        requestedFamily,
        recordedFinalFamily,
        resolvedFamily,
        mapping.version,
      ];
      if (JSON.stringify(current) === JSON.stringify(next)) {
        continue;
      }
      const attempt = rowToReconstructedAttempt(row, scope);
      const projected = {
        ...attempt,
        requestedFamily,
        recordedFinalFamily,
        resolvedFamily,
        mappingVersion: mapping.version,
      };
      const payload = attemptPayload(projected);
      const nextRevision = Number(row.revision) + 1;
      const projectionFingerprint = fingerprint(payload);
      this.recordMappingHistory(scope, attempt, recordedAt, "prior_projection");
      this.recordMappingHistory(scope, projected, recordedAt, source);
      this.updateAttemptRow(
        scope,
        projected,
        nextRevision,
        projectionFingerprint,
        recordedAt,
      );
      this.insertAttemptRevision(
        scope,
        projected,
        nextRevision,
        projectionFingerprint,
        payload,
        {
          runId: `mapping:${mapping.version}`,
          observedAt: recordedAt,
          sourceKind: "aggregate_rebuild",
          sourceId: attempt.attemptId,
          schemaVersion: "ledger-v2",
        },
        source,
      );
      changed += 1;
    }
    return changed;
  }

  conversations(scope: LedgerScope): string[] {
    return this.db
      .prepare(
        `
        SELECT conversation_id
        FROM conversation_state
        WHERE scope_key=?
        ORDER BY conversation_id
        `,
      )
      .all(scopeKey(scope))
      .map((row: unknown) => String((row as SqlRow).conversation_id));
  }

  private conversationOrigin(
    scope: LedgerScope,
    conversationId: string,
  ): string | null {
    const row = this.db
      .prepare(
        `
        SELECT origin
        FROM conversation_state
        WHERE scope_key=? AND conversation_id=?
        `,
      )
      .get(scopeKey(scope), conversationId) as SqlRow | undefined;
    return nullableString(row?.origin);
  }

  rebuildAttemptsFromMessages(
    scope: LedgerScope,
    mapping: ModelMappingVersion,
    recordedAt: string,
  ): number {
    let changed = 0;
    for (const conversationId of this.conversations(scope)) {
      const attempts = reconstructAttempts(this.messagesFor(scope, conversationId), {
        scope,
        conversationId,
        mapping,
      });
      for (const attempt of attempts) {
        const result = this.upsertAttempt(
          scope,
          attempt,
          {
            runId: `rebuild:${mapping.version}`,
            observedAt: recordedAt,
            sourceKind: "aggregate_rebuild",
            sourceId: conversationId,
            schemaVersion: "ledger-v2",
          },
        );
        if (result.status === "updated" || result.status === "inserted") {
          changed += 1;
        }
      }
    }
    return changed;
  }

  recordCoverageGap(
    scope: LedgerScope,
    gap: {
      sourceKind: string;
      sourceId: string;
      reason: string;
      details?: Record<string, unknown>;
      state?: "open" | "resolved";
    },
    seenAt: string,
  ): string {
    const key = scopeKey(scope);
    const gapId = stableId(key, gap.sourceKind, gap.sourceId, gap.reason);
    const details = gap.details ?? {};
    assertNoSecrets(details);
    this.db
      .prepare(
        `
        INSERT INTO coverage_gaps(
          gap_id, scope_key, collector_account_id, provider, provider_user_id,
          workspace_id, quota_owner_id, source_kind, source_id, reason, state,
          first_seen_at, last_seen_at, details_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(gap_id) DO UPDATE SET
          state=excluded.state,
          last_seen_at=excluded.last_seen_at,
          details_json=excluded.details_json
        `,
      )
      .run(
        gapId,
        key,
        scope.collectorAccountId,
        scope.provider,
        scope.providerUserId,
        scope.workspaceId,
        scope.quotaOwnerId,
        gap.sourceKind,
        gap.sourceId,
        gap.reason,
        gap.state ?? "open",
        seenAt,
        seenAt,
        JSON.stringify(details),
      );
    return gapId;
  }

  coverageGaps(scope: LedgerScope): Array<Record<string, unknown>> {
    return this.db
      .prepare(
        `
        SELECT *
        FROM coverage_gaps
        WHERE scope_key=?
        ORDER BY first_seen_at, gap_id
        `,
      )
      .all(scopeKey(scope))
      .map((row: unknown) => {
        const item = row as SqlRow;
        return { ...item, details: parseJsonObject(item.details_json) };
      });
  }

  writeAggregateRevision(
    scope: LedgerScope,
    mappingVersion: string,
    evaluatedAt: string,
    payload: Record<string, unknown>,
  ): AggregateRevision {
    assertNoSecrets(payload);
    const key = scopeKey(scope);
    const inputFingerprint = fingerprint({
      attempts: this.listAttempts(scope),
      coverageGaps: this.coverageGaps(scope),
    });
    const revisionId = stableId(key, mappingVersion, inputFingerprint, evaluatedAt);
    const createdAt = new Date().toISOString();
    this.db
      .prepare(
        `
        INSERT OR IGNORE INTO aggregate_revisions(
          revision_id, scope_key, collector_account_id, mapping_version,
          input_fingerprint, evaluated_at, created_at, payload_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        `,
      )
      .run(
        revisionId,
        key,
        scope.collectorAccountId,
        mappingVersion,
        inputFingerprint,
        evaluatedAt,
        createdAt,
        JSON.stringify(payload),
      );
    return {
      revisionId,
      mappingVersion,
      inputFingerprint,
      evaluatedAt,
      createdAt,
      payload,
    };
  }

  aggregateRevisions(scope: LedgerScope): AggregateRevision[] {
    return this.db
      .prepare(
        `
        SELECT *
        FROM aggregate_revisions
        WHERE scope_key=?
        ORDER BY evaluated_at, revision_id
        `,
      )
      .all(scopeKey(scope))
      .map((row: unknown) => {
        const item = row as SqlRow;
        return {
          revisionId: String(item.revision_id),
          mappingVersion: String(item.mapping_version),
          inputFingerprint: String(item.input_fingerprint),
          evaluatedAt: String(item.evaluated_at),
          createdAt: String(item.created_at),
          payload: parseJsonObject(item.payload_json),
        };
      });
  }

  attachEvidence(
    attemptId: string,
    evidenceKind: string,
    evidenceId: string,
    scope: LedgerScope,
  ): void {
    this.db
      .prepare(
        `
        INSERT OR IGNORE INTO attempt_evidence(
          attempt_id, evidence_kind, evidence_id, scope_key
        ) VALUES (?, ?, ?, ?)
        `,
      )
      .run(attemptId, evidenceKind, evidenceId, scopeKey(scope));
  }

  private insertAttemptRow(
    scope: LedgerScope,
    attempt: ReconstructedAttempt,
    revision: number,
    projectionFingerprint: string,
    updatedAt: string,
  ): void {
    this.db
      .prepare(
        `
        INSERT INTO attempts(
          attempt_id, scope_key, collector_account_id, provider, provider_user_id,
          workspace_id, quota_owner_id, conversation_id, identity_basis, time_basis,
          attempt_time, earliest_possible_at, latest_possible_at,
          requested_model_raw, requested_mode_raw, requested_reasoning_effort_raw,
          recorded_final_model_raw, resolved_model_raw, requested_family,
          recorded_final_family, resolved_family, mapping_version, outcome,
          completed_answer, generation_started, surface, origin, revision,
          projection_fingerprint, warnings_json, tombstone, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
        `,
      )
      .run(
        attempt.attemptId,
        scopeKey(scope),
        scope.collectorAccountId,
        scope.provider,
        scope.providerUserId,
        scope.workspaceId,
        scope.quotaOwnerId,
        attempt.conversationId,
        attempt.identityBasis,
        attempt.timeBasis,
        attempt.attemptTime,
        attempt.earliestPossibleAt,
        attempt.latestPossibleAt,
        attempt.requestedModelRaw,
        attempt.requestedModeRaw,
        attempt.requestedReasoningEffortRaw,
        attempt.recordedFinalModelRaw,
        attempt.resolvedModelRaw,
        attempt.requestedFamily,
        attempt.recordedFinalFamily,
        attempt.resolvedFamily,
        attempt.mappingVersion,
        attempt.outcome,
        attempt.completedAnswer ? 1 : 0,
        attempt.generationStarted ? 1 : 0,
        attempt.surface,
        attempt.origin,
        revision,
        projectionFingerprint,
        JSON.stringify(attempt.warnings),
        updatedAt,
      );
  }

  private updateAttemptRow(
    scope: LedgerScope,
    attempt: ReconstructedAttempt,
    revision: number,
    projectionFingerprint: string,
    updatedAt: string,
  ): void {
    this.db
      .prepare(
        `
        UPDATE attempts SET
          conversation_id=?, identity_basis=?, time_basis=?,
          attempt_time=?, earliest_possible_at=?, latest_possible_at=?,
          requested_model_raw=?, requested_mode_raw=?, requested_reasoning_effort_raw=?,
          recorded_final_model_raw=?, resolved_model_raw=?, requested_family=?,
          recorded_final_family=?, resolved_family=?, mapping_version=?, outcome=?,
          completed_answer=?, generation_started=?, surface=?, origin=?,
          revision=?, projection_fingerprint=?, warnings_json=?, updated_at=?
        WHERE attempt_id=? AND scope_key=?
        `,
      )
      .run(
        attempt.conversationId,
        attempt.identityBasis,
        attempt.timeBasis,
        attempt.attemptTime,
        attempt.earliestPossibleAt,
        attempt.latestPossibleAt,
        attempt.requestedModelRaw,
        attempt.requestedModeRaw,
        attempt.requestedReasoningEffortRaw,
        attempt.recordedFinalModelRaw,
        attempt.resolvedModelRaw,
        attempt.requestedFamily,
        attempt.recordedFinalFamily,
        attempt.resolvedFamily,
        attempt.mappingVersion,
        attempt.outcome,
        attempt.completedAnswer ? 1 : 0,
        attempt.generationStarted ? 1 : 0,
        attempt.surface,
        attempt.origin,
        revision,
        projectionFingerprint,
        JSON.stringify(attempt.warnings),
        updatedAt,
        attempt.attemptId,
        scopeKey(scope),
      );
  }

  private insertAttemptRevision(
    scope: LedgerScope,
    attempt: ReconstructedAttempt | Record<string, unknown>,
    revision: number,
    projectionFingerprint: string,
    payload: Record<string, unknown>,
    context: IngestContext,
    source: string,
  ): void {
    this.db
      .prepare(
        `
        INSERT OR IGNORE INTO attempt_revisions(
          attempt_id, revision, scope_key, projection_fingerprint,
          payload_json, recorded_at, source
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        `,
      )
      .run(
        String(attempt.attemptId ?? ""),
        revision,
        scopeKey(scope),
        projectionFingerprint,
        JSON.stringify(payload),
        context.observedAt,
        source,
      );
  }

  private recordMappingHistory(
    scope: LedgerScope,
    attempt: ReconstructedAttempt,
    recordedAt: string,
    source: string,
  ): void {
    const historyId = stableId(
      scopeKey(scope),
      attempt.attemptId,
      attempt.mappingVersion,
    );
    this.db
      .prepare(
        `
        INSERT OR IGNORE INTO attempt_mapping_history(
          history_id, attempt_id, scope_key, mapping_version,
          requested_family, recorded_final_family, resolved_family,
          recorded_at, source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        `,
      )
      .run(
        historyId,
        attempt.attemptId,
        scopeKey(scope),
        attempt.mappingVersion,
        attempt.requestedFamily,
        attempt.recordedFinalFamily,
        attempt.resolvedFamily,
        recordedAt,
        source,
      );
  }
}

function loadMigrations(): Array<{ version: number; name: string; sql: string }> {
  const moduleDirectory = dirname(fileURLToPath(import.meta.url));
  const candidates = [
    resolve(process.cwd(), "migrations"),
    resolve(moduleDirectory, "../../migrations"),
    resolve(moduleDirectory, "../../../migrations"),
  ];
  const directory = candidates.find((candidate) => existsSync(candidate));
  if (!directory) {
    throw new LedgerError("migrations directory not found");
  }
  return readdirSync(directory)
    .filter((name) => /^\d+_.+\.sql$/.test(name))
    .map((name) => {
      const match = /^(\d+)_(.+)\.sql$/.exec(name);
      if (!match) {
        throw new LedgerError(`invalid migration filename: ${name}`);
      }
      return {
        version: Number(match[1]),
        name,
        sql: readFileSync(resolve(directory, name), "utf8"),
      };
    })
    .sort((left, right) => left.version - right.version);
}

function scopeFromRow(row: SqlRow): LedgerScope {
  return {
    collectorAccountId: String(row.collector_account_id),
    provider: String(row.provider),
    providerUserId: nullableString(row.provider_user_id),
    workspaceId: nullableString(row.workspace_id),
    quotaOwnerId: nullableString(row.quota_owner_id),
    surface: String(row.surface) as LedgerScope["surface"],
  };
}

function sanitizeMessage(record: MessageRecord): MessageRecord {
  const clean: MessageRecord = {
    conversationId: sanitizeToken(record.conversationId) ?? "unknown-conversation",
    messageId: sanitizeToken(record.messageId) ?? "unknown-message",
    nodeId: sanitizeToken(record.nodeId) ?? null,
    parentId: sanitizeToken(record.parentId) ?? null,
    children: record.children
      .map((child) => sanitizeToken(child))
      .filter((child): child is string => child !== null),
    role: sanitizeToken(record.role) ?? null,
    channel: sanitizeToken(record.channel) ?? null,
    createdAt: safeTimestamp(record.createdAt),
    status: sanitizeToken(record.status) ?? null,
    endTurn: record.endTurn,
    requestedModelRaw: sanitizeToken(record.requestedModelRaw) ?? null,
    requestedModeRaw: sanitizeToken(record.requestedModeRaw) ?? null,
    requestedReasoningEffortRaw: sanitizeToken(record.requestedReasoningEffortRaw) ?? null,
    recordedFinalModelRaw: sanitizeToken(record.recordedFinalModelRaw) ?? null,
    generationId: sanitizeToken(record.generationId) ?? null,
    requestId: sanitizeToken(record.requestId) ?? null,
    surface: record.surface,
    origin: sanitizeToken(record.origin) ?? null,
    metadata: sanitizeMetadata(record.metadata),
  };
  return clean;
}

function messagePayload(record: MessageRecord): Record<string, unknown> {
  return {
    conversationId: record.conversationId,
    messageId: record.messageId,
    nodeId: record.nodeId,
    parentId: record.parentId,
    children: [...record.children].sort(),
    role: record.role,
    channel: record.channel,
    createdAt: record.createdAt,
    status: record.status,
    endTurn: record.endTurn,
    requestedModelRaw: record.requestedModelRaw,
    requestedModeRaw: record.requestedModeRaw,
    requestedReasoningEffortRaw: record.requestedReasoningEffortRaw,
    recordedFinalModelRaw: record.recordedFinalModelRaw,
    generationId: record.generationId,
    requestId: record.requestId,
    surface: record.surface,
    origin: record.origin,
    metadata: record.metadata,
  };
}

function stableObservationPayload(
  payload: Record<string, unknown>,
): Record<string, unknown> {
  const provenance = payload.provenance;
  if (provenance === null || typeof provenance !== "object" || Array.isArray(provenance)) {
    return payload;
  }
  const { run_id: _runId, ...stableProvenance } = provenance as Record<string, unknown>;
  return { ...payload, provenance: stableProvenance };
}

function attemptPayload(
  attempt: Pick<
    ReconstructedAttempt,
    | "attemptId"
    | "conversationId"
    | "identityBasis"
    | "timeBasis"
    | "attemptTime"
    | "earliestPossibleAt"
    | "latestPossibleAt"
    | "requestedModelRaw"
    | "requestedModeRaw"
    | "requestedReasoningEffortRaw"
    | "recordedFinalModelRaw"
    | "resolvedModelRaw"
    | "requestedFamily"
    | "recordedFinalFamily"
    | "resolvedFamily"
    | "mappingVersion"
    | "outcome"
    | "completedAnswer"
    | "generationStarted"
    | "surface"
    | "origin"
    | "aliases"
    | "evidenceMessageIds"
    | "warnings"
  >,
): Record<string, unknown> {
  return {
    attemptId: attempt.attemptId,
    conversationId: attempt.conversationId,
    identityBasis: attempt.identityBasis,
    timeBasis: attempt.timeBasis,
    attemptTime: attempt.attemptTime,
    earliestPossibleAt: attempt.earliestPossibleAt,
    latestPossibleAt: attempt.latestPossibleAt,
    requestedModelRaw: attempt.requestedModelRaw,
    requestedModeRaw: attempt.requestedModeRaw,
    requestedReasoningEffortRaw: attempt.requestedReasoningEffortRaw,
    recordedFinalModelRaw: attempt.recordedFinalModelRaw,
    resolvedModelRaw: attempt.resolvedModelRaw,
    requestedFamily: attempt.requestedFamily,
    recordedFinalFamily: attempt.recordedFinalFamily,
    resolvedFamily: attempt.resolvedFamily,
    mappingVersion: attempt.mappingVersion,
    outcome: attempt.outcome,
    completedAnswer: attempt.completedAnswer,
    generationStarted: attempt.generationStarted,
    surface: attempt.surface,
    origin: attempt.origin,
    aliases: [...attempt.aliases].sort((left, right) =>
      JSON.stringify(left).localeCompare(JSON.stringify(right)),
    ),
    evidenceMessageIds: [...attempt.evidenceMessageIds].sort(),
    warnings: [...attempt.warnings].sort(),
  };
}

function messageFromRow(row: SqlRow): MessageRecord {
  return {
    conversationId: String(row.conversation_id),
    messageId: String(row.message_id),
    nodeId: nullableString(row.node_id),
    parentId: nullableString(row.parent_id),
    children: parseJsonArray(row.children_json),
    role: nullableString(row.role),
    channel: nullableString(row.channel),
    createdAt: nullableString(row.created_at),
    status: nullableString(row.status),
    endTurn: row.end_turn === null ? null : Number(row.end_turn) === 1,
    requestedModelRaw: nullableString(row.requested_model_raw),
    requestedModeRaw: nullableString(row.requested_mode_raw),
    requestedReasoningEffortRaw: nullableString(row.requested_reasoning_effort_raw),
    recordedFinalModelRaw: nullableString(row.recorded_final_model_raw),
    generationId: nullableString(row.generation_id),
    requestId: nullableString(row.request_id),
    surface: String(row.surface) as MessageRecord["surface"],
    origin: nullableString(row.origin),
    metadata: parseJsonObject(row.metadata_json),
  };
}

function rowToAttemptPayload(row: SqlRow): Record<string, unknown> {
  return {
    attemptId: String(row.attempt_id),
    scopeKey: String(row.scope_key),
    collectorAccountId: String(row.collector_account_id),
    provider: String(row.provider),
    providerUserId: nullableString(row.provider_user_id),
    workspaceId: nullableString(row.workspace_id),
    quotaOwnerId: nullableString(row.quota_owner_id),
    conversationId: String(row.conversation_id),
    identityBasis: String(row.identity_basis),
    timeBasis: String(row.time_basis),
    attemptTime: nullableString(row.attempt_time),
    earliestPossibleAt: nullableString(row.earliest_possible_at),
    latestPossibleAt: nullableString(row.latest_possible_at),
    requestedModelRaw: nullableString(row.requested_model_raw),
    requestedModeRaw: nullableString(row.requested_mode_raw),
    requestedReasoningEffortRaw: nullableString(row.requested_reasoning_effort_raw),
    recordedFinalModelRaw: nullableString(row.recorded_final_model_raw),
    resolvedModelRaw: nullableString(row.resolved_model_raw),
    requestedFamily: nullableString(row.requested_family),
    recordedFinalFamily: nullableString(row.recorded_final_family),
    resolvedFamily: nullableString(row.resolved_family),
    mappingVersion: String(row.mapping_version),
    outcome: String(row.outcome),
    completedAnswer: Number(row.completed_answer) === 1,
    generationStarted: Number(row.generation_started) === 1,
    surface: String(row.surface),
    origin: nullableString(row.origin),
    revision: Number(row.revision),
    projectionFingerprint: String(row.projection_fingerprint),
    warnings: parseJsonArray(row.warnings_json),
    tombstone: Number(row.tombstone) === 1,
  };
}

function rowToReconstructedAttempt(
  row: Record<string, unknown>,
  scope: LedgerScope,
): ReconstructedAttempt {
  return {
    attemptId: String(row.attemptId),
    conversationId: String(row.conversationId),
    identityBasis: String(row.identityBasis),
    timeBasis: String(row.timeBasis),
    attemptTime: nullableString(row.attemptTime),
    earliestPossibleAt: nullableString(row.earliestPossibleAt),
    latestPossibleAt: nullableString(row.latestPossibleAt),
    requestedModelRaw: nullableString(row.requestedModelRaw),
    requestedModeRaw: nullableString(row.requestedModeRaw),
    requestedReasoningEffortRaw: nullableString(row.requestedReasoningEffortRaw),
    recordedFinalModelRaw: nullableString(row.recordedFinalModelRaw),
    resolvedModelRaw: nullableString(row.resolvedModelRaw),
    requestedFamily: nullableString(row.requestedFamily),
    recordedFinalFamily: nullableString(row.recordedFinalFamily),
    resolvedFamily: nullableString(row.resolvedFamily),
    mappingVersion: String(row.mappingVersion),
    outcome: String(row.outcome),
    completedAnswer: Boolean(row.completedAnswer),
    generationStarted: Boolean(row.generationStarted),
    surface: String(row.surface) as ReconstructedAttempt["surface"],
    origin: nullableString(row.origin),
    aliases: [],
    evidenceMessageIds: [],
    revision: Number(row.revision ?? 1),
    warnings: parseJsonArray(row.warnings),
    scope,
  };
}

function parseJsonObject(value: unknown): Record<string, unknown> {
  if (typeof value !== "string") {
    return {};
  }
  try {
    const parsed = JSON.parse(value) as unknown;
    return parsed !== null && typeof parsed === "object" && !Array.isArray(parsed)
      ? parsed as Record<string, unknown>
      : {};
  } catch {
    return {};
  }
}

function parseJsonArray(value: unknown): string[] {
  if (typeof value !== "string") {
    return [];
  }
  try {
    const parsed = JSON.parse(value) as unknown;
    return Array.isArray(parsed) ? parsed.map(String) : [];
  } catch {
    return [];
  }
}

function parseJsonUnknownArray(value: unknown): unknown[] {
  if (typeof value !== "string") {
    return [];
  }
  try {
    const parsed = JSON.parse(value) as unknown;
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function nullableString(value: unknown): string | null {
  return value === null || value === undefined ? null : String(value);
}

function safeTimestamp(value: string | null): string | null {
  return value !== null && Number.isFinite(Date.parse(value)) ? value : null;
}
