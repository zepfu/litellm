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
  isMappingPublished,
  ModelMappingError,
  normalizeMappingVersion,
  resolveModelEvidence,
  suggestModelMappings,
  validateMappingOwnerConsistency,
  validateMappingVersion,
} from "../normalize/model-mapping.js";
import {
  assertNoSecrets,
  observationProjection,
  sanitizeMetadata,
  sanitizeToken,
} from "../security/sanitizer.js";
import {
  canonicalJson,
  collectorScopeKey,
  fingerprint,
  sanitizeProvenance,
  scopeKey,
  stableId,
} from "./identity.js";
import type {
  IngestContext,
  IngestResult,
  LedgerMessage,
  LedgerQuarantine,
  LedgerScope,
  MappingOwnerBinding,
  ModelMappingVersion,
  QuarantinedTimestampEvidence,
  ReconstructedAttempt,
  StoredMessage,
} from "./types.js";

type SqlRow = Record<string, unknown>;
type SqliteDatabase = InstanceType<typeof Database>;
type AttemptAliasOwner = {
  aliasKind: string;
  aliasValue: string;
  attemptId: string;
  identityBasis: string;
  completedAnswer: boolean;
  tombstone: boolean;
  ambiguous: boolean;
  generationAliases: string[];
};

const MERGEABLE_ATTEMPT_ALIAS_KINDS = new Set([
  "branch",
  "generation",
  "message",
  "prompt",
]);

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
    try {
      this.migrate();
    } catch (error) {
      this.db.close();
      throw error;
    }
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
    const applied = new Map(
      this.db
        .prepare("SELECT version, name, checksum FROM schema_migrations ORDER BY version")
        .all()
        .map((row: unknown) => [Number((row as SqlRow).version), row as SqlRow]),
    );
    this.transaction(() => {
      const versions = new Set<number>();
      let migrateActivity = false;
      for (const migration of migrations) {
        if (versions.has(migration.version)) {
          throw new LedgerError(`duplicate migration version: ${migration.version}`);
        }
        versions.add(migration.version);
        const prior = applied.get(migration.version);
        if (prior) {
          if (prior.name !== migration.name || prior.checksum !== fingerprint(migration.sql)) {
            throw new LedgerError(`applied migration does not match source: ${migration.version}`);
          }
          continue;
        }
        this.db.exec(migration.sql);
        migrateActivity ||= migration.version === 5;
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
      }
      // The owner conversion uses current ledger writes, including quarantine.
      // Apply every pending schema change first, in the same transaction.
      if (migrateActivity) {
        this.migrateActivityScopes();
      }
    });
  }

  private migrateActivityScopes(): void {
    const groups = new Map<string, LedgerScope[]>();
    for (const account of this.listAccounts()) {
      const key = scopeKey(account);
      groups.set(key, [...(groups.get(key) ?? []), account]);
    }
    for (const [key, scopes] of groups) {
      const keys = [...new Set([key, ...scopes.map(collectorScopeKey)])];
      const placeholders = keys.map(() => "?").join(",");
      const rowsFor = (table: string): SqlRow[] =>
        this.db.prepare(`SELECT * FROM ${table} WHERE scope_key IN (${placeholders})`)
          .all(...keys) as SqlRow[];
      const legacyRows = rowsFor("attempts");
      const aliases = rowsFor("attempt_aliases");
      const evidence = rowsFor("attempt_evidence");
      const attempts = legacyRows.map((row) => {
        const scope = scopes.find((item) => item.collectorAccountId === row.collector_account_id)!;
        const payload = rowToAttemptPayload(row);
        payload.aliases = aliases.filter((item) => item.attempt_id === row.attempt_id)
          .map((item) => [String(item.alias_kind), String(item.alias_value)]);
        payload.evidenceMessageIds = evidence
          .filter((item) => item.attempt_id === row.attempt_id && item.evidence_kind === "message")
          .map((item) => String(item.evidence_id));
        const retainedAliases = payload.aliases as Array<[string, string]>;
        for (const messageId of payload.evidenceMessageIds as string[]) {
          if (retainedAliases.some(([kind, value]) =>
            kind === "prompt" && value === `${messageId}:prompt:${messageId}`,
          )) {
            retainedAliases.push(["prompt", `${String(row.conversation_id)}:${messageId}:prompt:${messageId}`]);
          }
        }
        return { row, scope, attempt: rowToReconstructedAttempt(payload, scope) };
      }).sort((left, right) => String(left.row.updated_at).localeCompare(String(right.row.updated_at)));

      // Immutable IDs survive the scope change. Occurrence numbers span collectors.
      for (const table of [
        "observations", "message_revisions", "attempts", "attempt_revisions",
        "attempt_evidence", "attempt_mapping_history", "collector_runs", "coverage_gaps",
      ]) {
        this.db.prepare(`UPDATE ${table} SET scope_key=? WHERE scope_key IN (${placeholders})`)
          .run(key, ...keys);
      }
      const previousObservations = new Map<string, { id: string; revision: number }>();
      const observations = rowsFor("observations").sort((left, right) =>
        String(left.observed_at).localeCompare(String(right.observed_at)) ||
        Number(left.revision_number) - Number(right.revision_number) ||
        String(left.observation_id).localeCompare(String(right.observation_id)),
      );
      for (const row of observations) {
        const source = canonicalJson([row.source_kind, row.source_id]);
        const previous = previousObservations.get(source);
        const revision = (previous?.revision ?? 0) + 1;
        this.db.prepare(`
          UPDATE observations SET revision_number=?, supersedes_observation_id=?
          WHERE observation_id=?
        `).run(revision, previous?.id ?? null, row.observation_id);
        previousObservations.set(source, { id: String(row.observation_id), revision });
      }
      const messageRevisions = new Map<string, number>();
      for (const row of rowsFor("message_revisions").sort((left, right) =>
        String(left.observed_at).localeCompare(String(right.observed_at)) ||
        Number(left.revision) - Number(right.revision) ||
        String(left.revision_id).localeCompare(String(right.revision_id)),
      )) {
        const message = canonicalJson([row.conversation_id, row.message_id]);
        const revision = (messageRevisions.get(message) ?? 0) + 1;
        this.db.prepare("UPDATE message_revisions SET revision=? WHERE revision_id=?")
          .run(revision, row.revision_id);
        messageRevisions.set(message, revision);
      }

      for (const [table, identity, timestamp] of [
        ["conversation_state", ["conversation_id"], "updated_at"],
        ["message_records", ["conversation_id", "message_id"], "updated_at"],
        ["aggregate_revisions", ["mapping_version", "input_fingerprint", "evaluated_at"], "created_at"],
      ] as const) {
        const selected = new Map<string, SqlRow>();
        for (const row of rowsFor(table).sort((left, right) =>
          String(left[timestamp] ?? "").localeCompare(String(right[timestamp] ?? "")),
        )) {
          selected.set(canonicalJson(identity.map((column) => row[column])), { ...row, scope_key: key });
        }
        this.db.prepare(`DELETE FROM ${table} WHERE scope_key IN (${placeholders})`).run(...keys);
        for (const row of selected.values()) {
          const columns = Object.keys(row);
          this.db.prepare(`INSERT INTO ${table}(${columns.join(",")}) VALUES (${columns.map(() => "?").join(",")})`)
            .run(...columns.map((column) => row[column]));
        }
      }

      const messages = rowsFor("message_records");
      for (const row of messages) {
        const revision = this.db.prepare(`
          SELECT revision FROM message_revisions
          WHERE scope_key=? AND conversation_id=? AND message_id=?
            AND collector_account_id=? AND revision_fingerprint=?
          ORDER BY observed_at DESC, revision DESC LIMIT 1
        `).get(key, row.conversation_id, row.message_id, row.collector_account_id,
          row.revision_fingerprint) as SqlRow | undefined;
        if (revision) {
          this.db.prepare(`
            UPDATE message_records SET revision=?
            WHERE scope_key=? AND conversation_id=? AND message_id=?
          `).run(revision.revision, key, row.conversation_id, row.message_id);
        }
      }
      const messageIds = new Map(messages.map((row) => [
        `${String(row.conversation_id)}:${String(row.message_id)}`,
        stableId("message", String(row.conversation_id), String(row.message_id)),
      ]));
      const provenance = rowsFor("activity_provenance");
      this.db.prepare(`DELETE FROM activity_provenance WHERE scope_key IN (${placeholders})`).run(...keys);
      for (const row of provenance) {
        const id = row.activity_kind === "message"
          ? messageIds.get(String(row.activity_id)) ?? String(row.activity_id)
          : String(row.activity_id);
        for (const seenAt of [row.first_seen_at, row.last_seen_at]) {
          this.recordActivityProvenance(key, String(row.activity_kind), id,
            String(row.collector_account_id), String(seenAt));
        }
      }
      this.db.prepare(`DELETE FROM attempt_aliases WHERE scope_key IN (${placeholders})`).run(...keys);
      for (const { row, scope, attempt } of attempts) {
        if (Number(row.tombstone) === 0) {
          this.upsertAttempt(scope, attempt, {
            runId: "migration:activity-scope",
            observedAt: String(row.updated_at),
            sourceKind: "migration",
            sourceId: attempt.attemptId,
            schemaVersion: "ledger-v5",
          });
        }
      }
    }
    for (const row of this.db.prepare("SELECT * FROM history_state").all() as SqlRow[]) {
      const scope = this.accountScope(String(row.collector_account_id));
      this.db.prepare("UPDATE history_state SET scope_key=? WHERE scope_key=?")
        .run(collectorScopeKey(scope), row.scope_key);
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
    for (const mapping of this.modelMappings()) {
      this.assertMappingOwnerConsistency(mapping, scope);
    }
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
        collectorScopeKey(scope),
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
    quarantine: LedgerQuarantine = clearQuarantine(),
  ): { observationId: string; inserted: boolean } {
    const sanitized = observationProjection(payload, {
      sourceKind: context.sourceKind,
      runId: context.runId,
      evidenceId: context.sourceId,
    });
    assertNoSecrets(sanitized);
    const key = scopeKey(scope);
    const previous = this.db
      .prepare(
        `
        SELECT observation_id, revision_number, revision_fingerprint,
               collector_account_id, quarantine_json
        FROM observations
        WHERE scope_key=? AND source_kind=? AND source_id=?
        ORDER BY revision_number DESC
        LIMIT 1
        `,
      )
      .get(key, context.sourceKind, context.sourceId) as SqlRow | undefined;
    const persistedQuarantine = mergeQuarantines(
      parseQuarantine(previous?.quarantine_json),
      quarantine,
    );
    assertNoSecrets(persistedQuarantine);
    const revisionFingerprint = fingerprint(
      observationFingerprintPayload(sanitized, persistedQuarantine),
    );
    if (
      previous &&
      String(previous.revision_fingerprint) === revisionFingerprint &&
      String(previous.collector_account_id) === scope.collectorAccountId
    ) {
      return { observationId: String(previous.observation_id), inserted: false };
    }
    const revisionNumber = Number(previous?.revision_number ?? 0) + 1;
    const observationId = stableId(
      key,
      context.sourceKind,
      context.sourceId,
      String(revisionNumber),
      revisionFingerprint,
    );
    const provenance = {
      ...sanitizeProvenance(context.provenance),
      collector_account_id: scope.collectorAccountId,
      activity_scope_key: key,
    };
    assertNoSecrets(provenance);
    this.db
      .prepare(
        `
        INSERT INTO observations(
          observation_id, scope_key, collector_account_id, provider,
          provider_user_id, workspace_id, quota_owner_id, source_kind, source_id,
          revision_fingerprint, revision_number, surface, conversation_id,
          payload_json, observed_at, run_id, schema_version, provenance_json,
          supersedes_observation_id, quarantine_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        JSON.stringify(provenance),
        previous?.observation_id ?? null,
        JSON.stringify(persistedQuarantine),
      );
    this.recordActivityProvenance(
      key,
      "observation",
      observationId,
      scope.collectorAccountId,
      context.observedAt,
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
    this.assertMappingOwnerConsistency(mapping, scope);
    return this.transaction(() => {
      const sanitizedMessages = detail.messages.map((message) =>
        sanitizeMessage(message, context.observedAt),
      );
      const observationQuarantine = mergeQuarantines(
        quarantineForTimestampFields(
          [
            { field: "createdAt", value: detail.createdAt },
            { field: "updatedAt", value: detail.updatedAt },
          ],
          context.observedAt,
        ),
        ...sanitizedMessages.map((message) => message.quarantine),
      );
      const observation = this.insertObservation(
        scope,
        context,
        {
          conversation_id: detail.conversationId,
          created_at: safeTimestamp(detail.createdAt, context.observedAt),
          updated_at: safeTimestamp(detail.updatedAt, context.observedAt),
          current_node: detail.currentNode,
          surface: detail.surface,
          detail_route: detail.detailRoute,
          pagination_state: detail.paginationState,
          messages: sanitizedMessages.map((message) => messagePayload(message)),
          coverage: detail.coverage,
          warnings: detail.warnings,
        },
        observationQuarantine,
      );
      if (summary) {
        this.upsertConversation(
          scope,
          summary,
          context.runId,
          context.observedAt,
          observationQuarantine,
        );
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
          context.observedAt,
          observationQuarantine,
        );
      }

      let messageInserted = 0;
      let messageDeduplicated = 0;
      for (const message of sanitizedMessages) {
        const result = this.upsertMessage(scope, message, context);
        if (result.inserted) {
          messageInserted += 1;
        } else {
          messageDeduplicated += 1;
        }
      }

      const messages = this.reconstructionMessages(scope, detail.conversationId);
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
        const preparedAttempt = applyMessageQuarantine(attempt, messages);
        const result = this.upsertAttempt(scope, preparedAttempt, context, mapping);
        attemptInserted += result.status === "inserted" ? 1 : 0;
        attemptUpdated += result.status === "updated" ? 1 : 0;
        attemptDeduplicated += result.status === "deduplicated" ? 1 : 0;
        aliasConflicts += result.aliasConflicts;
        this.attachEvidence(
          result.attemptId,
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
    observedAt?: string,
    quarantine?: LedgerQuarantine,
  ): void {
    const key = scopeKey(scope);
    const createdAt = safeTimestamp(summary.createdAt, observedAt);
    const updatedAt = safeTimestamp(summary.updatedAt, observedAt);
    const incomingQuarantine = mergeQuarantines(
      quarantine,
      observedAt === undefined
        ? clearQuarantine()
        : quarantineForTimestampFields(
            [
              { field: "createdAt", value: summary.createdAt },
              { field: "updatedAt", value: summary.updatedAt },
            ],
            observedAt,
          ),
    );
    const current = this.db
      .prepare(
        `
        SELECT warnings_json, quarantine_json
        FROM conversation_state
        WHERE scope_key=? AND conversation_id=?
        `,
      )
      .get(key, summary.conversationId) as SqlRow | undefined;
    const persistedQuarantine = mergeQuarantines(
      parseQuarantine(current?.quarantine_json),
      incomingQuarantine,
    );
    const warnings = uniqueStrings([
      ...parseJsonArray(current?.warnings_json),
      ...persistedQuarantine.warnings,
    ]);
    this.db
      .prepare(
        `
        INSERT INTO conversation_state(
          scope_key, collector_account_id, provider, provider_user_id,
          workspace_id, quota_owner_id, conversation_id, created_at, updated_at,
          is_archived, surface, origin, current_node, page_coverage,
          warnings_json, last_seen_run_id, quarantine_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(scope_key, conversation_id) DO UPDATE SET
          created_at=COALESCE(excluded.created_at, conversation_state.created_at),
          updated_at=CASE
            WHEN conversation_state.updated_at IS NULL
              OR excluded.updated_at IS NULL
              OR excluded.updated_at >= conversation_state.updated_at
            THEN COALESCE(excluded.updated_at, conversation_state.updated_at)
            ELSE conversation_state.updated_at
          END,
          is_archived=CASE
            WHEN conversation_state.updated_at IS NULL
              OR excluded.updated_at IS NULL
              OR excluded.updated_at >= conversation_state.updated_at
            THEN excluded.is_archived
            ELSE conversation_state.is_archived
          END,
          surface=CASE
            WHEN conversation_state.updated_at IS NULL
              OR excluded.updated_at IS NULL
              OR excluded.updated_at >= conversation_state.updated_at
            THEN excluded.surface
            ELSE conversation_state.surface
          END,
          origin=CASE
            WHEN conversation_state.updated_at IS NULL
              OR excluded.updated_at IS NULL
              OR excluded.updated_at >= conversation_state.updated_at
            THEN COALESCE(excluded.origin, conversation_state.origin)
            ELSE conversation_state.origin
          END,
          current_node=CASE
            WHEN conversation_state.updated_at IS NULL
              OR excluded.updated_at IS NULL
              OR excluded.updated_at >= conversation_state.updated_at
            THEN COALESCE(excluded.current_node, conversation_state.current_node)
            ELSE conversation_state.current_node
          END,
          page_coverage=CASE
            WHEN conversation_state.updated_at IS NULL
              OR excluded.updated_at IS NULL
              OR excluded.updated_at >= conversation_state.updated_at
            THEN excluded.page_coverage
            ELSE conversation_state.page_coverage
          END,
          warnings_json=CASE
            WHEN conversation_state.updated_at IS NULL
              OR excluded.updated_at IS NULL
              OR excluded.updated_at >= conversation_state.updated_at
            THEN excluded.warnings_json
            ELSE conversation_state.warnings_json
          END,
          quarantine_json=excluded.quarantine_json,
          last_seen_run_id=CASE
            WHEN conversation_state.updated_at IS NULL
              OR excluded.updated_at IS NULL
              OR excluded.updated_at >= conversation_state.updated_at
            THEN excluded.last_seen_run_id
            ELSE conversation_state.last_seen_run_id
          END
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
        createdAt,
        updatedAt,
        summary.isArchived ? 1 : 0,
        summary.surface,
        summary.origin,
        summary.currentNode,
        summary.coverage,
        JSON.stringify(warnings),
        runId,
        JSON.stringify(persistedQuarantine),
      );
    this.recordActivityProvenance(
      key,
      "conversation",
      summary.conversationId,
      scope.collectorAccountId,
      observedAt ?? updatedAt ?? new Date().toISOString(),
    );
  }

  upsertMessage(
    scope: LedgerScope,
    record: MessageRecord,
    context: IngestContext,
  ): { inserted: boolean; revision: number; revisionFingerprint: string } {
    const key = scopeKey(scope);
    const initial = sanitizeMessage(record, context.observedAt);
    const current = this.db
      .prepare(
        `
        SELECT revision, revision_fingerprint, updated_at,
               collector_account_id, quarantine_json
        FROM message_records
        WHERE scope_key=? AND conversation_id=? AND message_id=?
        `,
      )
      .get(key, initial.conversationId, initial.messageId) as SqlRow | undefined;
    const clean: LedgerMessage = {
      ...initial,
      quarantine: mergeQuarantines(
        parseQuarantine(current?.quarantine_json),
        initial.quarantine,
      ),
    };
    assertNoSecrets(clean);
    const payload = messageRevisionPayload(clean);
    const revisionFingerprint = fingerprint(messageFingerprintPayload(clean));
    const latestRevision = this.db
      .prepare(
        `
        SELECT revision, revision_fingerprint, collector_account_id
        FROM message_revisions
        WHERE scope_key=? AND conversation_id=? AND message_id=?
        ORDER BY revision DESC
        LIMIT 1
        `,
      )
      .get(key, clean.conversationId, clean.messageId) as SqlRow | undefined;
    const currentMatches =
      current &&
      String(current.revision_fingerprint) === revisionFingerprint &&
      String(current.collector_account_id) === scope.collectorAccountId;
    const latestHistoryMatches =
      latestRevision &&
      String(latestRevision.revision_fingerprint) === revisionFingerprint &&
      String(latestRevision.collector_account_id) === scope.collectorAccountId;
    if (
      currentMatches
    ) {
      this.recordActivityProvenance(
        key,
        "message",
        stableId("message", clean.conversationId, clean.messageId),
        scope.collectorAccountId,
        context.observedAt,
      );
      return {
        inserted: false,
        revision: Number(current?.revision ?? latestRevision?.revision ?? 0),
        revisionFingerprint,
      };
    }
    const reuseHistoryRevision = Boolean(latestHistoryMatches);
    const revision = reuseHistoryRevision
      ? Number(latestRevision?.revision ?? 0)
      : Number(latestRevision?.revision ?? current?.revision ?? 0) + 1;
    const freshEvidence = !isOlderEvidence(context.observedAt, current?.updated_at);
    const revisionId = stableId(
      key,
      clean.conversationId,
      clean.messageId,
      String(revision),
      revisionFingerprint,
    );
    if (!reuseHistoryRevision) {
      this.db
        .prepare(
          `
          INSERT INTO message_revisions(
            revision_id, scope_key, collector_account_id, provider,
            provider_user_id, workspace_id, quota_owner_id, conversation_id,
            message_id, revision, revision_fingerprint, payload_json,
            observed_at, run_id, quarantine_json
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
          JSON.stringify(clean.quarantine),
        );
    }
    if (!current) {
      this.db
        .prepare(
          `
          INSERT INTO message_records(
            scope_key, collector_account_id, provider, provider_user_id,
            workspace_id, quota_owner_id, conversation_id, message_id, node_id,
            parent_id, children_json, role, channel, created_at, status, end_turn,
            requested_model_raw, requested_mode_raw, requested_reasoning_effort_raw,
            recorded_final_model_raw, generation_id, request_id, surface, origin,
            metadata_json, revision, revision_fingerprint, updated_at, quarantine_json
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
          JSON.stringify(clean.quarantine),
        );
    } else if (freshEvidence) {
      this.db
        .prepare(
          `
          UPDATE message_records SET
            collector_account_id=?, provider=?, provider_user_id=?,
            workspace_id=?, quota_owner_id=?, node_id=?, parent_id=?,
            children_json=?, role=?, channel=?, created_at=?, status=?,
            end_turn=?, requested_model_raw=?, requested_mode_raw=?,
            requested_reasoning_effort_raw=?, recorded_final_model_raw=?,
            generation_id=?, request_id=?, surface=?, origin=?, metadata_json=?,
            revision=?, revision_fingerprint=?, updated_at=?, quarantine_json=?
          WHERE scope_key=? AND conversation_id=? AND message_id=?
          `,
        )
        .run(
          scope.collectorAccountId,
          scope.provider,
          scope.providerUserId,
          scope.workspaceId,
          scope.quotaOwnerId,
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
          JSON.stringify(clean.quarantine),
          key,
          clean.conversationId,
          clean.messageId,
        );
    }
    this.recordActivityProvenance(
      key,
      "message",
      stableId("message", clean.conversationId, clean.messageId),
      scope.collectorAccountId,
      context.observedAt,
    );
    return {
      inserted: !reuseHistoryRevision || freshEvidence || !current,
      revision,
      revisionFingerprint,
    };
  }

  messagesFor(scope: LedgerScope, conversationId: string): LedgerMessage[] {
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
        return {
          ...item,
          payload: parseJsonObject(item.payload_json),
          quarantine: parseQuarantine(item.quarantine_json),
        };
      });
  }

  upsertAttempt(
    scope: LedgerScope,
    attempt: ReconstructedAttempt,
    context: IngestContext,
    mapping?: ModelMappingVersion,
  ): {
    attemptId: string;
    status: "inserted" | "updated" | "deduplicated";
    aliasConflicts: number;
  } {
    const key = scopeKey(scope);
    const incomingAttempt = sanitizeAttempt(attempt, context.observedAt);
    let cleanAttempt = incomingAttempt;
    let effectiveMapping: ModelMappingVersion | undefined;
    const aliases = uniqueAttemptAliases(incomingAttempt.aliases);
    const aliasOwners = this.findAttemptAliasOwners(scope, aliases);
    const persistedIncomingGenerationAliases = this.generationAliasesFor(
      scope,
      incomingAttempt.attemptId,
    );
    const incomingGenerationAliases = new Set([
      ...persistedIncomingGenerationAliases,
      ...aliases
        .filter(([aliasKind]) => aliasKind === "generation")
        .map(([, aliasValue]) => aliasValue),
    ]);
    const mergeableOwnerIds = new Set<string>();
    for (const owner of aliasOwners) {
      if (
        owner.attemptId === incomingAttempt.attemptId ||
        !MERGEABLE_ATTEMPT_ALIAS_KINDS.has(owner.aliasKind) ||
        owner.ambiguous
      ) {
        continue;
      }
      if (
        conflictingGenerationAliases(
          incomingGenerationAliases,
          owner.generationAliases,
        )
      ) {
        this.recordGenerationIdentityConflict(
          scope,
          incomingAttempt.attemptId,
          owner.attemptId,
          [...incomingGenerationAliases],
          owner.generationAliases,
          context.observedAt,
        );
        continue;
      }
      mergeableOwnerIds.add(owner.attemptId);
    }
    const incomingCurrent = this.db
      .prepare("SELECT * FROM attempts WHERE attempt_id=? AND scope_key=?")
      .get(incomingAttempt.attemptId, key) as SqlRow | undefined;
    if (incomingGenerationAliases.size === 0) {
      const authoritativeOwnerIdentities = new Set(
        aliasOwners
          .map((owner) => generationIdentityKey(owner.generationAliases))
          .filter((identity) => identity !== ""),
      );
      if (authoritativeOwnerIdentities.size > 1) {
        this.recordCoverageGap(
          scope,
          {
            sourceKind: "attempt_merge",
            sourceId: incomingAttempt.attemptId,
            reason: "ambiguous_generation_identity",
            details: {
              incomingAttemptId: incomingAttempt.attemptId,
              generationIdentities: [...authoritativeOwnerIdentities].sort(),
            },
          },
          context.observedAt,
        );
        for (const attemptId of [...mergeableOwnerIds]) {
          const owner = aliasOwners.find((candidate) => candidate.attemptId === attemptId);
          if (owner && owner.generationAliases.length > 0) {
            mergeableOwnerIds.delete(attemptId);
          }
        }
      }
    }
    const canonicalAttemptId = this.selectCanonicalAttemptId(
      incomingAttempt,
      aliases,
      aliasOwners,
      mergeableOwnerIds,
      persistedIncomingGenerationAliases,
    );
    const current =
      canonicalAttemptId === incomingAttempt.attemptId
        ? incomingCurrent
        : (this.db
            .prepare("SELECT * FROM attempts WHERE attempt_id=? AND scope_key=?")
            .get(canonicalAttemptId, key) as SqlRow | undefined);
    const inheritedQuarantine = mergeQuarantines(
      parseQuarantine(current?.quarantine_json),
      quarantineFromWarnings(parseJsonArray(current?.warnings_json)),
      incomingAttempt.quarantine,
    );
    cleanAttempt = sanitizeAttempt(
      {
        ...incomingAttempt,
        attemptId: canonicalAttemptId,
        aliases,
        quarantine: inheritedQuarantine,
        warnings: uniqueStrings([
          ...incomingAttempt.warnings,
          ...inheritedQuarantine.warnings,
        ]),
      },
      context.observedAt,
    );
    if (
      canonicalAttemptId !== incomingAttempt.attemptId &&
      incomingCurrent &&
      !cannotRetireGenerationIdentity(
        persistedIncomingGenerationAliases,
        this.generationAliasesFor(scope, canonicalAttemptId),
      )
    ) {
      mergeableOwnerIds.add(incomingAttempt.attemptId);
    } else if (
      canonicalAttemptId !== incomingAttempt.attemptId &&
      incomingCurrent
    ) {
      this.recordGenerationIdentityConflict(
        scope,
        incomingAttempt.attemptId,
        canonicalAttemptId,
        persistedIncomingGenerationAliases,
        this.generationAliasesFor(scope, canonicalAttemptId),
        context.observedAt,
      );
    }
    if (mapping) {
      validateMappingVersion(mapping);
      this.assertMappingOwnerConsistency(mapping, scope);
      const projection = this.projectAttemptMapping(scope, cleanAttempt, mapping);
      if (
        current &&
        rawEvidenceMatchesCurrent(current, cleanAttempt) &&
        (projection.mapping === null ||
          projection.mapping.version === String(current.mapping_version))
      ) {
        cleanAttempt = preserveCurrentMappingProjection(
          cleanAttempt,
          current,
          mapping,
        );
        effectiveMapping = projection.mapping ?? undefined;
      } else {
        cleanAttempt = projection.attempt;
        effectiveMapping = projection.mapping ?? mapping;
      }
    }
    const payload = attemptPayload(cleanAttempt);
    assertNoSecrets(payload);
    const projectionFingerprint = fingerprint(attemptFingerprintPayload(cleanAttempt));
    const latestRevision = this.db
      .prepare(
        `
        SELECT revision, projection_fingerprint, collector_account_id
        FROM attempt_revisions
        WHERE attempt_id=? AND scope_key=?
        ORDER BY revision DESC
        LIMIT 1
        `,
      )
      .get(cleanAttempt.attemptId, key) as SqlRow | undefined;
    const currentMatches =
      current &&
      String(current.projection_fingerprint) === projectionFingerprint &&
      String(current.collector_account_id) === scope.collectorAccountId &&
      Number(current.tombstone) === 0;
    const latestHistoryMatches =
      latestRevision &&
      String(latestRevision.projection_fingerprint) === projectionFingerprint &&
      String(latestRevision.collector_account_id) === scope.collectorAccountId;
    const freshEvidence = !isOlderEvidence(context.observedAt, current?.updated_at);
    let status: "inserted" | "updated" | "deduplicated";
    let revision: number;
    let currentChanged = false;
    if (!current) {
      revision = Number(latestRevision?.revision ?? 0) + 1;
      this.insertAttemptRow(
        scope,
        cleanAttempt,
        revision,
        projectionFingerprint,
        context.observedAt,
      );
      this.insertAttemptRevision(
        scope,
        cleanAttempt,
        revision,
        projectionFingerprint,
        payload,
        context,
        "ingest",
      );
      status = "inserted";
      currentChanged = true;
    } else if (currentMatches) {
      revision = Number(current.revision);
      status = "deduplicated";
    } else {
      const reuseHistoryRevision = Boolean(latestHistoryMatches);
      revision = reuseHistoryRevision
        ? Number(latestRevision?.revision ?? 0)
        : Number(latestRevision?.revision ?? current.revision ?? 0) + 1;
      if (!reuseHistoryRevision && !latestRevision) {
        const priorAttempt = this.loadAttemptRecord(scope, current);
        this.insertAttemptRevision(
          scope,
          priorAttempt,
          Number(current.revision),
          String(current.projection_fingerprint),
          attemptPayload(priorAttempt),
          context,
          "prior_projection",
        );
      }
      if (!reuseHistoryRevision) {
        this.insertAttemptRevision(
          scope,
          cleanAttempt,
          revision,
          projectionFingerprint,
          payload,
          context,
          isOlderEvidence(context.observedAt, current.updated_at)
            ? "stale_ingest"
            : "ingest",
        );
      }
      if (freshEvidence) {
        this.updateAttemptRow(
          scope,
          cleanAttempt,
          revision,
          projectionFingerprint,
          context.observedAt,
        );
        currentChanged = true;
        status = "updated";
      } else {
        status = "deduplicated";
      }
    }
    if (!current || !isOlderEvidence(context.observedAt, current.updated_at)) {
      for (const duplicateAttemptId of mergeableOwnerIds) {
        if (duplicateAttemptId !== cleanAttempt.attemptId) {
          this.retireDuplicateAttempt(
            scope,
            duplicateAttemptId,
            cleanAttempt.attemptId,
            context.observedAt,
          );
        }
      }
    }
    const aliasConflicts = this.mergeAttemptLinks(
      scope,
      cleanAttempt,
      context.observedAt,
    );
    this.recordActivityProvenance(
      key,
      "attempt",
      cleanAttempt.attemptId,
      scope.collectorAccountId,
      context.observedAt,
    );
    if (mergeableOwnerIds.size > 0 && status === "inserted") {
      status = "updated";
    }
    if (currentChanged) {
      this.recordMappingHistory(
        scope,
        cleanAttempt,
        context.observedAt,
        mappingHistorySource(context.sourceKind),
        effectiveMapping
          ? {
              mapping: effectiveMapping,
            }
          : undefined,
      );
    }
    return { attemptId: cleanAttempt.attemptId, status, aliasConflicts };
  }

  private findAttemptAliasOwners(
    scope: LedgerScope,
    aliases: Array<[string, string]>,
  ): AttemptAliasOwner[] {
    const owners: AttemptAliasOwner[] = [];
    const statement = this.db.prepare(
      `
      SELECT a.alias_kind, a.alias_value, a.attempt_id,
             a.ambiguous, t.identity_basis, t.completed_answer, t.tombstone
      FROM attempt_aliases a
      JOIN attempts t
        ON t.scope_key=a.scope_key AND t.attempt_id=a.attempt_id
      WHERE a.scope_key=? AND a.alias_kind=? AND a.alias_value=?
        `,
    );
    const generationStatement = this.db.prepare(
      `
      SELECT alias_value
      FROM attempt_aliases
      WHERE scope_key=? AND attempt_id=? AND alias_kind='generation'
      ORDER BY alias_value
      `,
    );
    for (const [aliasKind, aliasValue] of aliases) {
      const row = statement.get(
        scopeKey(scope),
        aliasKind,
        aliasValue,
      ) as SqlRow | undefined;
      if (!row) {
        continue;
      }
      owners.push({
        aliasKind: String(row.alias_kind),
        aliasValue: String(row.alias_value),
        attemptId: String(row.attempt_id),
        identityBasis: String(row.identity_basis),
        completedAnswer: Number(row.completed_answer) === 1,
        tombstone: Number(row.tombstone) === 1,
        ambiguous: Number(row.ambiguous) === 1,
        generationAliases: generationStatement
          .all(scopeKey(scope), String(row.attempt_id))
          .map((item: unknown) => String((item as SqlRow).alias_value)),
      });
    }
    return owners;
  }

  private generationAliasesFor(scope: LedgerScope, attemptId: string): string[] {
    return this.db
      .prepare(
        `
        SELECT alias_value
        FROM attempt_aliases
        WHERE scope_key=? AND attempt_id=? AND alias_kind='generation'
        ORDER BY alias_value
        `,
      )
      .all(scopeKey(scope), attemptId)
      .map((row: unknown) => String((row as SqlRow).alias_value));
  }

  private selectCanonicalAttemptId(
    attempt: ReconstructedAttempt,
    aliases: Array<[string, string]>,
    aliasOwners: AttemptAliasOwner[],
    mergeableOwnerIds: Set<string>,
    persistedIncomingGenerationAliases: ReadonlyArray<string>,
  ): string {
    const incomingHasPromptAlias = aliases.some(
      ([aliasKind, aliasValue]) =>
        aliasKind === "prompt" &&
        !aliasOwners.some(
          (owner) =>
            owner.aliasKind === aliasKind &&
            owner.aliasValue === aliasValue &&
            owner.ambiguous,
        ),
    );
    const candidates = [
      {
        attemptId: attempt.attemptId,
        identityBasis: persistedIncomingGenerationAliases.length > 0
          ? "generation"
          : attempt.identityBasis,
        completedAnswer: attempt.completedAnswer,
        tombstone: false,
        hasPromptAlias: incomingHasPromptAlias,
      },
      ...[...mergeableOwnerIds].map((attemptId) => {
        const owners = aliasOwners.filter((owner) => owner.attemptId === attemptId);
        const first = owners[0];
        return {
          attemptId,
          identityBasis: first && first.generationAliases.length > 0
            ? "generation"
            : first?.identityBasis ?? "unresolved",
          completedAnswer: owners.some((owner) => owner.completedAnswer),
          tombstone: owners.every((owner) => owner.tombstone),
          hasPromptAlias: owners.some(
            (owner) => !owner.ambiguous && owner.aliasKind === "prompt",
          ),
        };
      }),
    ];
    return [...candidates].sort((left, right) => {
      return (
        (Number(left.tombstone) - Number(right.tombstone)) ||
        (identityRank(right.identityBasis) - identityRank(left.identityBasis)) ||
        (Number(right.hasPromptAlias) - Number(left.hasPromptAlias)) ||
        (Number(right.completedAnswer) - Number(left.completedAnswer)) ||
        left.attemptId.localeCompare(right.attemptId)
      );
    })[0]?.attemptId ?? attempt.attemptId;
  }

  private recordGenerationIdentityConflict(
    scope: LedgerScope,
    leftAttemptId: string,
    rightAttemptId: string,
    leftGenerationAliases: ReadonlyArray<string>,
    rightGenerationAliases: ReadonlyArray<string>,
    seenAt: string,
  ): void {
    this.recordCoverageGap(
      scope,
      {
        sourceKind: "attempt_merge",
        sourceId: `${leftAttemptId}:${rightAttemptId}`,
        reason: "generation_alias_conflict",
        details: {
          leftAttemptId,
          leftGenerationAliases: [...new Set(leftGenerationAliases)].sort(),
          rightAttemptId,
          rightGenerationAliases: [...new Set(rightGenerationAliases)].sort(),
        },
      },
      seenAt,
    );
  }

  private retireDuplicateAttempt(
    scope: LedgerScope,
    duplicateAttemptId: string,
    canonicalAttemptId: string,
    updatedAt: string,
  ): void {
    const duplicateGenerationAliases = this.generationAliasesFor(
      scope,
      duplicateAttemptId,
    );
    const canonicalGenerationAliases = this.generationAliasesFor(
      scope,
      canonicalAttemptId,
    );
    if (
      cannotRetireGenerationIdentity(
        duplicateGenerationAliases,
        canonicalGenerationAliases,
      )
    ) {
      this.recordGenerationIdentityConflict(
        scope,
        duplicateAttemptId,
        canonicalAttemptId,
        duplicateGenerationAliases,
        canonicalGenerationAliases,
        updatedAt,
      );
      return;
    }
    const key = scopeKey(scope);
    this.db
      .prepare("UPDATE attempt_aliases SET attempt_id=? WHERE scope_key=? AND attempt_id=?")
      .run(canonicalAttemptId, key, duplicateAttemptId);
    for (const provenance of this.activityProvenance(scope, "attempt", duplicateAttemptId)) {
      for (const seenAt of [provenance.first_seen_at, provenance.last_seen_at]) {
        this.recordActivityProvenance(key, "attempt", canonicalAttemptId,
          String(provenance.collector_account_id), String(seenAt));
      }
    }

    this.db
      .prepare(
        `
        INSERT OR IGNORE INTO attempt_evidence(
          attempt_id, evidence_kind, evidence_id, scope_key
        )
        SELECT ?, evidence_kind, evidence_id, scope_key
        FROM attempt_evidence
        WHERE attempt_id=? AND scope_key=?
        `,
      )
      .run(canonicalAttemptId, duplicateAttemptId, key);
    this.db
      .prepare("DELETE FROM attempt_evidence WHERE attempt_id=? AND scope_key=?")
      .run(duplicateAttemptId, key);

    const row = this.db
      .prepare("SELECT warnings_json FROM attempts WHERE attempt_id=? AND scope_key=?")
      .get(duplicateAttemptId, key) as SqlRow | undefined;
    const marker = `retired_duplicate:${canonicalAttemptId}`;
    const warnings = parseJsonArray(row?.warnings_json);
    if (!warnings.includes(marker)) {
      warnings.push(marker);
    }
    this.db
      .prepare(
        `
        UPDATE attempts
        SET tombstone=1, warnings_json=?, updated_at=?
        WHERE attempt_id=? AND scope_key=?
        `,
      )
      .run(JSON.stringify(warnings.sort()), updatedAt, duplicateAttemptId, key);
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
      .map((row: unknown) => {
        const item = row as SqlRow;
        const payload = rowToAttemptPayload(item);
        payload.aliases = this.aliasesFor(scope, String(item.attempt_id));
        payload.evidenceMessageIds = this.evidenceFor(
          scope,
          String(item.attempt_id),
        );
        return payload;
      });
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
        return {
          ...item,
          payload: parseJsonObject(item.payload_json),
          quarantine: parseQuarantine(item.quarantine_json),
        };
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
      .map((row: unknown) => {
        const item = row as SqlRow;
        return {
          ...item,
          warnings: parseJsonArray(item.warnings_json),
          provenance: parseJsonObject(item.provenance_json),
        };
      });
  }

  saveModelMapping(mapping: ModelMappingVersion): void {
    const candidate = normalizeMappingVersion(mapping);
    validateMappingVersion(candidate);
    assertNoSecrets(candidate);
    this.assertMappingOwnerConsistency(candidate);
    const existingRow = this.db
      .prepare("SELECT * FROM model_mapping_versions WHERE version=?")
      .get(candidate.version) as SqlRow | undefined;
    if (existingRow) {
      const existing = this.modelMapping(candidate.version);
      if (mappingFingerprint(existing) === mappingFingerprint(candidate)) {
        return;
      }
      if (isMappingPublished(existing)) {
        throw new LedgerError(
          `published model mapping version is immutable: ${candidate.version}`,
        );
      }
    }

    const values = [
      JSON.stringify(candidate.canonicalFamilies),
      JSON.stringify(candidate.rules),
      candidate.reviewStatus,
      candidate.source,
      candidate.createdAt,
      candidate.reviewedAt ?? null,
      candidate.reviewedBy ?? null,
      candidate.changeKind ?? "prospective",
      candidate.validFrom ?? null,
      candidate.validUntil ?? null,
      candidate.publishedAt ?? null,
      candidate.supersedesVersion ?? null,
      candidate.correctionOfVersion ?? null,
      JSON.stringify(candidate.provenance ?? {}),
      JSON.stringify(candidate.warnings ?? []),
    ];
    if (existingRow) {
      this.db
        .prepare(
          `
          UPDATE model_mapping_versions SET
            canonical_families_json=?, rules_json=?, review_status=?,
            source=?, created_at=?, reviewed_at=?, reviewed_by=?,
            change_kind=?, valid_from=?, valid_until=?, published_at=?,
            supersedes_version=?, correction_of_version=?,
            provenance_json=?, warnings_json=?
          WHERE version=?
          `,
        )
        .run(...values, candidate.version);
      return;
    }
    this.db
      .prepare(
        `
        INSERT INTO model_mapping_versions(
          version, canonical_families_json, rules_json, review_status,
          source, created_at, reviewed_at, reviewed_by, change_kind,
          valid_from, valid_until, published_at, supersedes_version,
          correction_of_version, provenance_json, warnings_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        `,
      )
      .run(candidate.version, ...values);
  }

  modelMapping(version: string): ModelMappingVersion {
    const row = this.db
      .prepare("SELECT * FROM model_mapping_versions WHERE version=?")
      .get(version) as SqlRow | undefined;
    if (!row) {
      throw new LedgerError(`model mapping not found: ${version}`);
    }
    return normalizeMappingVersion({
      version: String(row.version),
      canonicalFamilies: parseJsonArray(row.canonical_families_json),
      rules: parseJsonUnknownArray(row.rules_json) as ModelMappingVersion["rules"],
      reviewStatus: row.review_status as ModelMappingVersion["reviewStatus"],
      source: String(row.source),
      createdAt: String(row.created_at),
      reviewedAt: nullableString(row.reviewed_at),
      reviewedBy: nullableString(row.reviewed_by),
      changeKind: (row.change_kind as ModelMappingVersion["changeKind"]) ?? "prospective",
      validFrom: nullableString(row.valid_from),
      validUntil: nullableString(row.valid_until),
      publishedAt: nullableString(row.published_at),
      supersedesVersion: nullableString(row.supersedes_version),
      correctionOfVersion: nullableString(row.correction_of_version),
      provenance: parseJsonObject(row.provenance_json),
      warnings: parseJsonArray(row.warnings_json),
    });
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
    this.assertMappingOwnerConsistency(mapping, scope);
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

  private assertMappingOwnerConsistency(
    mapping: ModelMappingVersion,
    extraScope?: LedgerScope,
  ): void {
    try {
      validateMappingOwnerConsistency(mapping, this.mappingOwnerBindings(extraScope));
    } catch (error) {
      if (error instanceof ModelMappingError) {
        throw new LedgerError(error.message);
      }
      throw error;
    }
  }

  private mappingOwnerBindings(extraScope?: LedgerScope): MappingOwnerBinding[] {
    const bindings = [
      ...this.listAccounts(),
      ...(extraScope ? [extraScope] : []),
    ];
    const seen = new Set<string>();
    return bindings
      .map((scope) => ({
        collectorAccountId: scope.collectorAccountId,
        canonicalOwnerKey: scopeKey(scope),
      }))
      .filter((binding) => {
        const key = `${binding.collectorAccountId}\u0000${binding.canonicalOwnerKey}`;
        if (seen.has(key)) {
          return false;
        }
        seen.add(key);
        return true;
      });
  }

  private projectAttemptMapping(
    scope: LedgerScope,
    attempt: ReconstructedAttempt,
    requestedMapping: ModelMappingVersion,
  ): {
    attempt: ReconstructedAttempt;
    mapping: ModelMappingVersion | null;
  } {
    const applicationTime = attempt.attemptTime;
    const requestedResolution = resolveModelEvidence(
      { slug: null, mode: null, reasoningEffort: null },
      requestedMapping,
      scope.collectorAccountId,
      { at: applicationTime },
    );
    const effectiveMapping = requestedResolution.applied
      ? requestedMapping
      : this.findApplicableStoredMapping(
          scope,
          requestedMapping,
          applicationTime,
        );
    if (effectiveMapping === null) {
      return {
        attempt: {
          ...attempt,
          requestedFamily: null,
          recordedFinalFamily: null,
          resolvedFamily: null,
          mappingVersion: requestedMapping.version,
          warnings: uniqueStrings([
            ...attempt.warnings,
            ...requestedResolution.warnings,
            `mapping_unresolved:${requestedMapping.version}`,
          ]),
        },
        mapping: null,
      };
    }

    const requested = resolveModelEvidence(
      {
        slug: attempt.requestedModelRaw,
        mode: attempt.requestedModeRaw,
        reasoningEffort: attempt.requestedReasoningEffortRaw,
      },
      effectiveMapping,
      scope.collectorAccountId,
      { at: applicationTime },
    );
    const recordedFinal = resolveModelEvidence(
      {
        slug: attempt.recordedFinalModelRaw,
        mode: null,
        reasoningEffort: null,
      },
      effectiveMapping,
      scope.collectorAccountId,
      { at: applicationTime },
    );
    const resolved = resolveModelEvidence(
      {
        slug: attempt.resolvedModelRaw,
        mode: null,
        reasoningEffort: null,
      },
      effectiveMapping,
      scope.collectorAccountId,
      { at: applicationTime },
    );
    const fallbackWarnings =
      effectiveMapping.version === requestedMapping.version ||
      effectiveMapping.version === attempt.mappingVersion
        ? []
        : [`mapping_fallback:${effectiveMapping.version}`];
    return {
      attempt: {
        ...attempt,
        requestedFamily: requested.family,
        recordedFinalFamily: recordedFinal.family,
        resolvedFamily: resolved.family,
        mappingVersion: effectiveMapping.version,
        warnings: uniqueStrings([
          ...attempt.warnings,
          ...requested.warnings,
          ...recordedFinal.warnings,
          ...resolved.warnings,
          ...fallbackWarnings,
        ]),
      },
      mapping: effectiveMapping,
    };
  }

  private findApplicableStoredMapping(
    scope: LedgerScope,
    requestedMapping: ModelMappingVersion,
    applicationTime: string | null,
  ): ModelMappingVersion | null {
    return this.modelMappings()
      .filter((mapping) => mapping.version !== requestedMapping.version)
      .filter((mapping) =>
        resolveModelEvidence(
          { slug: null, mode: null, reasoningEffort: null },
          mapping,
          scope.collectorAccountId,
          { at: applicationTime },
        ).applied,
      )
      .sort(compareMappingRecency)
      .at(-1) ?? null;
  }

  reclassifyAttempts(
    scope: LedgerScope,
    mapping: ModelMappingVersion,
    recordedAt: string,
    source = "aggregate_rebuild",
  ): number {
    const candidate = normalizeMappingVersion(mapping);
    validateMappingVersion(candidate);
    assertNoSecrets(candidate);
    this.assertMappingOwnerConsistency(candidate, scope);
    if (!isMappingPublished(candidate)) {
      return 0;
    }
    let changed = 0;
    for (const row of this.listAttempts(scope)) {
      const attempt = rowToReconstructedAttempt(row, scope);
      const projection = this.projectAttemptMapping(scope, attempt, candidate);
      if (projection.mapping === null) {
        continue;
      }
      const effectiveMapping = projection.mapping ?? candidate;
      const mappedAttempt = projection.attempt;
      const projectionChanged =
        mappedAttempt.requestedFamily !== attempt.requestedFamily ||
        mappedAttempt.recordedFinalFamily !== attempt.recordedFinalFamily ||
        mappedAttempt.resolvedFamily !== attempt.resolvedFamily ||
        mappedAttempt.mappingVersion !== attempt.mappingVersion;
      const reclassificationWarning =
        projection.mapping?.version === candidate.version || projectionChanged
          ? [`mapping_reclassified:${effectiveMapping.version}`]
          : [];
      const nextWarnings = uniqueStrings([
        ...mappedAttempt.warnings,
        ...(candidate.warnings ?? []),
        ...reclassificationWarning,
      ]);
      const current = [
        nullableString(row.requestedFamily),
        nullableString(row.recordedFinalFamily),
        nullableString(row.resolvedFamily),
        String(row.mappingVersion),
        [...attempt.warnings].sort(),
      ];
      const next = [
        mappedAttempt.requestedFamily,
        mappedAttempt.recordedFinalFamily,
        mappedAttempt.resolvedFamily,
        mappedAttempt.mappingVersion,
        [...nextWarnings].sort(),
      ];
      if (JSON.stringify(current) === JSON.stringify(next)) {
        continue;
      }
      const projected = {
        ...mappedAttempt,
        warnings: nextWarnings,
      };
      const payload = attemptPayload(projected);
      assertNoSecrets(payload);
      const nextRevision = this.nextAttemptRevision(scope, attempt.attemptId) + 1;
      const projectionFingerprint = fingerprint(attemptFingerprintPayload(projected));
      const provenance = {
        source,
        mapping_version: effectiveMapping.version,
        ...(effectiveMapping.version === candidate.version
          ? {}
          : { requested_mapping_version: candidate.version }),
        previous_mapping_version: attempt.mappingVersion,
        change_kind: effectiveMapping.changeKind ?? "prospective",
        valid_from: effectiveMapping.validFrom ?? null,
        valid_until: effectiveMapping.validUntil ?? null,
        recorded_at: recordedAt,
        ...(effectiveMapping.provenance ?? {}),
      };
      this.recordMappingHistory(scope, attempt, recordedAt, "prior_projection", {
        warnings: attempt.warnings,
        provenance: {
          source: "prior_projection",
          mapping_version: attempt.mappingVersion,
          recorded_at: recordedAt,
        },
      });
      this.recordMappingHistory(scope, projected, recordedAt, source, {
        mapping: effectiveMapping,
        warnings: nextWarnings,
        provenance,
      });
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
          runId: `mapping:${candidate.version}`,
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

  private reconstructionMessages(scope: LedgerScope, conversationId: string): LedgerMessage[] {
    const origin = this.conversationOrigin(scope, conversationId);
    return this.messagesFor(scope, conversationId).map((message) =>
      origin !== null && message.origin === null ? { ...message, origin } : message,
    );
  }

  rebuildAttemptsFromMessages(
    scope: LedgerScope,
    mapping: ModelMappingVersion,
    recordedAt: string,
  ): number {
    validateMappingVersion(mapping);
    this.assertMappingOwnerConsistency(mapping, scope);
    let changed = 0;
    for (const conversationId of this.conversations(scope)) {
      const messages = this.reconstructionMessages(scope, conversationId);
      const attempts = reconstructAttempts(messages, {
        scope,
        conversationId,
        mapping,
      });
      for (const attempt of attempts) {
        const result = this.upsertAttempt(
          scope,
          applyMessageQuarantine(attempt, messages),
          {
            runId: `rebuild:${mapping.version}`,
            observedAt: recordedAt,
            sourceKind: "aggregate_rebuild",
            sourceId: conversationId,
            schemaVersion: "ledger-v2",
          },
          mapping,
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
          state=CASE
            WHEN excluded.last_seen_at >= coverage_gaps.last_seen_at
            THEN excluded.state
            ELSE coverage_gaps.state
          END,
          last_seen_at=MAX(coverage_gaps.last_seen_at, excluded.last_seen_at),
          details_json=CASE
            WHEN excluded.last_seen_at >= coverage_gaps.last_seen_at
            THEN excluded.details_json
            ELSE coverage_gaps.details_json
          END
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

  resolveCoverageGaps(
    scope: LedgerScope,
    sourceKind: string,
    sourceId: string,
    seenAt: string,
  ): void {
    this.db.prepare(`
      UPDATE coverage_gaps SET state='resolved', last_seen_at=?
      WHERE scope_key=? AND source_kind=? AND source_id=? AND state='open'
        AND (last_seen_at IS NULL OR last_seen_at <= ?)
    `).run(seenAt, scopeKey(scope), sourceKind, sourceId, seenAt);
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

  activityProvenance(
    scope: LedgerScope,
    activityKind: string,
    activityId: string,
  ): Array<Record<string, unknown>> {
    return this.db
      .prepare(
        `
        SELECT *
        FROM activity_provenance
        WHERE scope_key=? AND activity_kind=? AND activity_id=?
        ORDER BY first_seen_at, collector_account_id
        `,
      )
      .all(scopeKey(scope), activityKind, activityId)
      .map((row: unknown) => row as SqlRow);
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
          projection_fingerprint, warnings_json, quarantine_json, tombstone, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
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
        JSON.stringify(normalizeQuarantine(attempt.quarantine)),
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
          revision=?, projection_fingerprint=?, warnings_json=?, quarantine_json=?,
          tombstone=0, updated_at=?
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
        JSON.stringify(normalizeQuarantine(attempt.quarantine)),
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
          attempt_id, revision, scope_key, collector_account_id,
          projection_fingerprint, payload_json, recorded_at, source, quarantine_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        `,
      )
      .run(
        String(attempt.attemptId ?? ""),
        revision,
        scopeKey(scope),
        scope.collectorAccountId,
        projectionFingerprint,
        JSON.stringify(payload),
        context.observedAt,
        source,
        JSON.stringify(
          normalizeQuarantine(
            "quarantine" in attempt ? attempt.quarantine : payload.quarantine,
          ),
        ),
      );
  }

  private nextAttemptRevision(scope: LedgerScope, attemptId: string): number {
    const row = this.db
      .prepare(
        `
        SELECT MAX(revision) AS revision
        FROM attempt_revisions
        WHERE scope_key=? AND attempt_id=?
        `,
      )
      .get(scopeKey(scope), attemptId) as SqlRow | undefined;
    return Number(row?.revision ?? 0);
  }

  private aliasesFor(
    scope: LedgerScope,
    attemptId: string,
  ): Array<[string, string]> {
    return this.db
      .prepare(
        `
        SELECT alias_kind, alias_value
        FROM attempt_aliases
        WHERE scope_key=? AND attempt_id=?
        ORDER BY alias_kind, alias_value
        `,
      )
      .all(scopeKey(scope), attemptId)
      .map((row: unknown) => {
        const item = row as SqlRow;
        return [String(item.alias_kind), String(item.alias_value)] as [string, string];
      });
  }

  private evidenceFor(scope: LedgerScope, attemptId: string): string[] {
    return this.db
      .prepare(
        `
        SELECT evidence_id
        FROM attempt_evidence
        WHERE scope_key=? AND attempt_id=? AND evidence_kind='message'
        ORDER BY evidence_id
        `,
      )
      .all(scopeKey(scope), attemptId)
      .map((row: unknown) => String((row as SqlRow).evidence_id));
  }

  private loadAttemptRecord(
    scope: LedgerScope,
    row: SqlRow,
  ): ReconstructedAttempt {
    const payload = rowToAttemptPayload(row);
    payload.aliases = this.aliasesFor(scope, String(row.attempt_id));
    payload.evidenceMessageIds = this.evidenceFor(scope, String(row.attempt_id));
    return rowToReconstructedAttempt(payload, scope);
  }

  private mergeAttemptLinks(
    scope: LedgerScope,
    attempt: ReconstructedAttempt,
    seenAt: string,
  ): number {
    const key = scopeKey(scope);
    let aliasConflicts = 0;
    for (const [kind, value] of attempt.aliases) {
      const aliasKind = sanitizeToken(kind);
      const aliasValue = sanitizeToken(value);
      if (aliasKind === null || aliasValue === null) {
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
        .get(key, aliasKind, aliasValue) as SqlRow | undefined;
      if (existing && existing.attempt_id !== attempt.attemptId) {
        aliasConflicts += 1;
        this.db
          .prepare(
            `
            UPDATE attempt_aliases
            SET ambiguous=1, last_seen_at=MAX(last_seen_at, ?)
            WHERE scope_key=? AND alias_kind=? AND alias_value=?
            `,
          )
          .run(seenAt, key, aliasKind, aliasValue);
        this.recordCoverageGap(
          scope,
          {
            sourceKind: "attempt_alias",
            sourceId: `${aliasKind}:${aliasValue}`,
            reason: "alias_collision",
            details: {
              existingAttemptId: existing.attempt_id,
              incomingAttemptId: attempt.attemptId,
              aliasKind,
            },
          },
          seenAt,
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
            collector_account_id=excluded.collector_account_id,
            last_seen_at=MAX(attempt_aliases.last_seen_at, excluded.last_seen_at)
          `,
        )
        .run(
          key,
          scope.collectorAccountId,
          scope.provider,
          scope.providerUserId,
          scope.workspaceId,
          scope.quotaOwnerId,
          aliasKind,
          aliasValue,
          attempt.attemptId,
          seenAt,
          seenAt,
        );
    }
    for (const messageId of attempt.evidenceMessageIds) {
      this.attachEvidence(attempt.attemptId, "message", messageId, scope);
    }
    return aliasConflicts;
  }

  private recordActivityProvenance(
    activityScopeKey: string,
    activityKind: string,
    activityId: string,
    collectorAccountId: string,
    seenAt: string,
  ): void {
    this.db
      .prepare(
        `
        INSERT INTO activity_provenance(
          scope_key, activity_kind, activity_id, collector_account_id,
          first_seen_at, last_seen_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(scope_key, activity_kind, activity_id, collector_account_id)
        DO UPDATE SET
          first_seen_at=MIN(activity_provenance.first_seen_at, excluded.first_seen_at),
          last_seen_at=MAX(activity_provenance.last_seen_at, excluded.last_seen_at)
        `,
      )
      .run(
        activityScopeKey,
        activityKind,
        activityId,
        collectorAccountId,
        seenAt,
        seenAt,
      );
  }

  private recordMappingHistory(
    scope: LedgerScope,
    attempt: ReconstructedAttempt,
    recordedAt: string,
    source: string,
    options: {
      mapping?: ModelMappingVersion;
      warnings?: string[];
      provenance?: Record<string, unknown>;
    } = {},
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
          recorded_at, source, change_kind, valid_from, valid_until,
          warnings_json, provenance_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        options.mapping?.changeKind ?? "prospective",
        options.mapping?.validFrom ?? null,
        options.mapping?.validUntil ?? null,
        JSON.stringify(options.warnings ?? attempt.warnings),
        JSON.stringify(options.provenance ?? {
          source,
          mapping_version: attempt.mappingVersion,
          recorded_at: recordedAt,
        }),
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

function sanitizeMessage(
  record: MessageRecord,
  observedAt?: string,
): LedgerMessage {
  const conversationId = sanitizeToken(record.conversationId) ?? "unknown-conversation";
  const messageId = sanitizeToken(record.messageId) ?? "unknown-message";
  const quarantine = mergeQuarantines(
    normalizeQuarantine((record as Partial<LedgerMessage>).quarantine),
    observedAt === undefined
      ? clearQuarantine()
      : quarantineForTimestampFields(
          [{ field: "createdAt", value: record.createdAt, messageId }],
          observedAt,
        ),
  );
  const clean: LedgerMessage = {
    conversationId,
    messageId,
    nodeId: sanitizeToken(record.nodeId) ?? null,
    parentId: sanitizeToken(record.parentId) ?? null,
    children: record.children
      .map((child) => sanitizeToken(child))
      .filter((child): child is string => child !== null),
    role: sanitizeToken(record.role) ?? null,
    channel: sanitizeToken(record.channel) ?? null,
    createdAt: safeTimestamp(record.createdAt, observedAt),
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
    quarantine,
  };
  return clean;
}

function sanitizeAttempt(
  attempt: ReconstructedAttempt,
  observedAt: string,
): ReconstructedAttempt {
  const quarantine = mergeQuarantines(
    attempt.quarantine,
    quarantineForTimestampFields(
      [
        { field: "attemptTime", value: attempt.attemptTime },
        { field: "earliestPossibleAt", value: attempt.earliestPossibleAt },
        { field: "latestPossibleAt", value: attempt.latestPossibleAt },
      ],
      observedAt,
    ),
  );
  const aliases = attempt.aliases
    .map(([kind, value]) => {
      const cleanKind = sanitizeToken(kind);
      const cleanValue = sanitizeToken(value);
      return cleanKind !== null && cleanValue !== null
        ? [cleanKind, cleanValue] as [string, string]
        : null;
    })
    .filter((alias): alias is [string, string] => alias !== null);
  const evidenceMessageIds = attempt.evidenceMessageIds
    .map((messageId) => sanitizeToken(messageId))
    .filter((messageId): messageId is string => messageId !== null);
  const warnings = uniqueStrings([
    ...attempt.warnings.filter((warning): warning is string => typeof warning === "string"),
    ...quarantine.warnings,
  ]);
  return {
    ...attempt,
    attemptTime: safeTimestamp(attempt.attemptTime, observedAt),
    earliestPossibleAt: safeTimestamp(attempt.earliestPossibleAt, observedAt),
    latestPossibleAt: safeTimestamp(attempt.latestPossibleAt, observedAt),
    aliases,
    evidenceMessageIds,
    warnings,
    quarantine,
  };
}

function rawEvidenceMatchesCurrent(
  current: SqlRow,
  attempt: ReconstructedAttempt,
): boolean {
  return (
    nullableString(current.requested_model_raw) === attempt.requestedModelRaw &&
    nullableString(current.requested_mode_raw) === attempt.requestedModeRaw &&
    nullableString(current.requested_reasoning_effort_raw) ===
      attempt.requestedReasoningEffortRaw &&
    nullableString(current.recorded_final_model_raw) === attempt.recordedFinalModelRaw &&
    nullableString(current.resolved_model_raw) === attempt.resolvedModelRaw
  );
}

function preserveCurrentMappingProjection(
  attempt: ReconstructedAttempt,
  current: SqlRow,
  mapping: ModelMappingVersion,
): ReconstructedAttempt {
  return {
    ...attempt,
    requestedFamily: nullableString(current.requested_family),
    recordedFinalFamily: nullableString(current.recorded_final_family),
    resolvedFamily: nullableString(current.resolved_family),
    mappingVersion: String(current.mapping_version),
    warnings: uniqueStrings([
      ...attempt.warnings,
      ...parseJsonArray(current.warnings_json).filter(
        (warning) => warning.startsWith("mapping_") || mapping.warnings?.includes(warning),
      ),
    ]),
  };
}

function applyMessageQuarantine(
  attempt: ReconstructedAttempt,
  messages: ReadonlyArray<LedgerMessage>,
): ReconstructedAttempt {
  const evidenceMessageIds = new Set(attempt.evidenceMessageIds);
  const quarantine = mergeQuarantines(
    attempt.quarantine,
    ...messages
      .filter((message) => evidenceMessageIds.has(message.messageId))
      .map((message) => message.quarantine),
  );
  return {
    ...attempt,
    warnings: uniqueStrings([...attempt.warnings, ...quarantine.warnings]),
    quarantine,
  };
}

function messagePayload(record: LedgerMessage): Record<string, unknown> {
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

function messageRevisionPayload(record: LedgerMessage): Record<string, unknown> {
  return {
    ...messagePayload(record),
    quarantine: normalizeQuarantine(record.quarantine),
  };
}

function messageFingerprintPayload(record: LedgerMessage): Record<string, unknown> {
  return fingerprintPayload(
    messageRevisionPayload(record),
    record.quarantine,
  );
}

function attemptFingerprintPayload(attempt: ReconstructedAttempt): Record<string, unknown> {
  return fingerprintPayload(attemptPayload(attempt), attempt.quarantine);
}

function observationFingerprintPayload(
  payload: Record<string, unknown>,
  quarantine: LedgerQuarantine,
): Record<string, unknown> {
  const stablePayload = stableObservationPayload(payload);
  const stableQuarantine = quarantineFingerprint(quarantine);
  return isEmptyQuarantineFingerprint(stableQuarantine)
    ? stablePayload
    : { payload: stablePayload, quarantine: stableQuarantine };
}

function fingerprintPayload(
  payload: Record<string, unknown>,
  quarantine: LedgerQuarantine | null | undefined,
): Record<string, unknown> {
  const stableQuarantine = quarantineFingerprint(quarantine);
  if (isEmptyQuarantineFingerprint(stableQuarantine)) {
    const { quarantine: _quarantine, ...legacyPayload } = payload;
    return legacyPayload;
  }
  return { ...payload, quarantine: stableQuarantine };
}

function isEmptyQuarantineFingerprint(
  quarantine: Record<string, unknown>,
): boolean {
  return quarantine.state === "clear" &&
    Array.isArray(quarantine.warnings) &&
    quarantine.warnings.length === 0 &&
    Array.isArray(quarantine.timestamps) &&
    quarantine.timestamps.length === 0;
}

function quarantineFingerprint(
  quarantine: LedgerQuarantine | null | undefined,
): Record<string, unknown> {
  const normalized = mergeQuarantines(quarantine);
  return {
    state: normalized.state,
    warnings: normalized.warnings,
    timestamps: normalized.timestamps.map(({ field, value, messageId }) =>
      messageId === undefined
        ? { field, value }
        : { field, value, messageId }),
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
    | "quarantine"
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
    quarantine: normalizeQuarantine(attempt.quarantine),
  };
}

function uniqueAttemptAliases(
  aliases: ReadonlyArray<[string, string]>,
): Array<[string, string]> {
  const seen = new Set<string>();
  const unique: Array<[string, string]> = [];
  for (const [kind, value] of aliases) {
    const key = `${kind}\u0000${value}`;
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    unique.push([kind, value]);
  }
  return unique;
}

function identityRank(identityBasis: string): number {
  switch (identityBasis) {
    case "generation":
      return 4;
    case "request":
      return 3;
    case "prompt":
      return 2;
    case "provisional":
      return 1;
    default:
      return 0;
  }
}

function conflictingGenerationAliases(
  incoming: ReadonlySet<string>,
  existing: ReadonlyArray<string>,
): boolean {
  const stored = new Set(existing);
  if (incoming.size === 0 || stored.size === 0) {
    return false;
  }
  return incoming.size !== stored.size || [...incoming].some((alias) => !stored.has(alias));
}

function cannotRetireGenerationIdentity(
  duplicate: ReadonlyArray<string>,
  canonical: ReadonlyArray<string>,
): boolean {
  const duplicateSet = new Set(duplicate);
  const canonicalSet = new Set(canonical);
  if (duplicateSet.size === 0) {
    return false;
  }
  return canonicalSet.size === 0 ||
    conflictingGenerationAliases(duplicateSet, [...canonicalSet]);
}

function generationIdentityKey(aliases: ReadonlyArray<string>): string {
  return [...new Set(aliases)].sort().join("\u0000");
}

function messageFromRow(row: SqlRow): LedgerMessage {
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
    quarantine: parseQuarantine(row.quarantine_json),
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
    quarantine: parseQuarantine(row.quarantine_json),
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
    aliases: parseAliasArray(row.aliases),
    evidenceMessageIds: parseJsonArray(row.evidenceMessageIds),
    revision: Number(row.revision ?? 1),
    warnings: parseJsonArray(row.warnings),
    quarantine: parseQuarantine(row.quarantine),
    scope,
  };
}

function mappingFingerprint(mapping: ModelMappingVersion): string {
  const normalized = normalizeMappingVersion(mapping);
  return canonicalJson({
    version: normalized.version,
    canonicalFamilies: normalized.canonicalFamilies,
    rules: normalized.rules,
    reviewStatus: normalized.reviewStatus,
    source: normalized.source,
    createdAt: normalized.createdAt,
    reviewedAt: normalized.reviewedAt ?? null,
    reviewedBy: normalized.reviewedBy ?? null,
    changeKind: normalized.changeKind ?? "prospective",
    validFrom: normalized.validFrom ?? null,
    validUntil: normalized.validUntil ?? null,
    publishedAt: normalized.publishedAt ?? null,
    supersedesVersion: normalized.supersedesVersion ?? null,
    correctionOfVersion: normalized.correctionOfVersion ?? null,
    provenance: normalized.provenance ?? {},
    warnings: normalized.warnings ?? [],
  });
}

function compareMappingRecency(
  left: ModelMappingVersion,
  right: ModelMappingVersion,
): number {
  return (
    Number(left.changeKind === "historical_correction") -
      Number(right.changeKind === "historical_correction") ||
    (left.validFrom ?? "").localeCompare(right.validFrom ?? "") ||
    left.createdAt.localeCompare(right.createdAt) ||
    (left.publishedAt ?? "").localeCompare(right.publishedAt ?? "") ||
    left.version.localeCompare(right.version)
  );
}

function mappingHistorySource(sourceKind: string): string {
  return sourceKind === "aggregate_rebuild" ? "aggregate_rebuild" : "ingest";
}

function uniqueStrings(values: ReadonlyArray<string>): string[] {
  return [...new Set(values)].sort();
}

const QUARANTINE_WARNING = "future_timestamp_quarantined";
const QUARANTINE_FIELDS = new Set<QuarantinedTimestampEvidence["field"]>([
  "createdAt",
  "updatedAt",
  "attemptTime",
  "earliestPossibleAt",
  "latestPossibleAt",
]);

function clearQuarantine(): LedgerQuarantine {
  return {
    state: "clear",
    warnings: [],
    timestamps: [],
  };
}

function quarantineFromWarnings(warnings: ReadonlyArray<string>): LedgerQuarantine {
  const retained = warnings.filter((warning) => warning === QUARANTINE_WARNING);
  return retained.length === 0
    ? clearQuarantine()
    : {
        state: "quarantined",
        warnings: uniqueStrings(retained),
        timestamps: [],
      };
}

function quarantineForTimestampFields(
  fields: ReadonlyArray<{
    field: QuarantinedTimestampEvidence["field"];
    value: string | null;
    messageId?: string;
  }>,
  observedAt: string,
): LedgerQuarantine {
  const timestamps: QuarantinedTimestampEvidence[] = [];
  for (const field of fields) {
    if (field.value === null || !isFutureTimestamp(field.value, observedAt)) {
      continue;
    }
    const evidence: QuarantinedTimestampEvidence = {
      field: field.field,
      value: field.value,
      observedAt,
    };
    if (field.messageId !== undefined) {
      evidence.messageId = field.messageId;
    }
    timestamps.push(evidence);
  }
  return timestamps.length === 0
    ? clearQuarantine()
    : {
        state: "quarantined",
        warnings: [QUARANTINE_WARNING],
        timestamps,
      };
}

function mergeQuarantines(
  ...values: Array<LedgerQuarantine | null | undefined>
): LedgerQuarantine {
  const normalized = values.map((value) => normalizeQuarantine(value));
  const timestamps = new Map<string, QuarantinedTimestampEvidence>();
  for (const quarantine of normalized) {
    for (const timestamp of quarantine.timestamps) {
      const key = canonicalJson({
        field: timestamp.field,
        value: timestamp.value,
        messageId: timestamp.messageId ?? null,
      });
      if (!timestamps.has(key)) {
        timestamps.set(key, timestamp);
      }
    }
  }
  const mergedTimestamps = [...timestamps.values()].sort((left, right) =>
    left.field.localeCompare(right.field) ||
    left.value.localeCompare(right.value) ||
    left.observedAt.localeCompare(right.observedAt) ||
    (left.messageId ?? "").localeCompare(right.messageId ?? ""),
  );
  const warnings = uniqueStrings(normalized.flatMap((value) => value.warnings));
  return {
    state: normalized.some((value) => value.state === "quarantined") ||
      mergedTimestamps.length > 0 ||
      warnings.includes(QUARANTINE_WARNING)
      ? "quarantined"
      : "clear",
    warnings,
    timestamps: mergedTimestamps,
  };
}

function parseQuarantine(value: unknown): LedgerQuarantine {
  if (typeof value === "string") {
    return normalizeQuarantine(parseJsonObject(value));
  }
  return normalizeQuarantine(value);
}

function normalizeQuarantine(value: unknown): LedgerQuarantine {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    return clearQuarantine();
  }
  const raw = value as Record<string, unknown>;
  const warnings = Array.isArray(raw.warnings)
    ? raw.warnings.filter((warning): warning is string => typeof warning === "string")
    : [];
  const timestamps: QuarantinedTimestampEvidence[] = [];
  if (Array.isArray(raw.timestamps)) {
    for (const item of raw.timestamps) {
      if (item === null || typeof item !== "object" || Array.isArray(item)) {
        continue;
      }
      const candidate = item as Record<string, unknown>;
      if (
        !isQuarantineTimestampField(candidate.field) ||
        typeof candidate.value !== "string" ||
        !Number.isFinite(Date.parse(candidate.value)) ||
        typeof candidate.observedAt !== "string" ||
        !Number.isFinite(Date.parse(candidate.observedAt))
      ) {
        continue;
      }
      const evidence: QuarantinedTimestampEvidence = {
        field: candidate.field,
        value: candidate.value,
        observedAt: candidate.observedAt,
      };
      const messageId = sanitizeToken(nullableString(candidate.messageId));
      if (messageId !== null) {
        evidence.messageId = messageId;
      }
      timestamps.push(evidence);
    }
  }
  const state = raw.state === "quarantined" ||
    timestamps.length > 0 ||
    warnings.includes(QUARANTINE_WARNING)
    ? "quarantined"
    : "clear";
  return {
    state,
    warnings: uniqueStrings(warnings),
    timestamps,
  };
}

function isQuarantineTimestampField(
  value: unknown,
): value is QuarantinedTimestampEvidence["field"] {
  return typeof value === "string" &&
    QUARANTINE_FIELDS.has(value as QuarantinedTimestampEvidence["field"]);
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
  if (Array.isArray(value)) {
    return value.map(String);
  }
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

function parseAliasArray(value: unknown): Array<[string, string]> {
  let parsed: unknown = value;
  if (typeof value === "string") {
    try {
      parsed = JSON.parse(value) as unknown;
    } catch {
      return [];
    }
  }
  if (!Array.isArray(parsed)) {
    return [];
  }
  return parsed
    .filter((item): item is [unknown, unknown] =>
      Array.isArray(item) && item.length >= 2,
    )
    .map(([kind, alias]) => [String(kind), String(alias)] as [string, string]);
}

function parseJsonUnknownArray(value: unknown): unknown[] {
  if (Array.isArray(value)) {
    return value;
  }
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

function safeTimestamp(value: string | null, notAfter?: string): string | null {
  if (value === null || !Number.isFinite(Date.parse(value))) {
    return null;
  }
  if (notAfter !== undefined && isFutureTimestamp(value, notAfter)) {
    return null;
  }
  return value;
}

function isFutureTimestamp(value: string | null, notAfter: string): boolean {
  if (value === null) {
    return false;
  }
  const timestamp = Date.parse(value);
  const bound = Date.parse(notAfter);
  return Number.isFinite(timestamp) &&
    Number.isFinite(bound) &&
    timestamp > bound;
}

function isOlderEvidence(incoming: string, current: unknown): boolean {
  if (typeof current !== "string") {
    return false;
  }
  const incomingTime = Date.parse(incoming);
  const currentTime = Date.parse(current);
  return Number.isFinite(incomingTime) &&
    Number.isFinite(currentTime) &&
    incomingTime < currentTime;
}
