import { randomUUID } from "node:crypto";

import type { AccountConfig } from "../config.js";
import type {
  HistoryCollectionRequest,
  HistoryCollectionResult,
  HistoryReader,
} from "../contracts/history.js";
import { ADAPTER_VERSION } from "../contracts/records.js";
import { scopeKey } from "../ledger/identity.js";
import { Ledger, LedgerError } from "../ledger/store.js";
import type { LedgerScope, ModelMappingVersion } from "../ledger/types.js";
import { INITIAL_MODEL_MAPPING } from "../normalize/model-mapping.js";
import { assertNoSecrets } from "../security/sanitizer.js";
import { SqliteCheckpointStore } from "./checkpoints.js";
import { HistoryCollector } from "./collector.js";

export interface PersistedHistoryResult extends HistoryCollectionResult {
  ledger: {
    databasePath: string;
    runId: string | null;
    committed: boolean;
    observationsInserted: number;
    messagesInserted: number;
    messagesDeduplicated: number;
    attemptsInserted: number;
    attemptsUpdated: number;
    attemptsDeduplicated: number;
    aliasConflicts: number;
  };
}

export function configuredScope(account: AccountConfig): LedgerScope {
  if (account.provider !== "openai" || account.surface !== "chat") {
    throw new LedgerError("history collection requires an openai ordinary Chat account");
  }
  const scope: LedgerScope = {
    collectorAccountId: account.id,
    provider: account.provider,
    providerUserId: account.expectedProviderUserId,
    workspaceId: account.expectedWorkspaceId,
    quotaOwnerId: account.quotaOwnerId,
    surface: "chat",
  };
  assertNoSecrets(scope);
  return scope;
}

export function assertAccountBinding(ledger: Ledger, scope: LedgerScope): void {
  const stored = ledger.listAccounts().find(
    (account) => account.collectorAccountId === scope.collectorAccountId,
  );
  if (stored && scopeKey(stored) !== scopeKey(scope)) {
    throw new LedgerError(
      "configured identity differs from the ledger binding; use a distinct account id",
    );
  }
}

export function selectMapping(
  ledger: Ledger,
  version: string | null,
): ModelMappingVersion {
  if (version) {
    return ledger.modelMapping(version);
  }
  return ledger.modelMappings().at(-1) ?? INITIAL_MODEL_MAPPING;
}

export async function collectIntoLedger(
  reader: HistoryReader,
  ledger: Ledger,
  account: AccountConfig,
  request: HistoryCollectionRequest,
  mappingVersion: string | null = null,
): Promise<PersistedHistoryResult> {
  const scope = configuredScope(account);
  assertAccountBinding(ledger, scope);
  const mapping = selectMapping(ledger, mappingVersion);
  const store = new SqliteCheckpointStore(ledger, scope);
  const result = await new HistoryCollector(reader, {
    accountId: account.id,
    store,
  }).collect(request);
  const summary: PersistedHistoryResult["ledger"] = {
    databasePath: ledger.path,
    runId: null,
    committed: false,
    observationsInserted: 0,
    messagesInserted: 0,
    messagesDeduplicated: 0,
    attemptsInserted: 0,
    attemptsUpdated: 0,
    attemptsDeduplicated: 0,
    aliasConflicts: 0,
  };
  if (result.status === "blocked") {
    return { ...result, ledger: summary };
  }
  if (
    result.identity.providerUserId !== scope.providerUserId ||
    result.identity.workspaceId !== scope.workspaceId ||
    result.identity.quotaOwnerId !== scope.quotaOwnerId
  ) {
    throw new LedgerError("collected identity differs from the configured ledger binding");
  }

  const runId = randomUUID();
  const observedAt = new Date().toISOString();
  ledger.transaction(() => {
    assertAccountBinding(ledger, scope);
    ledger.upsertAccount(scope, {
      authState: result.identity.authState,
      planPolicyId: account.planPolicyId || null,
      enabled: account.enabled,
    });
    if (ledger.modelMappings().length === 0) {
      ledger.saveModelMapping(mapping);
    }
    ledger.startRun(scope, {
      runId,
      mode: result.mode,
      startedAt: result.scanStartedAt,
    });
    for (const conversation of result.conversations) {
      const sourceKind = "history_conversation";
      const sourceId = conversation.summary.conversationId;
      ledger.resolveCoverageGaps(scope, sourceKind, sourceId, observedAt);
      ledger.upsertConversation(scope, conversation.summary, runId);
      if (!conversation.detail) {
        ledger.recordCoverageGap(scope, {
          sourceKind,
          sourceId,
          reason: "detail_unavailable",
          details: { warnings: conversation.warnings },
        }, observedAt);
        continue;
      }
      const ingested = ledger.ingestConversation(
        scope,
        {
          ...conversation.detail,
          messages: conversation.messages,
          coverage: conversation.detail.coverage === "unrecognized"
            ? "unrecognized"
            : conversation.coverage === "complete" ? "validated_page" : "partial",
          warnings: conversation.warnings,
        },
        mapping,
        {
          runId,
          observedAt,
          sourceKind,
          sourceId,
          schemaVersion: ADAPTER_VERSION,
          provenance: {
            adapter_version: ADAPTER_VERSION,
            detail_route: conversation.detail.detailRoute,
            pagination_state: conversation.detail.paginationState,
            scopes: conversation.scopes,
            coverage: conversation.coverage,
          },
        },
        conversation.summary,
      );
      summary.observationsInserted += Number(ingested.observationInserted);
      summary.messagesInserted += ingested.messageInserted;
      summary.messagesDeduplicated += ingested.messageDeduplicated;
      summary.attemptsInserted += ingested.attemptInserted;
      summary.attemptsUpdated += ingested.attemptUpdated;
      summary.attemptsDeduplicated += ingested.attemptDeduplicated;
      summary.aliasConflicts += ingested.aliasConflicts;
    }
    ledger.recordCoverageGap(scope, {
      sourceKind: "history_collection",
      sourceId: account.id,
      reason: "incomplete_history_coverage",
      state: result.status === "complete" ? "resolved" : "open",
      details: { coverage: result.coverage, pending_revisits: result.revisits.length },
    }, observedAt);
    store.persist();
    summary.runId = runId;
    summary.committed = true;
    ledger.finishRun(runId, result.status, observedAt, {
      range: result.range,
      coverage: result.coverage,
      pages_fetched: result.pagesFetched,
      detail_pages_fetched: result.detailPagesFetched,
      ...summary,
    });
  });
  return { ...result, ledger: summary };
}
