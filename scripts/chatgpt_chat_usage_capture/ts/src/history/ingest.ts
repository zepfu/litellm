import { randomUUID } from "node:crypto";

import type { AccountConfig } from "../config.js";
import type {
  HistoryCollectionRequest,
  HistoryCollectionResult,
  HistoryPageCommit,
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
  const runId = randomUUID();
  let runStarted = false;
  const persistedConversations = new Set<string>();

  const ensureRunStarted = (
    identity: HistoryPageCommit["identity"],
    mode: HistoryCollectionRequest["mode"],
    scanStartedAt: string,
  ): void => {
    if (runStarted) {
      return;
    }
    assertAccountBinding(ledger, scope);
    ledger.upsertAccount(scope, {
      authState: identity.authState,
      planPolicyId: account.planPolicyId || null,
      enabled: account.enabled,
    });
    if (ledger.modelMappings().length === 0) {
      ledger.saveModelMapping(mapping);
    }
    ledger.startRun(scope, {
      runId,
      mode,
      startedAt: scanStartedAt,
    });
    runStarted = true;
  };

  const persistConversation = (
    page: Pick<
      HistoryPageCommit,
      | "summary"
      | "scopes"
      | "detail"
      | "messages"
      | "coverage"
      | "warnings"
      | "pageKind"
      | "pageNumber"
      | "nextContinuation"
      | "revisit"
    >,
    observedAt: string,
  ): void => {
    const sourceKind = "history_conversation";
    const sourceId = page.summary.conversationId;
    const warnings = [...new Set([...page.detail?.warnings ?? [], ...page.warnings])];
    ledger.resolveCoverageGaps(scope, sourceKind, sourceId, observedAt);
    ledger.upsertConversation(scope, page.summary, runId);
    if (!page.detail) {
      ledger.recordCoverageGap(
        scope,
        {
          sourceKind,
          sourceId,
          reason: "detail_unavailable",
          details: {
            warnings,
            revisit_reason: page.revisit?.reason ?? null,
            continuation: page.revisit?.continuation ?? null,
          },
        },
        observedAt,
      );
      return;
    }

    const ingested = ledger.ingestConversation(
      scope,
      {
        ...page.detail,
        messages: page.messages,
        coverage:
          page.detail.coverage === "unrecognized"
            ? "unrecognized"
            : page.detail.coverage === "partial"
              ? "partial"
            : page.coverage === "complete"
              ? "validated_page"
              : "partial",
        warnings,
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
          detail_route: page.detail.detailRoute,
          pagination_state: page.detail.paginationState,
          scopes: page.scopes,
          coverage: page.coverage,
          page_kind: page.pageKind,
          page_number: page.pageNumber,
          next_continuation: page.nextContinuation,
          revisit_reason: page.revisit?.reason ?? null,
        },
      },
      page.summary,
    );
    summary.observationsInserted += Number(ingested.observationInserted);
    summary.messagesInserted += ingested.messageInserted;
    summary.messagesDeduplicated += ingested.messageDeduplicated;
    summary.attemptsInserted += ingested.attemptInserted;
    summary.attemptsUpdated += ingested.attemptUpdated;
    summary.attemptsDeduplicated += ingested.attemptDeduplicated;
    summary.aliasConflicts += ingested.aliasConflicts;

    if (page.revisit) {
      ledger.recordCoverageGap(
        scope,
        {
          sourceKind: "history_revisit",
          sourceId,
          reason: page.revisit.reason,
          details: {
            continuation: page.revisit.continuation,
            detail_pages_fetched: page.revisit.detailPagesFetched,
            attempts: page.revisit.attempts,
          },
        },
        observedAt,
      );
    } else {
      ledger.resolveCoverageGaps(
        scope,
        "history_revisit",
        sourceId,
        observedAt,
      );
    }
  };

  const onPageCommit = (page: HistoryPageCommit): void => {
    assertCollectedIdentity(page.identity, scope);
    ledger.transaction(() => {
      ensureRunStarted(page.identity, page.mode, page.scanStartedAt);
      persistConversation(page, new Date().toISOString());
      store.persist();
    });
    persistedConversations.add(page.summary.conversationId);
  };

  let result: HistoryCollectionResult;
  try {
    result = await new HistoryCollector(reader, {
      accountId: account.id,
      store,
      onPageCommit,
    }).collect(request);
  } catch (error) {
    if (runStarted) {
      ledger.transaction(() => {
        ledger.finishRun(runId, "failed", new Date().toISOString(), {
          error: error instanceof Error ? error.name : "collection_error",
          ...summary,
        });
      });
    }
    throw error;
  }

  if (result.status === "blocked") {
    if (result.accountState.status === "paused") {
      ledger.transaction(() => {
        ledger.upsertAccount(scope, {
          authState: result.identity.authState,
          planPolicyId: account.planPolicyId || null,
          enabled: account.enabled,
        });
        store.persist();
      });
    }
    return { ...result, ledger: summary };
  }
  assertCollectedIdentity(result.identity, scope);

  ledger.transaction(() => {
    ensureRunStarted(result.identity, result.mode, result.scanStartedAt);
    for (const conversation of result.conversations) {
      if (persistedConversations.has(conversation.summary.conversationId)) {
        continue;
      }
      persistConversation(
        {
          summary: conversation.summary,
          scopes: conversation.scopes,
          detail: conversation.detail,
          messages: conversation.messages,
          coverage: conversation.coverage,
          warnings: conversation.warnings,
          pageKind: "detail",
          pageNumber: conversation.revisit?.detailPagesFetched ?? 0,
          nextContinuation: conversation.revisit?.continuation ?? null,
          revisit: conversation.revisit,
        },
        new Date().toISOString(),
      );
    }
    const observedAt = new Date().toISOString();
    ledger.recordCoverageGap(
      scope,
      {
        sourceKind: "history_collection",
        sourceId: account.id,
        reason: "incomplete_history_coverage",
        state: result.status === "complete" ? "resolved" : "open",
        details: {
          coverage: result.coverage,
          pending_revisits: result.revisits.length,
        },
      },
      observedAt,
    );
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

function assertCollectedIdentity(
  identity: HistoryPageCommit["identity"],
  scope: LedgerScope,
): void {
  if (
    identity.authState === "paused" &&
    identity.providerUserId === null &&
    identity.workspaceId === null &&
    identity.quotaOwnerId === null
  ) {
    return;
  }
  if (
    identity.providerUserId !== scope.providerUserId ||
    identity.workspaceId !== scope.workspaceId ||
    identity.quotaOwnerId !== scope.quotaOwnerId
  ) {
    throw new LedgerError(
      "collected identity differs from the configured ledger binding",
    );
  }
}
