import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { buildRawModelReport } from "../../src/accounting/raw-model.js";
import { reaggregate } from "../../src/accounting/reaggregate.js";
import { adaptConversationDetail } from "../../src/adapters/chatgpt/adapter.js";
import type {
  ConversationDetailProjection,
  ConversationSummary,
  MessageRecord,
} from "../../src/contracts/records.js";
import { scopeKey } from "../../src/ledger/identity.js";
import { Ledger } from "../../src/ledger/store.js";
import type {
  LedgerScope,
  ModelMappingVersion,
} from "../../src/ledger/types.js";
import { reconstructAttempts } from "../../src/normalize/reconstruct.js";

const mappingUnmapped: ModelMappingVersion = {
  version: "mapping-v1",
  canonicalFamilies: ["astra_pro", "sol_pro", "unknown"],
  rules: [],
  reviewStatus: "approved",
  source: "synthetic-test",
  createdAt: "2026-09-07T00:00:00.000Z",
  reviewedAt: "2026-09-07T00:00:00.000Z",
  reviewedBy: "test",
};

const mappingReviewed: ModelMappingVersion = {
  ...mappingUnmapped,
  version: "mapping-v2",
  rules: [
    {
      slug: "model-requested",
      family: "astra_pro",
      reviewed: true,
      source: "operator-review",
    },
    {
      slug: "model-final",
      family: "astra_pro",
      reviewed: true,
      source: "operator-review",
    },
  ],
};

const temporaryDirectories: string[] = [];

afterEach(() => {
  for (const directory of temporaryDirectories.splice(0)) {
    rmSync(directory, { recursive: true, force: true });
  }
});

describe("D1-752 Stage 2B ledger", () => {
  it("runs migrations and scopes identical upstream generation IDs by account identity", () => {
    const directory = mkdtempSync(join(tmpdir(), "stage2b-ledger-"));
    temporaryDirectories.push(directory);
    const ledger = new Ledger(join(directory, "usage.sqlite"));
    expect(ledger.schemaVersion).toBe(3);

    const first = scope("account-one", "provider-user-one", "workspace-one", "quota-one");
    const second = scope("account-two", "provider-user-one", "workspace-one", "quota-one");
    ledger.upsertAccount(first);
    ledger.upsertAccount(second);

    const detail = simpleDetail("conversation-shared");
    ledger.ingestConversation(first, detail, mappingReviewed, context("run-one", "conversation-shared"), undefined);
    ledger.ingestConversation(second, detail, mappingReviewed, context("run-two", "conversation-shared"), undefined);

    const firstAttempts = ledger.listAttempts(first);
    const secondAttempts = ledger.listAttempts(second);
    expect(firstAttempts).toHaveLength(1);
    expect(secondAttempts).toHaveLength(1);
    expect(firstAttempts[0]?.attemptId).not.toBe(secondAttempts[0]?.attemptId);
    expect(scopeKey(first)).not.toBe(scopeKey(second));
    expect(
      ledger.db
        .prepare("SELECT COUNT(*) AS count FROM attempt_aliases")
        .get() as { count: number },
    ).toEqual({ count: 8 });
    ledger.close();
  });

  it("reconstructs a whole chained generation and keeps regenerations distinct", () => {
    const directory = mkdtempSync(join(tmpdir(), "stage2b-chain-"));
    temporaryDirectories.push(directory);
    const ledger = new Ledger(join(directory, "usage.sqlite"));
    const account = scope("account-chain", "provider-user", "workspace", "quota");
    ledger.upsertAccount(account);
    const detail = chainedDetail();

    const result = ledger.ingestConversation(
      account,
      detail,
      mappingReviewed,
      context("run-chain", "conversation-chain"),
    );
    expect(result.attemptInserted).toBe(2);

    const attempts = ledger.listAttempts(account);
    expect(attempts).toHaveLength(2);
    expect(attempts.every((attempt) => attempt.completedAnswer)).toBe(true);
    expect(attempts.every((attempt) => attempt.generationStarted)).toBe(true);
    expect(new Set(attempts.map((attempt) => attempt.recordedFinalModelRaw))).toEqual(
      new Set(["model-final"]),
    );
    expect(
      ledger.db
        .prepare("SELECT COUNT(*) AS count FROM attempt_evidence WHERE evidence_kind='message'")
        .get(),
    ).toEqual({ count: 10 });
    ledger.close();
  });

  it("selects the terminal answer by timestamp rather than message identifier", () => {
    const account = scope("account-final-order", "provider-user", "workspace", "quota");
    const messages = [
      message({
        conversationId: "conversation-final-order",
        messageId: "message-user",
        nodeId: "node-user",
        role: "user",
        children: ["node-old", "node-new"],
        createdAt: "2026-09-07T10:00:00.000Z",
        requestedModelRaw: "model-requested",
      }),
      message({
        conversationId: "conversation-final-order",
        messageId: "zz-old-final",
        nodeId: "node-old",
        parentId: "node-user",
        role: "assistant",
        createdAt: "2026-09-07T10:00:02.000Z",
        status: "finished_successfully",
        endTurn: true,
        recordedFinalModelRaw: "model-old",
        generationId: "generation-order",
        requestId: "request-order",
      }),
      message({
        conversationId: "conversation-final-order",
        messageId: "aa-new-final",
        nodeId: "node-new",
        parentId: "node-user",
        role: "assistant",
        createdAt: "2026-09-07T10:00:03.000Z",
        status: "finished_successfully",
        endTurn: true,
        recordedFinalModelRaw: "model-new",
        generationId: "generation-order",
        requestId: "request-order",
      }),
    ];
    const attempts = reconstructAttempts(messages, {
      scope: account,
      conversationId: "conversation-final-order",
      mapping: mappingReviewed,
    });
    expect(attempts).toHaveLength(1);
    expect(attempts[0]?.recordedFinalModelRaw).toBe("model-new");
  });

  it("deduplicates replay and preserves in-progress to completed revisions", () => {
    const directory = mkdtempSync(join(tmpdir(), "stage2b-revision-"));
    temporaryDirectories.push(directory);
    const ledger = new Ledger(join(directory, "usage.sqlite"));
    const account = scope("account-revision", "provider-user", "workspace", "quota");
    ledger.upsertAccount(account);

    const first = progressDetail("conversation-revision", "in_progress", false, null);
    const firstResult = ledger.ingestConversation(
      account,
      first,
      mappingReviewed,
      context("run-revision-1", "conversation-revision", "2026-09-07T10:00:00.000Z"),
    );
    expect(firstResult.attemptInserted).toBe(1);
    const replay = ledger.ingestConversation(
      account,
      first,
      mappingReviewed,
      context("run-revision-2", "conversation-revision", "2026-09-07T10:05:00.000Z"),
    );
    expect(replay.observationInserted).toBe(false);
    expect(replay.messageDeduplicated).toBe(2);
    expect(replay.attemptDeduplicated).toBe(1);

    const completed = progressDetail(
      "conversation-revision",
      "finished_successfully",
      true,
      "model-final",
    );
    const completedResult = ledger.ingestConversation(
      account,
      completed,
      mappingReviewed,
      context("run-revision-3", "conversation-revision", "2026-09-07T10:10:00.000Z"),
    );
    expect(completedResult.observationInserted).toBe(true);
    expect(completedResult.attemptUpdated).toBe(1);

    const attempt = ledger.listAttempts(account)[0]!;
    expect(attempt.completedAnswer).toBe(true);
    expect(attempt.outcome).toBe("completed");
    expect(
      ledger.attemptRevisions(account, String(attempt.attemptId)).map((item) => item.revision),
    ).toEqual([1, 2]);
    expect(ledger.messageRevisions(account, "conversation-revision", "message-final")).toHaveLength(2);
    expect(
      ledger.db
        .prepare(
          "SELECT COUNT(*) AS count FROM observations WHERE scope_key=? AND source_kind=? AND source_id=?",
        )
        .get(scopeKey(account), "conversation_detail", "conversation-revision"),
    ).toEqual({ count: 2 });
    ledger.close();
  });

  it("sanitizes evidence before storage and records coverage provenance", () => {
    const directory = mkdtempSync(join(tmpdir(), "stage2b-privacy-"));
    temporaryDirectories.push(directory);
    const ledger = new Ledger(join(directory, "usage.sqlite"));
    const account = scope("account-private", "provider-user", "workspace", "quota");
    ledger.upsertAccount(account);
    const detail = simpleDetail("conversation-private");
    detail.messages[0]!.metadata = {
      model_slug: "model-requested",
      content: "private prompt must not persist",
      authorization: "Bearer secret",
      resolved_model: "model-requested",
    };
    detail.coverage = "partial";
    detail.warnings = [
      "missing_branch_page",
      ...adaptConversationDetail({
        surface: "chat",
        mapping: { "private prompt must not persist": null },
      }, detail.conversationId).warnings,
    ];

    ledger.ingestConversation(
      account,
      detail,
      mappingReviewed,
      context("run-private", "conversation-private"),
    );
    const evidenceText = String(
      (
        ledger.db
          .prepare("SELECT GROUP_CONCAT(payload_json) AS payloads FROM observations")
          .get() as { payloads: string | null }
      ).payloads ?? "",
    );
    expect(evidenceText).not.toContain("private prompt");
    expect(evidenceText).not.toContain("Bearer secret");
    expect(ledger.coverageGaps(account)).toHaveLength(1);
    expect(ledger.coverageGaps(account)[0]?.reason).toBe("partial_conversation_detail");
    ledger.close();
  });

  it("reclassifies from retained raw evidence, preserves mapping history, and rebuilds without website access", () => {
    const directory = mkdtempSync(join(tmpdir(), "stage2b-rebuild-"));
    temporaryDirectories.push(directory);
    const ledger = new Ledger(join(directory, "usage.sqlite"));
    const account = scope("account-rebuild", "provider-user", "workspace", "quota");
    ledger.upsertAccount(account);
    ledger.saveModelMapping(mappingUnmapped);
    ledger.saveModelMapping(mappingReviewed);
    ledger.ingestConversation(
      account,
      simpleDetail("conversation-rebuild"),
      mappingUnmapped,
      context("run-rebuild", "conversation-rebuild"),
    );
    expect(ledger.listAttempts(account)[0]?.requestedFamily).toBeNull();
    expect(
      ledger.modelMappingSuggestions(account, mappingUnmapped).map((item) => ({
        slug: item.slug,
        observedAttempts: item.observedAttempts,
      })),
    ).toEqual([
      { slug: "model-final", observedAttempts: 1 },
      { slug: "model-requested", observedAttempts: 1 },
      { slug: "model-resolved", observedAttempts: 1 },
    ]);

    const preview = reaggregate(ledger, account, mappingReviewed, {
      evaluatedAt: "2026-09-07T12:00:00.000Z",
      apply: false,
    });
    expect(preview.reclassifiedAttempts).toBe(1);
    expect(ledger.listAttempts(account)[0]?.requestedFamily).toBeNull();

    const applied = reaggregate(ledger, account, mappingReviewed, {
      evaluatedAt: "2026-09-07T12:00:00.000Z",
      apply: true,
    });
    expect(applied.report.observedAttemptsByRequestedModel["model-requested"]).toBe(1);
    expect(ledger.listAttempts(account)[0]?.requestedFamily).toBe("astra_pro");
    expect(
      ledger.attemptMappingHistory(
        account,
        String(ledger.listAttempts(account)[0]?.attemptId),
      ).map((item) => item.mapping_version),
    ).toEqual(["mapping-v1", "mapping-v2"]);
    expect(ledger.aggregateRevisions(account)).toHaveLength(1);
    ledger.close();
  });

  it("reports raw-model activity over a half-open last-N duration", () => {
    const directory = mkdtempSync(join(tmpdir(), "stage2b-report-"));
    temporaryDirectories.push(directory);
    const ledger = new Ledger(join(directory, "usage.sqlite"));
    const account = scope("account-report", "provider-user", "workspace", "quota");
    ledger.upsertAccount(account);
    ledger.ingestConversation(
      account,
      progressDetail(
        "conversation-report",
        "finished_successfully",
        true,
        "model-final",
        "2026-09-07T11:00:00.000Z",
        "model-requested",
      ),
      mappingReviewed,
      context("run-report", "conversation-report", "2026-09-07T11:01:00.000Z"),
    );
    const report = buildRawModelReport(ledger, account, {
      durationMs: 60 * 60 * 1000,
      now: "2026-09-07T12:00:00.000Z",
    });
    expect(report.observedAttemptsByRequestedModel).toEqual({
      "model-requested": 1,
    });
    expect(report.completedAnswersByRecordedFinalModel).toEqual({
      "model-final": 1,
    });
    expect(report.observedAttemptsByResolvedModel).toEqual({
      "model-resolved": 1,
    });
    expect(report.modelMismatches).toBe(1);
    ledger.close();
  });

  it("retains but excludes conversation-level shared origins", () => {
    const directory = mkdtempSync(join(tmpdir(), "stage2b-origin-"));
    temporaryDirectories.push(directory);
    const ledger = new Ledger(join(directory, "usage.sqlite"));
    const account = scope("account-origin", "provider-user", "workspace", "quota");
    ledger.upsertAccount(account);
    const detail = simpleDetail("conversation-shared-origin");
    const summary: ConversationSummary = {
      conversationId: detail.conversationId,
      createdAt: detail.createdAt,
      updatedAt: detail.updatedAt,
      isArchived: false,
      workspaceId: account.workspaceId,
      projectId: null,
      surface: "chat",
      origin: "shared",
      hasVersions: null,
      currentNode: detail.currentNode,
      coverage: "validated_page",
    };

    ledger.ingestConversation(
      account,
      detail,
      mappingReviewed,
      context("run-origin", detail.conversationId),
      summary,
    );
    const report = buildRawModelReport(ledger, account, {
      durationMs: 24 * 60 * 60 * 1000,
      now: "2026-09-08T00:00:00.000Z",
    });
    expect(report.includedAttempts).toBe(1);
    expect(report.excludedOriginAttempts).toBe(1);
    expect(report.observedAttemptsByRequestedModel).toEqual({});
    expect(ledger.listAttempts(account)[0]?.origin).toBe("shared");
    const rebuilt = reaggregate(ledger, account, mappingReviewed, {
      evaluatedAt: report.end,
      apply: true,
    });
    expect(rebuilt.rebuiltAttempts).toBe(0);
    expect(rebuilt.report.excludedOriginAttempts).toBe(1);
    expect(rebuilt.report.observedAttemptsByRequestedModel).toEqual({});
    expect(ledger.listAttempts(account)[0]?.origin).toBe("shared");
    ledger.close();
  });
});

function scope(
  collectorAccountId: string,
  providerUserId: string,
  workspaceId: string,
  quotaOwnerId: string,
): LedgerScope {
  return {
    collectorAccountId,
    provider: "openai",
    providerUserId,
    workspaceId,
    quotaOwnerId,
    surface: "chat",
  };
}

function context(
  runId: string,
  sourceId: string,
  observedAt = "2026-09-07T12:00:00.000Z",
) {
  return {
    runId,
    observedAt,
    sourceKind: "conversation_detail",
    sourceId,
    schemaVersion: "chatgpt-chat-history-v1",
    provenance: { fixture: "stage2b-synthetic" },
  };
}

function simpleDetail(conversationId: string): ConversationDetailProjection {
  return progressDetail(conversationId, "finished_successfully", true, "model-final");
}

function progressDetail(
  conversationId: string,
  status: string,
  endTurn: boolean,
  finalModel: string | null,
  createdAt = "2026-09-07T11:30:00.000Z",
  requestedModel = "model-requested",
): ConversationDetailProjection {
  return {
    conversationId,
    createdAt: "2026-09-07T11:00:00.000Z",
    updatedAt: createdAt,
    currentNode: "node-final",
    surface: "chat",
    detailRoute: "modern",
    paginationState: "complete",
    coverage: "validated_page",
    warnings: [],
    messages: [
      message({
        conversationId,
        messageId: "message-user",
        nodeId: "node-user",
        role: "user",
        children: ["node-final"],
        createdAt: "2026-09-07T11:00:00.000Z",
        requestedModelRaw: requestedModel,
        generationId: null,
        requestId: null,
        metadata: { resolved_model: "model-resolved" },
      }),
      message({
        conversationId,
        messageId: "message-final",
        nodeId: "node-final",
        parentId: "node-user",
        role: "assistant",
        createdAt,
        status,
        endTurn,
        recordedFinalModelRaw: finalModel,
        generationId: "generation-1",
        requestId: "request-1",
      }),
    ],
  };
}

function chainedDetail(): ConversationDetailProjection {
  const conversationId = "conversation-chain";
  return {
    conversationId,
    createdAt: "2026-09-07T10:00:00.000Z",
    updatedAt: "2026-09-07T10:01:00.000Z",
    currentNode: "node-final-2",
    surface: "chat",
    detailRoute: "modern",
    paginationState: "complete",
    coverage: "validated_page",
    warnings: [],
    messages: [
      message({
        conversationId,
        messageId: "message-user",
        nodeId: "node-user",
        role: "user",
        children: ["node-analysis-1", "node-final-2"],
        createdAt: "2026-09-07T10:00:00.000Z",
        requestedModelRaw: "model-requested",
      }),
      message({
        conversationId,
        messageId: "message-analysis",
        nodeId: "node-analysis-1",
        parentId: "node-user",
        role: "assistant",
        channel: "analysis",
        createdAt: "2026-09-07T10:00:01.000Z",
        status: "finished_successfully",
        endTurn: false,
        generationId: "generation-1",
        requestId: "request-1",
      }),
      message({
        conversationId,
        messageId: "message-reasoning",
        nodeId: "node-reasoning-1",
        parentId: "node-analysis-1",
        role: "assistant",
        channel: "reasoning",
        createdAt: "2026-09-07T10:00:02.000Z",
        status: "finished_successfully",
        endTurn: false,
        generationId: "generation-1",
        requestId: "request-1",
      }),
      message({
        conversationId,
        messageId: "message-tool",
        nodeId: "node-tool-1",
        parentId: "node-reasoning-1",
        role: "tool",
        channel: "tool",
        createdAt: "2026-09-07T10:00:03.000Z",
        status: "finished_successfully",
        endTurn: false,
        generationId: "generation-1",
        requestId: "request-1",
      }),
      message({
        conversationId,
        messageId: "message-final-1",
        nodeId: "node-final-1",
        parentId: "node-tool-1",
        role: "assistant",
        createdAt: "2026-09-07T10:00:04.000Z",
        status: "finished_successfully",
        endTurn: true,
        recordedFinalModelRaw: "model-final",
        generationId: "generation-1",
        requestId: "request-1",
      }),
      message({
        conversationId,
        messageId: "message-analysis-2",
        nodeId: "node-analysis-2",
        parentId: "node-user",
        role: "assistant",
        channel: "analysis",
        createdAt: "2026-09-07T10:00:04.000Z",
        status: "finished_successfully",
        endTurn: false,
        generationId: "generation-2",
        requestId: "request-1",
      }),
      message({
        conversationId,
        messageId: "message-reasoning-2",
        nodeId: "node-reasoning-2",
        parentId: "node-analysis-2",
        role: "assistant",
        channel: "reasoning",
        createdAt: "2026-09-07T10:00:06.000Z",
        status: "finished_successfully",
        endTurn: false,
        generationId: "generation-2",
        requestId: "request-1",
      }),
      message({
        conversationId,
        messageId: "message-tool-2",
        nodeId: "node-tool-2",
        parentId: "node-reasoning-2",
        role: "tool",
        channel: "tool",
        createdAt: "2026-09-07T10:00:07.000Z",
        status: "finished_successfully",
        endTurn: false,
        generationId: "generation-2",
        requestId: "request-1",
      }),
      message({
        conversationId,
        messageId: "message-final-2",
        nodeId: "node-final-2",
        parentId: "node-tool-2",
        role: "assistant",
        createdAt: "2026-09-07T10:00:08.000Z",
        status: "finished_successfully",
        endTurn: true,
        recordedFinalModelRaw: "model-final",
        generationId: "generation-2",
        requestId: "request-1",
      }),
    ],
  };
}

function message(
  overrides: Partial<MessageRecord> & Pick<MessageRecord, "conversationId" | "messageId" | "role">,
): MessageRecord {
  return {
    conversationId: overrides.conversationId,
    messageId: overrides.messageId,
    nodeId: overrides.nodeId ?? overrides.messageId,
    parentId: overrides.parentId ?? null,
    children: overrides.children ?? [],
    role: overrides.role,
    channel: overrides.channel ?? null,
    createdAt: overrides.createdAt ?? "2026-09-07T10:00:00.000Z",
    status: overrides.status ?? "finished_successfully",
    endTurn: overrides.endTurn ?? null,
    requestedModelRaw: overrides.requestedModelRaw ?? null,
    requestedModeRaw: overrides.requestedModeRaw ?? null,
    requestedReasoningEffortRaw: overrides.requestedReasoningEffortRaw ?? null,
    recordedFinalModelRaw: overrides.recordedFinalModelRaw ?? null,
    generationId: overrides.generationId ?? null,
    requestId: overrides.requestId ?? null,
    surface: overrides.surface ?? "chat",
    origin: overrides.origin ?? null,
    metadata: overrides.metadata ?? {},
  };
}
