import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { reaggregate } from "../../src/accounting/reaggregate.js";
import type {
  ConversationDetailProjection,
  MessageRecord,
} from "../../src/contracts/records.js";
import { Ledger, LedgerError } from "../../src/ledger/store.js";
import type {
  LedgerScope,
  MappingRule,
  ModelMappingVersion,
} from "../../src/ledger/types.js";

const temporaryDirectories: string[] = [];

afterEach(() => {
  for (const directory of temporaryDirectories.splice(0)) {
    rmSync(directory, { recursive: true, force: true });
  }
});

describe("model mapping ledger lifecycle", () => {
  it("allows draft revision, publishes once, and rejects later mutation", () => {
    const ledger = newLedger("mapping-publish");
    const draft = mapping("mapping-publish", {
      reviewStatus: "draft",
      rules: [],
    });
    ledger.saveModelMapping(draft);

    const published = {
      ...draft,
      reviewStatus: "approved" as const,
      reviewedAt: "2026-09-07T01:00:00.000Z",
      reviewedBy: "operator",
      rules: [reviewedRule("model-requested", "astra_pro")],
    };
    ledger.saveModelMapping(published);

    const stored = ledger.modelMapping("mapping-publish");
    expect(stored.reviewStatus).toBe("approved");
    expect(stored.publishedAt).toBe("2026-09-07T01:00:00.000Z");
    expect(() =>
      ledger.saveModelMapping({
        ...published,
        rules: [reviewedRule("model-requested", "sol_pro")],
      }),
    ).toThrow(LedgerError);
    expect(ledger.modelMapping("mapping-publish").rules[0]?.family).toBe("astra_pro");
    ledger.close();
  });

  it("separates prospective changes from historical corrections and preserves provenance", () => {
    const ledger = newLedger("mapping-correction");
    const account = scope("account-correction");
    ledger.upsertAccount(account);
    const initial = mapping("mapping-initial", { rules: [] });
    ledger.ingestConversation(
      account,
      detail("conversation-correction"),
      initial,
      context("run-correction"),
    );

    const prospective = mapping("mapping-prospective", {
      validFrom: "2026-09-08T00:00:00.000Z",
      rules: allModelRules(),
    });
    ledger.saveModelMapping(prospective);
    expect(
      ledger.reclassifyAttempts(
        account,
        prospective,
        "2026-09-07T12:00:00.000Z",
      ),
    ).toBe(0);
    expect(ledger.listAttempts(account)[0]?.requestedFamily).toBeNull();
    reaggregate(ledger, account, prospective, {
      evaluatedAt: "2026-09-07T12:00:00.000Z",
      apply: true,
    });
    expect(ledger.listAttempts(account)[0]?.mappingVersion).toBe("mapping-initial");

    const correction = mapping("mapping-correction", {
      changeKind: "historical_correction",
      validFrom: "2026-09-07T00:00:00.000Z",
      validUntil: "2026-09-08T00:00:00.000Z",
      correctionOfVersion: "mapping-initial",
      rules: allModelRules(),
      warnings: ["operator_corrected_historical_mapping"],
      provenance: {
        reason: "reviewed fixture evidence",
        reviewer_case: "case-752",
      },
    });
    ledger.saveModelMapping(correction);
    expect(
      ledger.reclassifyAttempts(
        account,
        correction,
        "2026-09-07T12:30:00.000Z",
        "historical_correction",
      ),
    ).toBe(1);

    const attempt = ledger.listAttempts(account)[0]!;
    expect(attempt.requestedFamily).toBe("astra_pro");
    expect(attempt.warnings).toEqual(
      expect.arrayContaining([
        "operator_corrected_historical_mapping",
        "mapping_reclassified:mapping-correction",
      ]),
    );
    const history = ledger.attemptMappingHistory(
      account,
      String(attempt.attemptId),
    );
    const correctionHistory = history.find(
      (item) => item.mapping_version === "mapping-correction",
    )!;
    expect(correctionHistory).toMatchObject({
      change_kind: "historical_correction",
      valid_from: "2026-09-07T00:00:00.000Z",
      valid_until: "2026-09-08T00:00:00.000Z",
      source: "historical_correction",
    });
    expect(correctionHistory.warnings).toEqual(
      expect.arrayContaining(["operator_corrected_historical_mapping"]),
    );
    expect(correctionHistory.provenance).toMatchObject({
      source: "historical_correction",
      mapping_version: "mapping-correction",
      previous_mapping_version: "mapping-initial",
      reason: "reviewed fixture evidence",
    });
    const priorRevisions = ledger.attemptRevisions(account, String(attempt.attemptId));
    reaggregate(ledger, account, correction, {
      evaluatedAt: "2026-09-07T13:00:00.000Z",
      apply: true,
    });
    expect(ledger.listAttempts(account)[0]).toEqual(attempt);
    expect(ledger.attemptRevisions(account, String(attempt.attemptId))).toEqual(priorRevisions);
    ledger.close();
  });
});

function newLedger(name: string): Ledger {
  const directory = mkdtempSync(join(tmpdir(), `stage2b-${name}-`));
  temporaryDirectories.push(directory);
  return new Ledger(join(directory, "usage.sqlite"));
}

function scope(collectorAccountId: string): LedgerScope {
  return {
    collectorAccountId,
    provider: "openai",
    providerUserId: "provider-user",
    workspaceId: "workspace",
    quotaOwnerId: "quota-owner",
    surface: "chat",
  };
}

function mapping(
  version: string,
  overrides: Partial<ModelMappingVersion> = {},
): ModelMappingVersion {
  return {
    version,
    canonicalFamilies: ["astra_pro", "sol_pro", "other_chat", "unknown"],
    rules: [],
    reviewStatus: "approved",
    source: "integration-test",
    createdAt: "2026-09-07T00:00:00.000Z",
    reviewedAt: "2026-09-07T00:00:00.000Z",
    reviewedBy: "integration-test",
    ...overrides,
  };
}

function reviewedRule(slug: string, family: string): MappingRule {
  return {
    slug,
    family,
    reviewed: true,
    source: "integration-test",
  };
}

function allModelRules(): MappingRule[] {
  return [
    reviewedRule("model-requested", "astra_pro"),
    reviewedRule("model-final", "astra_pro"),
    reviewedRule("model-resolved", "astra_pro"),
  ];
}

function context(runId: string) {
  return {
    runId,
    observedAt: "2026-09-07T12:00:00.000Z",
    sourceKind: "conversation_detail",
    sourceId: "conversation-correction",
    schemaVersion: "chatgpt-chat-history-v1",
    provenance: { fixture: "mapping-lifecycle" },
  };
}

function detail(conversationId: string): ConversationDetailProjection {
  return {
    conversationId,
    createdAt: "2026-09-07T11:00:00.000Z",
    updatedAt: "2026-09-07T11:30:00.000Z",
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
        requestedModelRaw: "model-requested",
      }),
      message({
        conversationId,
        messageId: "message-final",
        nodeId: "node-final",
        parentId: "node-user",
        role: "assistant",
        createdAt: "2026-09-07T11:30:00.000Z",
        status: "finished_successfully",
        endTurn: true,
        recordedFinalModelRaw: "model-final",
        generationId: "generation-1",
        requestId: "request-1",
        metadata: { resolved_model: "model-resolved" },
      }),
    ],
  };
}

function message(
  overrides: Partial<MessageRecord> &
    Pick<MessageRecord, "conversationId" | "messageId" | "role">,
): MessageRecord {
  return {
    conversationId: overrides.conversationId,
    messageId: overrides.messageId,
    nodeId: overrides.nodeId ?? overrides.messageId,
    parentId: overrides.parentId ?? null,
    children: overrides.children ?? [],
    role: overrides.role,
    channel: overrides.channel ?? null,
    createdAt: overrides.createdAt ?? "2026-09-07T11:00:00.000Z",
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
