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
import { collectorScopeKey, scopeKey } from "../../src/ledger/identity.js";
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
  it("runs migrations and converges identical upstream activity for one verified owner", () => {
    const directory = mkdtempSync(join(tmpdir(), "stage2b-ledger-"));
    temporaryDirectories.push(directory);
    const ledger = new Ledger(join(directory, "usage.sqlite"));
    expect(ledger.schemaVersion).toBe(5);

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
    expect(firstAttempts[0]?.attemptId).toBe(secondAttempts[0]?.attemptId);
    expect(scopeKey(first)).toBe(scopeKey(second));
    expect(collectorScopeKey(first)).not.toBe(collectorScopeKey(second));
    expect(
      ledger.db
        .prepare("SELECT COUNT(*) AS count FROM attempt_aliases")
        .get() as { count: number },
    ).toEqual({ count: 5 });
    expect(
      ledger.activityProvenance(
        first,
        "attempt",
        String(firstAttempts[0]?.attemptId),
      ).map((row) => row.collector_account_id),
    ).toEqual(["account-one", "account-two"]);
    expect(
      ledger.db
        .prepare(
          "SELECT COUNT(*) AS count FROM observations WHERE scope_key=?",
        )
        .get(scopeKey(first)),
    ).toEqual({ count: 2 });
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

  it("groups disjoint generations when linkage is only visible in either graph direction", () => {
    const account = scope("account-bidirectional", "provider-user", "workspace", "quota");
    const conversationId = "conversation-bidirectional";
    const messages = [
      message({
        conversationId,
        messageId: "message-user",
        nodeId: "node-user",
        role: "user",
        children: ["node-a-progress", "node-b-progress"],
        createdAt: "2026-09-07T10:00:00.000Z",
        requestedModelRaw: "model-requested",
      }),
      message({
        conversationId,
        messageId: "message-a-progress",
        nodeId: "node-a-progress",
        parentId: "node-user",
        role: "assistant",
        channel: "analysis",
        createdAt: "2026-09-07T10:00:01.000Z",
        status: "finished_successfully",
        generationId: "generation-a",
      }),
      message({
        conversationId,
        messageId: "message-a-final",
        nodeId: "node-a-final",
        parentId: "node-a-progress",
        role: "assistant",
        createdAt: "2026-09-07T10:00:02.000Z",
        status: "finished_successfully",
        endTurn: true,
        recordedFinalModelRaw: "model-final",
      }),
      message({
        conversationId,
        messageId: "message-b-progress",
        nodeId: "node-b-progress",
        parentId: "node-user",
        role: "assistant",
        channel: "analysis",
        createdAt: "2026-09-07T10:01:01.000Z",
        status: "finished_successfully",
      }),
      message({
        conversationId,
        messageId: "message-b-final",
        nodeId: "node-b-final",
        parentId: "node-b-progress",
        role: "assistant",
        createdAt: "2026-09-07T10:01:02.000Z",
        status: "finished_successfully",
        endTurn: true,
        recordedFinalModelRaw: "model-final",
        generationId: "generation-b",
      }),
    ];

    const attempts = reconstructAttempts(messages, {
      scope: account,
      conversationId,
      mapping: mappingReviewed,
    });

    expect(attempts).toHaveLength(2);
    expect(attempts.every((attempt) => attempt.completedAnswer)).toBe(true);
    const nonPromptEvidence = attempts.map(
      (attempt) =>
        new Set(attempt.evidenceMessageIds.filter((id) => id !== "message-user")),
    );
    expect(nonPromptEvidence[0] && nonPromptEvidence[1]).toBeTruthy();
    expect(
      [...(nonPromptEvidence[0] ?? [])].some((id) =>
        (nonPromptEvidence[1] ?? new Set()).has(id),
      ),
    ).toBe(false);
    expect(
      new Set(
        attempts.flatMap((attempt) =>
          attempt.aliases
            .filter(([kind]) => kind === "generation")
            .map(([, value]) => value),
        ),
      ),
    ).toEqual(
      new Set([
        `${conversationId}:generation-a`,
        `${conversationId}:generation-b`,
      ]),
    );
  });

  it("joins a generation through a child-only edge in the reverse direction", () => {
    const account = scope("account-child-edge", "provider-user", "workspace", "quota");
    const conversationId = "conversation-child-edge";
    const attempts = reconstructAttempts(
      [
        message({
          conversationId,
          messageId: "message-user",
          nodeId: "node-user",
          role: "user",
          children: ["node-generation"],
          createdAt: "2026-09-07T10:00:00.000Z",
          requestedModelRaw: "model-requested",
        }),
        message({
          conversationId,
          messageId: "message-generation",
          nodeId: "node-generation",
          role: "assistant",
          children: ["node-final"],
          channel: "analysis",
          createdAt: "2026-09-07T10:00:01.000Z",
          status: "finished_successfully",
          generationId: "generation-child-edge",
          requestId: "request-child-edge",
        }),
        message({
          conversationId,
          messageId: "message-final",
          nodeId: "node-final",
          role: "assistant",
          createdAt: "2026-09-07T10:00:02.000Z",
          status: "finished_successfully",
          endTurn: true,
          recordedFinalModelRaw: "model-final",
        }),
      ],
      {
        scope: account,
        conversationId,
        mapping: mappingReviewed,
      },
    );

    expect(attempts).toHaveLength(1);
    expect(attempts[0]?.completedAnswer).toBe(true);
    expect(attempts[0]?.evidenceMessageIds).toEqual([
      "message-final",
      "message-generation",
      "message-user",
    ]);
  });

  it("merges later linkage into one active attempt and retires the provisional row", () => {
    const directory = mkdtempSync(join(tmpdir(), "stage2b-linkage-"));
    temporaryDirectories.push(directory);
    const ledger = new Ledger(join(directory, "usage.sqlite"));
    const account = scope("account-linkage", "provider-user", "workspace", "quota");
    ledger.upsertAccount(account);

    const provisional = progressDetail(
      "conversation-linkage",
      "finished_successfully",
      true,
      "model-final",
    );
    provisional.messages[1] = {
      ...provisional.messages[1]!,
      generationId: null,
      requestId: null,
    };
    ledger.ingestConversation(
      account,
      provisional,
      mappingReviewed,
      context("run-linkage-1", "conversation-linkage", "2026-09-07T12:00:00.000Z"),
    );

    const linked = ledger.ingestConversation(
      account,
      progressDetail("conversation-linkage", "finished_successfully", true, "model-final"),
      mappingReviewed,
      context("run-linkage-2", "conversation-linkage", "2026-09-07T12:05:00.000Z"),
    );

    expect(linked.attemptUpdated).toBe(1);
    expect(ledger.listAttempts(account)).toHaveLength(1);
    expect(ledger.listAttempts(account)[0]?.identityBasis).toBe("generation");
    expect(ledger.listAttempts(account, true)).toHaveLength(2);
    expect(
      ledger.listAttempts(account, true).filter((attempt) => attempt.tombstone),
    ).toHaveLength(1);
    expect(
      ledger.db
        .prepare(
          "SELECT attempt_id FROM attempt_aliases WHERE scope_key=? AND alias_kind='generation'",
        )
        .get(scopeKey(account)),
    ).toEqual({ attempt_id: ledger.listAttempts(account)[0]?.attemptId });
    expect(
      ledger.db
        .prepare(
          "SELECT COUNT(*) AS count FROM attempt_evidence WHERE scope_key=? AND evidence_kind='message'",
        )
        .get(scopeKey(account)),
    ).toEqual({ count: 2 });
    ledger.reclassifyAttempts(account, mappingReviewed, "2026-09-07T12:10:00.000Z");
    expect(ledger.listAttempts(account)).toHaveLength(1);
    expect(ledger.listAttempts(account, true).filter((attempt) => attempt.tombstone))
      .toHaveLength(1);
    expect(ledger.db.pragma("foreign_key_check")).toEqual([]);
    ledger.close();
  });

  it("keeps conflicting generation attempts separate when weak aliases collide", () => {
    const directory = mkdtempSync(join(tmpdir(), "stage2b-alias-generation-"));
    temporaryDirectories.push(directory);
    const ledger = new Ledger(join(directory, "usage.sqlite"));
    const account = scope("account-alias-generation", "provider-user", "workspace", "quota");
    const conversationId = "conversation-alias-generation";
    ledger.upsertAccount(account);

    const makeAttempt = (
      generationId: string,
      messageId: string,
      branchAlias: string,
      promptAlias: string,
    ) => {
      const attempt = reconstructAttempts(
        [
          message({
            conversationId,
            messageId,
            nodeId: messageId,
            role: "assistant",
            createdAt: "2026-09-07T10:00:02.000Z",
            status: "finished_successfully",
            endTurn: true,
            recordedFinalModelRaw: "model-final",
            generationId,
            requestId: `request-${generationId}`,
          }),
        ],
        {
          scope: account,
          conversationId,
          mapping: mappingReviewed,
        },
      )[0];
      if (!attempt) {
        throw new Error(`missing reconstructed attempt for ${generationId}`);
      }
      return {
        ...attempt,
        aliases: [
          ["generation", `${conversationId}:${generationId}`],
          ["branch", `${conversationId}:${branchAlias}`],
          ["prompt", `${conversationId}:${promptAlias}`],
        ] as Array<[string, string]>,
      };
    };

    const first = makeAttempt("generation-a", "message-a", "shared-branch", "prompt-a");
    const second = makeAttempt("generation-b", "message-b", "branch-b", "shared-prompt");
    const firstResult = ledger.upsertAttempt(
      account,
      first,
      context("run-alias-a", conversationId, "2026-09-07T10:05:00.000Z"),
    );
    const secondResult = ledger.upsertAttempt(
      account,
      second,
      context("run-alias-b", conversationId, "2026-09-07T10:06:00.000Z"),
    );
    const conflicting = makeAttempt(
      "generation-c",
      "message-c",
      "shared-branch",
      "prompt-a",
    );
    const conflictingResult = ledger.upsertAttempt(
      account,
      conflicting,
      context("run-alias-c", conversationId, "2026-09-07T10:07:00.000Z"),
    );
    const bridge = {
      ...first,
      attemptId: "attempt-provisional-bridge",
      identityBasis: "provisional",
      aliases: [
        ["branch", `${conversationId}:shared-branch`],
        ["prompt", `${conversationId}:shared-prompt`],
      ] as Array<[string, string]>,
      evidenceMessageIds: ["message-bridge"],
      warnings: [...new Set([...first.warnings, "provisional_identity"])],
    };
    const bridgeResult = ledger.upsertAttempt(
      account,
      bridge,
      context("run-alias-bridge", conversationId, "2026-09-07T10:08:00.000Z"),
    );

    expect(firstResult.status).toBe("inserted");
    expect(secondResult.status).toBe("inserted");
    expect(secondResult.aliasConflicts).toBe(0);
    expect(conflictingResult.status).toBe("inserted");
    expect(conflictingResult.aliasConflicts).toBe(2);
    expect(bridgeResult.status).toBe("inserted");
    expect(bridgeResult.aliasConflicts).toBe(2);
    expect(ledger.listAttempts(account)).toHaveLength(4);
    expect(ledger.listAttempts(account, true).filter((attempt) => attempt.tombstone))
      .toHaveLength(0);
    expect(
      ledger.db
        .prepare(
          "SELECT COUNT(*) AS count FROM attempt_aliases WHERE scope_key=? AND alias_kind='generation'",
        )
        .get(scopeKey(account)),
    ).toEqual({ count: 3 });
    ledger.close();
  });

  it("does not inherit prompt time or model into a later regeneration", () => {
    const account = scope("account-regeneration", "provider-user", "workspace", "quota");
    const conversationId = "conversation-regeneration";
    const attempts = reconstructAttempts(
      [
        message({
          conversationId,
          messageId: "message-user",
          nodeId: "node-user",
          role: "user",
          children: ["node-original", "node-regenerated"],
          createdAt: "2026-09-07T10:00:00.000Z",
          requestedModelRaw: "model-requested",
        }),
        message({
          conversationId,
          messageId: "message-original",
          nodeId: "node-original",
          parentId: "node-user",
          role: "assistant",
          createdAt: "2026-09-07T10:00:02.000Z",
          status: "finished_successfully",
          endTurn: true,
          recordedFinalModelRaw: "model-final",
          generationId: "generation-original",
        }),
        message({
          conversationId,
          messageId: "message-regenerated",
          nodeId: "node-regenerated",
          parentId: "node-user",
          role: "assistant",
          createdAt: "2026-09-07T10:10:02.000Z",
          status: "finished_successfully",
          endTurn: true,
          recordedFinalModelRaw: "model-final",
          generationId: "generation-regenerated",
        }),
      ],
      {
        scope: account,
        conversationId,
        mapping: mappingReviewed,
      },
    );

    const original = attempts.find((attempt) =>
      attempt.aliases.some(
        ([kind, value]) =>
          kind === "generation" && value === `${conversationId}:generation-original`,
      ),
    );
    const regenerated = attempts.find((attempt) =>
      attempt.aliases.some(
        ([kind, value]) =>
          kind === "generation" && value === `${conversationId}:generation-regenerated`,
      ),
    );
    expect(original?.requestedModelRaw).toBe("model-requested");
    expect(original?.attemptTime).toBe("2026-09-07T10:00:00.000Z");
    expect(regenerated?.requestedModelRaw).toBeNull();
    expect(regenerated?.attemptTime).toBeNull();
    expect(regenerated?.earliestPossibleAt).toBeNull();
    expect(regenerated?.latestPossibleAt).toBeNull();
  });

  it("does not distribute prompt evidence through a reused request ID", () => {
    const account = scope("account-reused-request", "provider-user", "workspace", "quota");
    const conversationId = "conversation-reused-request";
    const attempts = reconstructAttempts(
      [
        message({
          conversationId,
          messageId: "message-user",
          nodeId: "node-user",
          role: "user",
          children: ["node-generation-a", "node-generation-b"],
          createdAt: "2026-09-07T10:00:00.000Z",
          requestedModelRaw: "model-requested",
        }),
        message({
          conversationId,
          messageId: "message-final-a",
          nodeId: "node-generation-a",
          parentId: "node-user",
          role: "assistant",
          createdAt: "2026-09-07T10:00:02.000Z",
          status: "finished_successfully",
          endTurn: true,
          recordedFinalModelRaw: "model-final",
          generationId: "generation-a",
          requestId: "request-reused",
        }),
        message({
          conversationId,
          messageId: "message-final-b",
          nodeId: "node-generation-b",
          parentId: "node-user",
          role: "assistant",
          createdAt: "2026-09-07T10:01:02.000Z",
          status: "finished_successfully",
          endTurn: true,
          recordedFinalModelRaw: "model-final",
          generationId: "generation-b",
          requestId: "request-reused",
        }),
      ],
      {
        scope: account,
        conversationId,
        mapping: mappingReviewed,
      },
    );

    expect(attempts).toHaveLength(2);
    expect(attempts.every((attempt) => attempt.requestedModelRaw === null)).toBe(true);
    expect(attempts.every((attempt) => attempt.attemptTime === null)).toBe(true);
    expect(
      attempts.every((attempt) =>
        attempt.warnings.includes("prompt_evidence_not_linked"),
      ),
    ).toBe(true);
  });

  it("does not let a provisional response branch inherit prompt evidence", () => {
    const account = scope("account-provisional-prompt", "provider-user", "workspace", "quota");
    const conversationId = "conversation-provisional-prompt";
    const attempts = reconstructAttempts(
      [
        message({
          conversationId,
          messageId: "message-user",
          nodeId: "node-user",
          role: "user",
          children: ["node-provisional"],
          createdAt: "2026-09-07T10:00:00.000Z",
          requestedModelRaw: "model-requested",
        }),
        message({
          conversationId,
          messageId: "message-provisional",
          nodeId: "node-provisional",
          parentId: "node-user",
          role: "assistant",
          createdAt: "2026-09-07T10:00:02.000Z",
          status: "finished_successfully",
          endTurn: true,
          recordedFinalModelRaw: "model-final",
        }),
      ],
      {
        scope: account,
        conversationId,
        mapping: mappingReviewed,
      },
    );

    expect(attempts).toHaveLength(1);
    expect(attempts[0]?.identityBasis).toBe("provisional");
    expect(attempts[0]?.requestedModelRaw).toBeNull();
    expect(attempts[0]?.attemptTime).toBeNull();
    expect(attempts[0]?.earliestPossibleAt).toBeNull();
    expect(attempts[0]?.latestPossibleAt).toBeNull();
    expect(attempts[0]?.recordedFinalModelRaw).toBe("model-final");
  });

  it("does not fabricate request bounds from a final-only response", () => {
    const account = scope("account-final-only", "provider-user", "workspace", "quota");
    const conversationId = "conversation-final-only";
    const attempts = reconstructAttempts(
      [
        message({
          conversationId,
          messageId: "message-final-only",
          nodeId: "node-final-only",
          role: "assistant",
          createdAt: "2026-09-07T10:00:02.000Z",
          status: "finished_successfully",
          endTurn: true,
          recordedFinalModelRaw: "model-final",
        }),
      ],
      {
        scope: account,
        conversationId,
        mapping: mappingReviewed,
      },
    );

    expect(attempts).toHaveLength(1);
    expect(attempts[0]?.timeBasis).toBe("unknown");
    expect(attempts[0]?.attemptTime).toBeNull();
    expect(attempts[0]?.earliestPossibleAt).toBeNull();
    expect(attempts[0]?.latestPossibleAt).toBeNull();
  });

  it("does not use response progress as a request lower bound", () => {
    const account = scope("account-response-bound", "provider-user", "workspace", "quota");
    const conversationId = "conversation-response-bound";
    const attempts = reconstructAttempts(
      [
        message({
          conversationId,
          messageId: "message-progress",
          nodeId: "node-progress",
          role: "assistant",
          children: ["node-final"],
          channel: "analysis",
          createdAt: "2026-09-07T10:00:01.000Z",
          status: "in_progress",
          generationId: "generation-response-bound",
          requestId: "request-response-bound",
        }),
        message({
          conversationId,
          messageId: "message-final",
          nodeId: "node-final",
          role: "assistant",
          createdAt: "2026-09-07T10:00:02.000Z",
          status: "finished_successfully",
          endTurn: true,
          recordedFinalModelRaw: "model-final",
          generationId: "generation-response-bound",
          requestId: "request-response-bound",
        }),
      ],
      {
        scope: account,
        conversationId,
        mapping: mappingReviewed,
      },
    );

    expect(attempts).toHaveLength(1);
    expect(attempts[0]?.timeBasis).toBe("response_observed");
    expect(attempts[0]?.attemptTime).toBeNull();
    expect(attempts[0]?.earliestPossibleAt).toBeNull();
    expect(attempts[0]?.latestPossibleAt).toBe("2026-09-07T10:00:02.000Z");
  });

  it("classifies rejection after observed generation start separately", () => {
    const account = scope("account-rejection", "provider-user", "workspace", "quota");
    const conversationId = "conversation-rejection";
    const attempts = reconstructAttempts(
      [
        message({
          conversationId,
          messageId: "message-user",
          nodeId: "node-user",
          role: "user",
          children: ["node-progress"],
          createdAt: "2026-09-07T10:00:00.000Z",
        }),
        message({
          conversationId,
          messageId: "message-progress",
          nodeId: "node-progress",
          parentId: "node-user",
          children: ["node-rejected"],
          role: "assistant",
          channel: "analysis",
          createdAt: "2026-09-07T10:00:01.000Z",
          status: "in_progress",
          generationId: "generation-rejected",
          requestId: "request-rejected",
        }),
        message({
          conversationId,
          messageId: "message-rejected",
          nodeId: "node-rejected",
          parentId: "node-progress",
          role: "assistant",
          createdAt: "2026-09-07T10:00:02.000Z",
          status: "moderation_blocked",
          endTurn: true,
          generationId: "generation-rejected",
          requestId: "request-rejected",
        }),
      ],
      {
        scope: account,
        conversationId,
        mapping: mappingReviewed,
      },
    );

    expect(attempts).toHaveLength(1);
    expect(attempts[0]?.generationStarted).toBe(true);
    expect(attempts[0]?.outcome).toBe("rejected_after_start");
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

  it("preserves A-B-A observation, message, and attempt revision occurrences", () => {
    const directory = mkdtempSync(join(tmpdir(), "stage2b-aba-"));
    temporaryDirectories.push(directory);
    const ledger = new Ledger(join(directory, "usage.sqlite"));
    const account = scope("account-aba", "provider-user", "workspace", "quota");
    ledger.upsertAccount(account);

    for (const [index, finalModel] of ["model-a", "model-b", "model-a"].entries()) {
      ledger.ingestConversation(
        account,
        progressDetail(
          "conversation-aba",
          "finished_successfully",
          true,
          finalModel,
          `2026-09-07T10:0${index + 1}:00.000Z`,
        ),
        mappingReviewed,
        context(
          `run-aba-${index + 1}`,
          "conversation-aba",
          `2026-09-07T10:0${index + 1}:30.000Z`,
        ),
      );
    }

    const attempt = ledger.listAttempts(account)[0]!;
    expect(attempt.recordedFinalModelRaw).toBe("model-a");
    expect(
      ledger.attemptRevisions(account, String(attempt.attemptId)).map((item) =>
        String((item.payload as Record<string, unknown>).recordedFinalModelRaw),
      ),
    ).toEqual(["model-a", "model-b", "model-a"]);
    expect(
      ledger.messageRevisions(account, "conversation-aba", "message-final").map((item) =>
        String((item.payload as Record<string, unknown>).recordedFinalModelRaw),
      ),
    ).toEqual(["model-a", "model-b", "model-a"]);
    expect(
      ledger.db
        .prepare(
          "SELECT COUNT(*) AS count FROM observations WHERE scope_key=? AND source_kind=? AND source_id=?",
        )
        .get(scopeKey(account), "conversation_detail", "conversation-aba"),
    ).toEqual({ count: 3 });
    ledger.close();
  });

  it("does not let older evidence replace the current projection", () => {
    const directory = mkdtempSync(join(tmpdir(), "stage2b-stale-"));
    temporaryDirectories.push(directory);
    const ledger = new Ledger(join(directory, "usage.sqlite"));
    const account = scope("account-stale", "provider-user", "workspace", "quota");
    ledger.upsertAccount(account);

    ledger.ingestConversation(
      account,
      progressDetail(
        "conversation-stale",
        "finished_successfully",
        true,
        "model-new",
        "2026-09-07T10:02:00.000Z",
      ),
      mappingReviewed,
      context("run-stale-new", "conversation-stale", "2026-09-07T12:00:00.000Z"),
    );
    ledger.ingestConversation(
      account,
      progressDetail(
        "conversation-stale",
        "finished_successfully",
        true,
        "model-old",
        "2026-09-07T10:01:00.000Z",
      ),
      mappingReviewed,
      context("run-stale-old", "conversation-stale", "2026-09-07T11:00:00.000Z"),
    );

    const attempt = ledger.listAttempts(account)[0]!;
    expect(attempt.recordedFinalModelRaw).toBe("model-new");
    expect(
      ledger.messagesFor(account, "conversation-stale").find(
        (message) => message.messageId === "message-final",
      )?.recordedFinalModelRaw,
    ).toBe("model-new");
    expect(ledger.attemptRevisions(account, String(attempt.attemptId))).toHaveLength(1);
    expect(ledger.messageRevisions(account, "conversation-stale", "message-final")).toHaveLength(2);
    ledger.close();
  });

  it("allowlists persisted context provenance and quarantines future timestamps", () => {
    const directory = mkdtempSync(join(tmpdir(), "stage2b-provenance-"));
    temporaryDirectories.push(directory);
    const ledger = new Ledger(join(directory, "usage.sqlite"));
    const account = scope("account-provenance", "provider-user", "workspace", "quota");
    ledger.upsertAccount(account);

    ledger.ingestConversation(
      account,
      progressDetail(
        "conversation-provenance",
        "finished_successfully",
        true,
        "model-final",
        "2026-09-07T13:00:00.000Z",
      ),
      mappingReviewed,
      {
        ...context(
          "run-provenance",
          "conversation-provenance",
          "2026-09-07T12:00:00.000Z",
        ),
        provenance: {
          adapter_version: "chatgpt-chat-history-v1",
          detail_route: "modern",
          pagination_state: "complete",
          scopes: ["active"],
          fixture: "stage2b-synthetic",
          authorization: "Bearer should-not-persist",
          nested: { prompt: "should-not-persist" },
        },
      },
    );

    const observation = ledger.db
      .prepare("SELECT provenance_json FROM observations")
      .get() as { provenance_json: string };
    const provenance = JSON.parse(observation.provenance_json) as Record<string, unknown>;
    expect(provenance).toMatchObject({
      adapter_version: "chatgpt-chat-history-v1",
      detail_route: "modern",
      pagination_state: "complete",
      scopes: ["active"],
      fixture: "stage2b-synthetic",
      collector_account_id: "account-provenance",
    });
    expect(provenance.authorization).toBeUndefined();
    expect(provenance.nested).toBeUndefined();

    const finalMessage = ledger.messagesFor(account, "conversation-provenance").find(
      (message) => message.messageId === "message-final",
    );
    expect(finalMessage?.createdAt).toBeNull();
    const attempt = ledger.listAttempts(account)[0]!;
    expect(attempt.latestPossibleAt).toBe("2026-09-07T11:00:00.000Z");
    expect(attempt.warnings).toContain("future_timestamp_quarantined");
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
    const revisedAttempt = ledger.listAttempts(account)[0]!;
    const revisedPayload = ledger.attemptRevisions(
      account,
      String(revisedAttempt.attemptId),
    ).at(-1)?.payload as Record<string, unknown>;
    expect(revisedPayload.aliases).toEqual(revisedAttempt.aliases);
    expect(revisedPayload.evidenceMessageIds).toEqual(revisedAttempt.evidenceMessageIds);
    expect(revisedPayload.warnings).toEqual(revisedAttempt.warnings);
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
    expect(report.modelMismatches).toBe(0);
    expect(report.rawSlugDifferences).toBe(1);
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
    continuation: null,
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
    continuation: null,
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
