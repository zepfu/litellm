import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import {
  ChatGPTHistoryAdapter,
  type HistoryTransport,
} from "../../src/adapters/chatgpt/adapter.js";
import { FixtureTransport } from "../../src/adapters/chatgpt/fixture-transport.js";
import { HistoryCollector } from "../../src/history/collector.js";
import {
  SqliteCheckpointStore,
  MemoryCheckpointStore,
} from "../../src/history/checkpoints.js";
import { Ledger } from "../../src/ledger/store.js";
import type { LedgerScope } from "../../src/ledger/types.js";
import { defaultConfig } from "../../src/config.js";
import { collectIntoLedger } from "../../src/history/ingest.js";
import { emptyCapabilities } from "../../src/normalize/identity.js";
import type {
  AdaptedPage,
  ConversationDetailProjection,
  ConversationSummary,
  IdentityRecord,
  MessageRecord,
  PaginationState,
} from "../../src/contracts/records.js";
import type {
  DiscoveryCheckpoint,
  HistoryRange,
  HistoryReader,
  HistoryScope,
  RevisitEntry,
} from "../../src/contracts/history.js";

const NOW = new Date("2026-09-07T12:00:00.000Z");
const EXPLICIT_RANGE: HistoryRange = {
  start: "2026-09-01T00:00:00.000Z",
  end: "2026-09-08T00:00:00.000Z",
};
const LEDGER_SCOPE: LedgerScope = {
  collectorAccountId: "fixture-primary",
  provider: "openai",
  providerUserId: "user-abc123",
  workspaceId: "ws-xyz",
  quotaOwnerId: "user-abc123",
  surface: "chat",
};

class ScriptedReader implements HistoryReader {
  readonly capabilities = emptyCapabilities();
  readonly requests: Array<{ method: string; path: string }> = [];
  readonly indexPages = new Map<HistoryScope, Map<number, AdaptedPage<ConversationSummary>>>();
  readonly indexSequences = new Map<
    HistoryScope,
    AdaptedPage<ConversationSummary>[]
  >();
  readonly details = new Map<string, ConversationDetailProjection>();
  readonly messagePages = new Map<
    string,
    Map<string, AdaptedPage<MessageRecord>>
  >();
  readonly messageErrors = new Map<string, Error>();
  identity: IdentityRecord = {
    providerUserId: "user-abc123",
    workspaceId: "ws-xyz",
    quotaOwnerId: "user-abc123",
    surface: "chat",
    authState: "ready",
    identityErrors: [],
  };

  async inspectSessionIdentity(): Promise<IdentityRecord> {
    this.requests.push({ method: "GET", path: "/api/auth/session" });
    return this.identity;
  }

  async listConversations(options: {
    archived: boolean;
    offset?: number;
    limit?: number;
  }): Promise<AdaptedPage<ConversationSummary>> {
    const scope: HistoryScope = options.archived ? "archived" : "active";
    const offset = options.offset ?? 0;
    this.requests.push({
      method: "GET",
      path: `/backend-api/conversations?scope=${scope}&offset=${offset}`,
    });
    const sequence = this.indexSequences.get(scope);
    if (sequence && sequence.length > 0) {
      return sequence.shift()!;
    }
    return (
      this.indexPages.get(scope)?.get(offset) ??
      completePage<ConversationSummary>([])
    );
  }

  async fetchConversation(
    conversationId: string,
  ): Promise<ConversationDetailProjection> {
    this.requests.push({
      method: "GET",
      path: `/backend-api/conversations/${conversationId}`,
    });
    const detail = this.details.get(conversationId);
    if (!detail) {
      throw new Error("missing_detail");
    }
    return detail;
  }

  async fetchMessages(
    conversationId: string,
    options: { before?: string | null } = {},
  ): Promise<AdaptedPage<MessageRecord>> {
    const before = options.before ?? "latest";
    this.requests.push({
      method: "GET",
      path: `/backend-api/conversations/${conversationId}/messages?before=${before}`,
    });
    const error = this.messageErrors.get(`${conversationId}:${before}`);
    if (error) {
      throw error;
    }
    return (
      this.messagePages.get(conversationId)?.get(before) ??
      completePage<MessageRecord>([])
    );
  }
}

function summary(
  conversationId: string,
  options: Partial<ConversationSummary> = {},
): ConversationSummary {
  return {
    conversationId,
    createdAt: options.createdAt ?? "2025-01-01T00:00:00.000Z",
    updatedAt: options.updatedAt ?? "2026-09-06T12:00:00.000Z",
    isArchived: options.isArchived ?? false,
    workspaceId: options.workspaceId ?? "ws-xyz",
    projectId: options.projectId ?? null,
    surface: options.surface ?? "chat",
    origin: options.origin ?? null,
    hasVersions: options.hasVersions ?? null,
    currentNode: options.currentNode ?? null,
    coverage: options.coverage ?? "validated_page",
  };
}

function message(
  conversationId: string,
  messageId: string,
): MessageRecord {
  return {
    conversationId,
    messageId,
    nodeId: messageId,
    parentId: null,
    children: [],
    role: "assistant",
    channel: null,
    createdAt: "2026-09-06T12:00:05.000Z",
    status: "finished_successfully",
    endTurn: true,
    requestedModelRaw: null,
    requestedModeRaw: null,
    requestedReasoningEffortRaw: null,
    recordedFinalModelRaw: "gpt-5.6-astra-pro",
    generationId: null,
    requestId: null,
    surface: "chat",
    origin: null,
    metadata: {},
  };
}

function detail(
  conversationId: string,
  detailRoute: "modern" | "legacy" = "modern",
): ConversationDetailProjection {
  return {
    conversationId,
    createdAt: "2025-01-01T00:00:00.000Z",
    updatedAt: "2026-09-06T12:00:00.000Z",
    currentNode: null,
    surface: "chat",
    detailRoute,
    messages: [],
    continuation: null,
    paginationState: "complete",
    coverage: "validated_page",
    warnings: [],
  };
}

function completePage<T>(items: T[]): AdaptedPage<T> {
  return {
    items,
    continuation: null,
    exhausted: true,
    paginationState: "complete",
    schemaVersion: "chatgpt-chat-history-v1",
    coverage: "validated_page",
    warnings: [],
  };
}

function continuationPage<T>(
  items: T[],
  cursor: string,
): AdaptedPage<T> {
  return {
    items,
    continuation: cursor,
    exhausted: false,
    paginationState: "continuation",
    schemaVersion: "chatgpt-chat-history-v1",
    coverage: "validated_page",
    warnings: [],
  };
}

function indexPage(
  items: ConversationSummary[],
  continuation: number | null = null,
  exhausted = true,
  paginationState: PaginationState = exhausted ? "complete" : "continuation",
): AdaptedPage<ConversationSummary> {
  return {
    items,
    continuation,
    exhausted,
    paginationState,
    schemaVersion: "chatgpt-chat-history-v1",
    coverage: paginationState === "complete" ? "validated_page" : "partial",
    warnings: [],
  };
}

function addCompleteIndex(
  reader: ScriptedReader,
  active: ConversationSummary[],
  archived: ConversationSummary[],
): void {
  reader.indexPages.set("active", new Map([[0, indexPage(active)]]));
  reader.indexPages.set("archived", new Map([[0, indexPage(archived)]]));
}

describe("Stage-2A history collection", () => {
  const temporaryDirectories: string[] = [];

  afterEach(() => {
    for (const directory of temporaryDirectories.splice(0)) {
      rmSync(directory, { recursive: true, force: true });
    }
  });

  it("collects both scopes, deduplicates shared conversations, and exposes project/branch coverage", async () => {
    const reader = new ScriptedReader();
    const lateVisible = summary("conv-late-visible", {
      projectId: "project-1",
      hasVersions: true,
    });
    const active = summary("conv-active", {
      hasVersions: false,
      updatedAt: "2026-09-05T12:00:00.000Z",
    });
    addCompleteIndex(reader, [lateVisible, active], [summary("conv-late-visible", {
      isArchived: true,
      projectId: "project-1",
      hasVersions: true,
    })]);
    reader.details.set("conv-late-visible", detail("conv-late-visible"));
    reader.details.set("conv-active", detail("conv-active"));
    reader.messagePages.set(
      "conv-late-visible",
      new Map([["latest", completePage([message("conv-late-visible", "msg-late")])]]),
    );
    reader.messagePages.set(
      "conv-active",
      new Map([["latest", completePage([message("conv-active", "msg-active")])]]),
    );

    const store = new MemoryCheckpointStore();
    const collector = new HistoryCollector(reader, {
      accountId: "fixture-primary",
      store,
      clock: { now: () => NOW },
    });
    const result = await collector.collect({
      mode: "backfill",
      range: EXPLICIT_RANGE,
    });

    expect(result.status).toBe("partial");
    expect(result.conversations).toHaveLength(2);
    expect(
      result.conversations.find(
        (conversation) => conversation.summary.conversationId === "conv-late-visible",
      )?.scopes,
    ).toEqual(["active", "archived"]);
    expect(result.coverage.projects).toBe("validated_for_discovered_projects");
    expect(result.coverage.branches).toBe("version_metadata_observed");
    expect(result.coverage.overall).toBe("partial");
    expect(result.coverage.gaps).toContain("project_visibility_unproven");
    expect(result.coverage.gaps).toContain("branch_visibility_unproven");
    expect(store.loadDiscovery("active")?.lastCompleteDiscoveryStartedAt).toBeNull();
    expect(store.loadDiscovery("archived")?.lastCompleteDiscoveryStartedAt).toBeNull();
  });

  it("does not advance an incomplete index checkpoint after a page budget", async () => {
    const reader = new ScriptedReader();
    const first = summary("conv-page-budget");
    reader.indexPages.set(
      "active",
      new Map([[0, indexPage([first], 1, false, "continuation")]]),
    );
    reader.indexPages.set("archived", new Map([[0, indexPage([])]]));
    reader.details.set("conv-page-budget", detail("conv-page-budget"));
    reader.messagePages.set(
      "conv-page-budget",
      new Map([["latest", completePage([])]]),
    );

    const store = new MemoryCheckpointStore();
    const result = await new HistoryCollector(reader, {
      accountId: "fixture-primary",
      store,
      clock: { now: () => NOW },
      maxIndexPagesPerScope: 1,
    }).collect({ mode: "incremental" });

    const checkpoint = store.loadDiscovery("active");
    expect(result.status).toBe("partial");
    expect(checkpoint?.status).toBe("partial");
    expect(checkpoint?.continuation).toBe(1);
    expect(checkpoint?.lastCompleteDiscoveryStartedAt).toBeNull();
    expect(checkpoint?.paginationState).toBe("budget_exhausted");
  });

  it("stops at a message page budget and queues the conversation for revisit", async () => {
    const reader = new ScriptedReader();
    const candidate = summary("conv-message-budget");
    addCompleteIndex(reader, [candidate], []);
    reader.details.set("conv-message-budget", detail("conv-message-budget"));
    reader.messagePages.set(
      "conv-message-budget",
      new Map([
        [
          "latest",
          continuationPage(
            [message("conv-message-budget", "msg-1")],
            "cursor-1",
          ),
        ],
        [
          "cursor-1",
          continuationPage(
            [message("conv-message-budget", "msg-2")],
            "cursor-2",
          ),
        ],
      ]),
    );

    const store = new MemoryCheckpointStore();
    const commits: Array<{
      pageKind: string;
      warnings: string[];
      nextContinuation: string | null;
    }> = [];
    const result = await new HistoryCollector(reader, {
      accountId: "fixture-primary",
      store,
      clock: { now: () => NOW },
      maxMessagePagesPerConversation: 1,
      onPageCommit: (page) => {
        commits.push({
          pageKind: page.pageKind,
          warnings: page.warnings,
          nextContinuation: page.nextContinuation,
        });
      },
    }).collect({
      mode: "backfill",
      range: EXPLICIT_RANGE,
    });

    expect(result.detailPagesFetched).toBe(1);
    expect(result.revisits).toHaveLength(1);
    expect(result.revisits[0]?.reason).toBe("page_budget");
    expect(
      reader.requests.filter((request) =>
        request.path.includes("conv-message-budget/messages"),
      ),
    ).toHaveLength(1);
    expect(commits).toContainEqual({
      pageKind: "messages",
      warnings: ["message_page_budget_exhausted"],
      nextContinuation: "cursor-1",
    });
  });

  it("records a repeated index offset as partial without looping", async () => {
    const reader = new ScriptedReader();
    const candidate = summary("conv-repeated-index");
    reader.indexPages.set(
      "active",
      new Map([[0, indexPage([candidate], 0, false, "continuation")]]),
    );
    reader.indexPages.set("archived", new Map([[0, indexPage([])]]));
    reader.details.set("conv-repeated-index", detail("conv-repeated-index"));
    reader.messagePages.set(
      "conv-repeated-index",
      new Map([["latest", completePage([])]]),
    );

    const store = new MemoryCheckpointStore();
    const result = await new HistoryCollector(reader, {
      accountId: "fixture-primary",
      store,
      clock: { now: () => NOW },
    }).collect({
      mode: "backfill",
      range: EXPLICIT_RANGE,
    });

    expect(result.scopes.find((scope) => scope.scope === "active")).toMatchObject({
      coverage: "partial",
      paginationState: "repeated_cursor",
      continuation: 0,
    });
    expect(
      reader.requests.filter((request) => request.path.includes("scope=active")),
    ).toHaveLength(1);
  });

  it("acquires a conversation that becomes visible on a later index page", async () => {
    const reader = new ScriptedReader();
    const firstPage = summary("conv-first-page", {
      updatedAt: "2026-09-02T12:00:00.000Z",
    });
    const lateVisible = summary("conv-late-page", {
      projectId: "project-late",
      hasVersions: true,
    });
    reader.indexPages.set(
      "active",
      new Map([
        [0, indexPage([firstPage], 1, false, "continuation")],
        [1, indexPage([lateVisible])],
      ]),
    );
    reader.indexPages.set("archived", new Map([[0, indexPage([])]]));
    for (const conversation of [firstPage, lateVisible]) {
      reader.details.set(
        conversation.conversationId,
        detail(conversation.conversationId),
      );
      reader.messagePages.set(
        conversation.conversationId,
        new Map([["latest", completePage([])]]),
      );
    }

    const store = new MemoryCheckpointStore();
    const result = await new HistoryCollector(reader, {
      accountId: "fixture-primary",
      store,
      clock: { now: () => NOW },
    }).collect({
      mode: "backfill",
      range: EXPLICIT_RANGE,
    });

    expect(
      result.conversations.map(
        (conversation) => conversation.summary.conversationId,
      ),
    ).toEqual(["conv-first-page", "conv-late-page"]);
    expect(result.scopes.find((scope) => scope.scope === "active")).toMatchObject({
      coverage: "complete",
      pagesFetched: 3,
    });
  });

  it("discovers historical summaries beyond the range end and bounds message evidence at that end", async () => {
    const reader = new ScriptedReader();
    const committedMessages: string[][] = [];
    const candidate = summary("conv-historical", {
      updatedAt: "2026-09-10T12:00:00.000Z",
    });
    addCompleteIndex(reader, [candidate], []);
    reader.details.set("conv-historical", {
      ...detail("conv-historical"),
      messages: [
        {
          ...message("conv-historical", "msg-before-end"),
          createdAt: "2026-09-07T23:59:59.000Z",
        },
        {
          ...message("conv-historical", "msg-at-end"),
          createdAt: EXPLICIT_RANGE.end,
        },
        {
          ...message("conv-historical", "msg-after-end"),
          createdAt: "2026-09-08T00:00:01.000Z",
        },
      ],
    });
    reader.messagePages.set(
      "conv-historical",
      new Map([["latest", completePage([])]]),
    );

    const result = await new HistoryCollector(reader, {
      accountId: "fixture-primary",
      store: new MemoryCheckpointStore(),
      clock: { now: () => NOW },
      onPageCommit: (page) => {
        committedMessages.push(page.messages.map((item) => item.messageId));
      },
    }).collect({
      mode: "backfill",
      range: EXPLICIT_RANGE,
    });

    expect(result.conversations).toHaveLength(1);
    expect(result.conversations[0]?.messages.map((item) => item.messageId)).toEqual([
      "msg-before-end",
    ]);
    expect(
      result.conversations[0]?.detail?.messages.map((item) => item.messageId),
    ).toEqual(["msg-before-end"]);
    expect(committedMessages.flat()).toEqual(["msg-before-end"]);
  });

  it("rereads the leading index page and leaves changing scans partial", async () => {
    const reader = new ScriptedReader();
    const firstPage = summary("conv-leading", {
      updatedAt: "2026-09-02T12:00:00.000Z",
    });
    const changedLeadingPage = summary("conv-leading", {
      updatedAt: "2026-09-03T12:00:00.000Z",
    });
    const laterPage = summary("conv-later");
    reader.indexSequences.set("active", [
      indexPage([firstPage], 1, false, "continuation"),
      indexPage([laterPage]),
      indexPage([changedLeadingPage], 1, false, "continuation"),
    ]);
    reader.indexPages.set("archived", new Map([[0, indexPage([])]]));
    for (const conversation of [firstPage, laterPage, changedLeadingPage]) {
      reader.details.set(
        conversation.conversationId,
        detail(conversation.conversationId),
      );
      reader.messagePages.set(
        conversation.conversationId,
        new Map([["latest", completePage([])]]),
      );
    }

    const store = new MemoryCheckpointStore();
    const result = await new HistoryCollector(reader, {
      accountId: "fixture-primary",
      store,
      clock: { now: () => NOW },
    }).collect({
      mode: "backfill",
      range: EXPLICIT_RANGE,
    });

    expect(result.status).toBe("partial");
    expect(result.scopes.find((scope) => scope.scope === "active")).toMatchObject({
      coverage: "partial",
      continuation: 0,
      pagesFetched: 3,
    });
    expect(result.warnings).toContain("leading_index_changed_during_scan");
    expect(store.loadDiscovery("active")?.lastCompleteDiscoveryStartedAt).toBeNull();
    expect(
      reader.requests.filter((request) => request.path.includes("scope=active")),
    ).toHaveLength(3);
  });

  it("queues incomplete message revisits and clears them after a later complete pass", async () => {
    const reader = new ScriptedReader();
    const candidate = summary("conv-incomplete");
    addCompleteIndex(reader, [candidate], []);
    reader.details.set("conv-incomplete", detail("conv-incomplete"));
    reader.messagePages.set(
      "conv-incomplete",
      new Map([
        ["latest", continuationPage([message("conv-incomplete", "msg-1")], "cursor-1")],
        ["cursor-1", completePage([message("conv-incomplete", "msg-2")])],
      ]),
    );

    const store = new MemoryCheckpointStore();
    const collector = new HistoryCollector(reader, {
      accountId: "fixture-primary",
      store,
      clock: { now: () => NOW },
      maxMessagePagesPerConversation: 1,
    });
    const first = await collector.collect({
      mode: "backfill",
      range: EXPLICIT_RANGE,
    });
    expect(first.revisits).toHaveLength(1);
    expect(first.revisits[0]?.reason).toBe("page_budget");
    expect(first.conversations[0]?.coverage).toBe("partial");

    reader.messagePages.set(
      "conv-incomplete",
      new Map([["cursor-1", completePage([message("conv-incomplete", "msg-3")])]]),
    );
    const requestsBeforeSecondPass = reader.requests.length;
    const second = await collector.collect({
      mode: "incremental",
      now: new Date("2026-09-07T13:00:00.000Z"),
    });
    expect(second.revisits).toHaveLength(0);
    expect(second.conversations[0]?.coverage).toBe("complete");
    expect(
      reader.requests
        .slice(requestsBeforeSecondPass)
        .filter((request) => request.path.includes("conv-incomplete/messages")),
    ).toEqual([
      {
        method: "GET",
        path: "/backend-api/conversations/conv-incomplete/messages?before=cursor-1",
      },
    ]);
  });

  it("retains a revisit when detail is partial or unrecognized even if messages finish", async () => {
    const reader = new ScriptedReader();
    const candidate = summary("conv-unrecognized-detail");
    addCompleteIndex(reader, [candidate], []);
    reader.details.set("conv-unrecognized-detail", {
      ...detail("conv-unrecognized-detail"),
      coverage: "unrecognized",
      paginationState: "unknown",
      warnings: ["unrecognized_detail_shape"],
    });
    reader.messagePages.set(
      "conv-unrecognized-detail",
      new Map([["latest", completePage([])]]),
    );

    const result = await new HistoryCollector(reader, {
      accountId: "fixture-primary",
      store: new MemoryCheckpointStore(),
      clock: { now: () => NOW },
    }).collect({ mode: "backfill", range: EXPLICIT_RANGE });

    expect(result.conversations[0]?.coverage).toBe("partial");
    expect(result.revisits).toHaveLength(1);
    expect(result.revisits[0]).toMatchObject({
      conversationId: "conv-unrecognized-detail",
      reason: "unrecognized_detail",
      continuation: null,
    });
  });

  it("retains a revisit when any message page is only partially recognized", async () => {
    const reader = new ScriptedReader();
    const candidate = summary("conv-partial-message-page");
    addCompleteIndex(reader, [candidate], []);
    reader.details.set("conv-partial-message-page", detail("conv-partial-message-page"));
    reader.messagePages.set(
      "conv-partial-message-page",
      new Map([
        [
          "latest",
          {
            ...continuationPage([], "cursor-1"),
            warnings: ["message_shape_warning"],
          },
        ],
        ["cursor-1", completePage([])],
      ]),
    );

    const result = await new HistoryCollector(reader, {
      accountId: "fixture-primary",
      store: new MemoryCheckpointStore(),
      clock: { now: () => NOW },
    }).collect({ mode: "backfill", range: EXPLICIT_RANGE });

    expect(result.revisits).toHaveLength(1);
    expect(result.revisits[0]).toMatchObject({
      conversationId: "conv-partial-message-page",
      reason: "partial_detail",
      continuation: null,
    });
    expect(result.conversations[0]?.coverage).toBe("partial");
  });

  it("retains a revisit for an outstanding nonterminal generation after pagination completes", async () => {
    const reader = new ScriptedReader();
    const candidate = summary("conv-outstanding-generation");
    addCompleteIndex(reader, [candidate], []);
    reader.details.set("conv-outstanding-generation", detail("conv-outstanding-generation"));
    reader.messagePages.set(
      "conv-outstanding-generation",
      new Map([
        [
          "latest",
          completePage([
            {
              ...message("conv-outstanding-generation", "msg-pending"),
              status: "in_progress",
              endTurn: false,
            },
          ]),
        ],
      ]),
    );

    const result = await new HistoryCollector(reader, {
      accountId: "fixture-primary",
      store: new MemoryCheckpointStore(),
      clock: { now: () => NOW },
    }).collect({ mode: "backfill", range: EXPLICIT_RANGE });

    expect(result.revisits).toHaveLength(1);
    expect(result.revisits[0]?.reason).toBe("nonterminal_generation");
    expect(result.conversations[0]?.coverage).toBe("partial");
  });

  it("invalidates one bad saved continuation and performs only one bounded restart", async () => {
    const reader = new ScriptedReader();
    const candidate = summary("conv-bad-continuation");
    addCompleteIndex(reader, [candidate], []);
    reader.details.set("conv-bad-continuation", detail("conv-bad-continuation"));
    reader.messagePages.set(
      "conv-bad-continuation",
      new Map([
        [
          "stale-cursor",
          continuationPage(
            [message("conv-bad-continuation", "msg-stale")],
            "stale-cursor",
          ),
        ],
        ["latest", completePage([message("conv-bad-continuation", "msg-fresh")])],
      ]),
    );
    const store = new MemoryCheckpointStore();
    store.upsertRevisit({
      stateVersion: 1,
      accountId: "fixture-primary",
      conversationId: "conv-bad-continuation",
      scopes: ["active"],
      status: "pending",
      reason: "page_budget",
      firstSeenAt: NOW.toISOString(),
      lastSeenAt: NOW.toISOString(),
      attempts: 1,
      nextEligibleAt: NOW.toISOString(),
      lastError: null,
      detailPagesFetched: 1,
      continuation: "stale-cursor",
    });

    const result = await new HistoryCollector(reader, {
      accountId: "fixture-primary",
      store,
      clock: { now: () => NOW },
    }).collect({ mode: "incremental" });

    expect(
      reader.requests
        .filter((request) => request.path.includes("conv-bad-continuation/messages"))
        .map((request) => request.path),
    ).toEqual([
      "/backend-api/conversations/conv-bad-continuation/messages?before=stale-cursor",
      "/backend-api/conversations/conv-bad-continuation/messages?before=latest",
    ]);
    expect(result.revisits).toHaveLength(0);
    expect(result.conversations[0]?.warnings).toContain(
      "messages_bad_continuation_restarting",
    );
  });

  it("rotates opt-in older-history audit progress and reports its coverage", async () => {
    const reader = new ScriptedReader();
    const recent = summary("conv-recent-audit", {
      updatedAt: "2026-09-05T12:00:00.000Z",
    });
    const older = summary("conv-older-audit", {
      updatedAt: "2026-08-01T12:00:00.000Z",
    });
    reader.indexPages.set(
      "active",
      new Map([
        [0, indexPage([recent], 1, false, "continuation")],
        [1, indexPage([older])],
      ]),
    );
    reader.indexPages.set("archived", new Map([[0, indexPage([])]]));
    for (const conversation of [recent, older]) {
      reader.details.set(
        conversation.conversationId,
        detail(conversation.conversationId),
      );
      reader.messagePages.set(
        conversation.conversationId,
        new Map([["latest", completePage([])]]),
      );
    }
    const store = new MemoryCheckpointStore();
    const collector = new HistoryCollector(reader, {
      accountId: "fixture-primary",
      store,
      clock: { now: () => NOW },
      maxIndexPagesPerScope: 1,
    });

    const first = await collector.collect({
      mode: "incremental",
      olderHistoryAudit: { enabled: true, maxPages: 1 },
    });
    expect(first.coverage.active.olderHistoryAudit).toMatchObject({
      enabled: true,
      status: "partial",
      continuation: 1,
    });

    const second = await collector.collect({
      mode: "incremental",
      now: new Date("2026-09-07T13:00:00.000Z"),
      olderHistoryAudit: { enabled: true, maxPages: 1 },
    });
    expect(second.coverage.olderHistoryAudit.active.status).toBe("complete");
    expect(second.conversations.map((item) => item.summary.conversationId)).toContain(
      "conv-older-audit",
    );
    expect(store.loadDiscovery("active")?.olderHistoryAudit?.lastCompletedAt).toBe(
      "2026-09-07T13:00:00.000Z",
    );
  });

  it("uses the 48-hour overlap for incremental discovery but honors explicit ranges over newer watermarks", async () => {
    const reader = new ScriptedReader();
    const inOverlap = summary("conv-in-overlap", {
      updatedAt: "2026-09-04T12:00:00.000Z",
    });
    const outsideOverlap = summary("conv-outside-overlap", {
      updatedAt: "2026-09-03T11:59:59.000Z",
    });
    addCompleteIndex(reader, [inOverlap, outsideOverlap], []);
    reader.details.set("conv-in-overlap", detail("conv-in-overlap"));
    reader.details.set("conv-outside-overlap", detail("conv-outside-overlap"));
    reader.messagePages.set(
      "conv-in-overlap",
      new Map([["latest", completePage([])]]),
    );
    reader.messagePages.set(
      "conv-outside-overlap",
      new Map([["latest", completePage([])]]),
    );

    const store = new MemoryCheckpointStore();
    const prior: DiscoveryCheckpoint = {
      stateVersion: 1,
      accountId: "fixture-primary",
      scope: "active",
      status: "complete",
      mode: "backfill",
      range: EXPLICIT_RANGE,
      candidateCutoff: EXPLICIT_RANGE.start,
      scanStartedAt: "2026-09-06T12:00:00.000Z",
      continuation: null,
      pagesFetched: 1,
      pageBudget: 500,
      lastCompleteDiscoveryStartedAt: "2026-09-06T12:00:00.000Z",
      lastPageAt: "2026-09-06T12:00:00.000Z",
      paginationState: "complete",
      warnings: [],
      updatedAt: "2026-09-06T12:00:00.000Z",
    };
    store.saveDiscovery(prior);
    store.saveDiscovery({ ...prior, scope: "archived" });

    const collector = new HistoryCollector(reader, {
      accountId: "fixture-primary",
      store,
      clock: { now: () => NOW },
    });
    const incremental = await collector.collect({ mode: "incremental" });
    expect(incremental.conversations.map((item) => item.summary.conversationId)).toContain(
      "conv-in-overlap",
    );
    expect(incremental.conversations.map((item) => item.summary.conversationId)).not.toContain(
      "conv-outside-overlap",
    );

    const explicit = await collector.collect({
      mode: "reconciliation",
      range: {
        start: "2026-09-03T00:00:00.000Z",
        end: "2026-09-04T00:00:00.000Z",
      },
    });
    expect(explicit.range.start).toBe("2026-09-03T00:00:00.000Z");
    expect(explicit.conversations).toHaveLength(2);
    expect(explicit.conversations.map((item) => item.summary.conversationId)).toEqual([
      "conv-in-overlap",
      "conv-outside-overlap",
    ]);
  });

  it("refreshes from an old watermark minus overlap without the default-range clamp", async () => {
    const reader = new ScriptedReader();
    const candidate = summary("conv-old-watermark", {
      updatedAt: "2026-08-15T12:00:00.000Z",
      hasVersions: false,
    });
    addCompleteIndex(reader, [candidate], []);
    reader.details.set("conv-old-watermark", detail("conv-old-watermark"));
    reader.messagePages.set(
      "conv-old-watermark",
      new Map([["latest", completePage([])]]),
    );

    const oldWatermark = "2026-08-01T12:00:00.000Z";
    const store = new MemoryCheckpointStore();
    store.saveDiscovery(
      testCheckpoint("active", {
        lastCompleteDiscoveryStartedAt: oldWatermark,
      }),
    );
    store.saveDiscovery(
      testCheckpoint("archived", {
        lastCompleteDiscoveryStartedAt: oldWatermark,
      }),
    );

    const result = await new HistoryCollector(reader, {
      accountId: "fixture-primary",
      store,
      clock: { now: () => NOW },
    }).collect({ mode: "incremental" });

    expect(result.scopes.find((scope) => scope.scope === "active")).toMatchObject({
      candidateCutoff: "2026-07-30T12:00:00.000Z",
      coverage: "complete",
    });
    expect(result.conversations.map((item) => item.summary.conversationId)).toEqual([
      "conv-old-watermark",
    ]);
    expect(store.loadDiscovery("active")?.lastCompleteDiscoveryStartedAt).toBe(
      NOW.toISOString(),
    );
    expect(store.loadDiscovery("archived")?.lastCompleteDiscoveryStartedAt).toBe(
      NOW.toISOString(),
    );
  });

  it("resumes only when the frozen mode, range, and cutoff all match", async () => {
    const reader = new ScriptedReader();
    const candidate = summary("conv-resume");
    reader.indexPages.set(
      "active",
      new Map([
        [0, indexPage([candidate], 1, false, "continuation")],
        [1, indexPage([candidate])],
      ]),
    );
    reader.indexPages.set("archived", new Map([[0, indexPage([])]]));
    reader.details.set("conv-resume", detail("conv-resume"));
    reader.messagePages.set(
      "conv-resume",
      new Map([["latest", completePage([])]]),
    );

    const store = new MemoryCheckpointStore();
    const partial = testCheckpoint("active", {
      status: "partial",
      mode: "backfill",
      range: EXPLICIT_RANGE,
      candidateCutoff: EXPLICIT_RANGE.start,
      continuation: 1,
      paginationState: "budget_exhausted",
      lastCompleteDiscoveryStartedAt: null,
    });

    store.saveDiscovery(partial);
    await new HistoryCollector(reader, {
      accountId: "fixture-primary",
      store,
      clock: { now: () => NOW },
    }).collect({ mode: "backfill", range: EXPLICIT_RANGE });
    expect(
      reader.requests.find((request) => request.path.includes("scope=active"))?.path,
    ).toContain("offset=1");

    reader.requests.length = 0;
    for (const mismatch of [
      { mode: "reconciliation" as const },
      {
        range: {
          start: "2026-09-02T00:00:00.000Z",
          end: EXPLICIT_RANGE.end,
        },
      },
      { candidateCutoff: "2026-08-31T00:00:00.000Z" },
    ]) {
      store.saveDiscovery({ ...partial, ...mismatch });
      await new HistoryCollector(reader, {
        accountId: "fixture-primary",
        store,
        clock: { now: () => NOW },
      }).collect({ mode: "backfill", range: EXPLICIT_RANGE });
      expect(
        reader.requests.find((request) => request.path.includes("scope=active"))?.path,
      ).toContain("offset=0");
      reader.requests.length = 0;
    }
  });

  it("does not advance a prior watermark after an unrecognized terminal page", async () => {
    const reader = new ScriptedReader();
    const candidate = summary("conv-malformed-index");
    reader.indexPages.set(
      "active",
      new Map([
        [
          0,
          {
            ...indexPage([candidate]),
            coverage: "unrecognized",
          },
        ],
      ]),
    );
    reader.indexPages.set("archived", new Map([[0, indexPage([])]]));
    reader.details.set("conv-malformed-index", detail("conv-malformed-index"));
    reader.messagePages.set(
      "conv-malformed-index",
      new Map([["latest", completePage([])]]),
    );

    const priorWatermark = "2026-09-06T12:00:00.000Z";
    const store = new MemoryCheckpointStore();
    store.saveDiscovery(
      testCheckpoint("active", {
        lastCompleteDiscoveryStartedAt: priorWatermark,
      }),
    );
    store.saveDiscovery(
      testCheckpoint("archived", {
        lastCompleteDiscoveryStartedAt: priorWatermark,
      }),
    );

    const result = await new HistoryCollector(reader, {
      accountId: "fixture-primary",
      store,
      clock: { now: () => NOW },
    }).collect({ mode: "incremental" });

    expect(result.scopes.find((scope) => scope.scope === "active")).toMatchObject({
      coverage: "partial",
      warnings: ["index_unrecognized_page", "index_unvalidated_terminal_page"],
    });
    expect(store.loadDiscovery("active")?.lastCompleteDiscoveryStartedAt).toBe(
      priorWatermark,
    );
  });

  it("uses the fourteen-day reconciliation range while acquiring outstanding work", async () => {
    const reader = new ScriptedReader();
    addCompleteIndex(reader, [], []);
    reader.details.set("conv-outstanding", detail("conv-outstanding"));
    reader.messagePages.set(
      "conv-outstanding",
      new Map([["latest", completePage([])]]),
    );
    const store = new MemoryCheckpointStore();
    store.upsertRevisit({
      stateVersion: 1,
      accountId: "fixture-primary",
      conversationId: "conv-outstanding",
      scopes: ["active"],
      status: "pending",
      reason: "incomplete_detail",
      firstSeenAt: "2026-08-01T00:00:00.000Z",
      lastSeenAt: "2026-09-06T12:00:00.000Z",
      attempts: 1,
      nextEligibleAt: NOW.toISOString(),
      lastError: "adapter_error",
      detailPagesFetched: 1,
    });

    const result = await new HistoryCollector(reader, {
      accountId: "fixture-primary",
      store,
      clock: { now: () => NOW },
    }).collect({ mode: "reconciliation" });

    expect(result.range).toEqual({
      start: "2026-08-24T12:00:00.000Z",
      end: NOW.toISOString(),
    });
    expect(result.conversations.map((item) => item.summary.conversationId)).toEqual([
      "conv-outstanding",
    ]);
    expect(result.revisits).toEqual([]);
  });

  it("persists per-scope checkpoints and revisits across store instances", () => {
    const directory = mkdtempSync(join(tmpdir(), "usage-capture-history-"));
    temporaryDirectories.push(directory);
    const path = join(directory, "history.sqlite");
    const ledger = new Ledger(path);
    ledger.upsertAccount(LEDGER_SCOPE);
    const first = new SqliteCheckpointStore(ledger, LEDGER_SCOPE);
    const checkpoint: DiscoveryCheckpoint = {
      stateVersion: 1,
      accountId: "fixture-primary",
      scope: "active",
      status: "partial",
      mode: "incremental",
      range: EXPLICIT_RANGE,
      candidateCutoff: EXPLICIT_RANGE.start,
      scanStartedAt: NOW.toISOString(),
      continuation: 100,
      pagesFetched: 1,
      pageBudget: 1,
      lastCompleteDiscoveryStartedAt: null,
      lastPageAt: NOW.toISOString(),
      paginationState: "budget_exhausted",
      warnings: ["index_page_budget_exhausted"],
      updatedAt: NOW.toISOString(),
    };
    const revisit: RevisitEntry = {
      stateVersion: 1,
      accountId: "fixture-primary",
      conversationId: "conv-revisit",
      scopes: ["active"],
      status: "pending",
      reason: "incomplete_detail",
      firstSeenAt: NOW.toISOString(),
      lastSeenAt: NOW.toISOString(),
      attempts: 1,
      nextEligibleAt: NOW.toISOString(),
      lastError: "adapter_error",
      detailPagesFetched: 1,
      continuation: "cursor-1",
    };
    first.saveDiscovery(checkpoint);
    first.upsertRevisit(revisit);
    expect(new SqliteCheckpointStore(ledger, LEDGER_SCOPE).loadDiscovery("active"))
      .toBeNull();
    ledger.transaction(() => first.persist());
    const otherScope = { ...LEDGER_SCOPE, collectorAccountId: "fixture-other" };
    ledger.upsertAccount(otherScope);
    const other = new SqliteCheckpointStore(ledger, otherScope);
    expect(other.loadDiscovery("active")).toBeNull();
    expect(other.listRevisits()).toEqual([]);
    other.saveDiscovery({ ...checkpoint, accountId: "fixture-other", continuation: 200 });
    ledger.transaction(() => other.persist());
    ledger.close();

    const reopened = new Ledger(path);
    const second = new SqliteCheckpointStore(reopened, LEDGER_SCOPE);
    expect(second.loadDiscovery("active")).toEqual(checkpoint);
    expect(second.listRevisits()).toEqual([revisit]);
    expect(new SqliteCheckpointStore(reopened, otherScope).loadDiscovery("active")?.continuation)
      .toBe(200);
    second.completeRevisit("fixture-primary", "conv-revisit");
    expect(second.listRevisits()).toEqual([]);
    reopened.transaction(() => second.persist());
    expect(new SqliteCheckpointStore(reopened, LEDGER_SCOPE).listRevisits()).toEqual([]);
    reopened.close();
  });

  it("uses only GET fixture requests for acquisition", async () => {
    const requests: Array<{ method: string; path: string }> = [];
    const transport = new RecordingFixtureTransport(
      new FixtureTransport(
        new URL("../fixtures/v1/", import.meta.url).pathname,
      ),
      requests,
    );
    const adapter = new ChatGPTHistoryAdapter(transport, {
      providerUserId: "user-abc123",
      workspaceId: "ws-xyz",
      quotaOwnerId: "user-abc123",
    });
    const result = await new HistoryCollector(adapter, {
      accountId: "fixture-primary",
      store: new MemoryCheckpointStore(),
      clock: { now: () => NOW },
      legacyFallbackApproved: true,
    }).collect({
      mode: "backfill",
      range: EXPLICIT_RANGE,
    });

    expect(result.conversations).toHaveLength(2);
    expect(requests.every((request) => request.method === "GET")).toBe(true);
    expect(requests.some((request) => request.path.includes("/delete"))).toBe(false);
    const messages = result.conversations[0]!.messages;
    expect(messages.find((message) => message.messageId === "msg-002")?.parentId)
      .toBe("node-001");
    expect(messages.find((message) => message.messageId === "msg-002")?.nodeId)
      .toBe("node-002");
  });

  it("commits each successful page with its checkpoint when a later page fails", async () => {
    const directory = mkdtempSync(join(tmpdir(), "usage-capture-rollback-"));
    temporaryDirectories.push(directory);
    const ledger = new Ledger(join(directory, "usage.sqlite"));
    const account = defaultConfig().accounts[0]!;
    Object.assign(account, {
      id: LEDGER_SCOPE.collectorAccountId,
      expectedProviderUserId: LEDGER_SCOPE.providerUserId,
      expectedWorkspaceId: LEDGER_SCOPE.workspaceId,
      quotaOwnerId: LEDGER_SCOPE.quotaOwnerId,
    });
    const adapter = new ChatGPTHistoryAdapter(
      new FixtureTransport(new URL("../fixtures/v1/", import.meta.url).pathname),
      {
        providerUserId: LEDGER_SCOPE.providerUserId,
        workspaceId: LEDGER_SCOPE.workspaceId,
        quotaOwnerId: LEDGER_SCOPE.quotaOwnerId,
      },
    );
    try {
      ledger.db.exec(`
        CREATE TRIGGER fail_second_conversation BEFORE INSERT ON message_records
        WHEN NEW.conversation_id='conv-002'
        BEGIN SELECT RAISE(ABORT, 'injected ingestion failure'); END
      `);
      await expect(collectIntoLedger(adapter, ledger, account, {
        mode: "backfill", range: EXPLICIT_RANGE, now: NOW,
      })).rejects.toThrow("injected ingestion failure");
      expect(ledger.db.prepare("SELECT COUNT(*) AS n FROM accounts").get()).toEqual({ n: 1 });
      expect(ledger.db.prepare("SELECT COUNT(*) AS n FROM observations").get()).toEqual({ n: 1 });
      expect(ledger.db.prepare("SELECT COUNT(*) AS n FROM message_records").get()).toEqual({ n: 3 });
      expect(ledger.db.prepare("SELECT COUNT(*) AS n FROM attempts").get()).toEqual({ n: 2 });
      expect(ledger.db.prepare("SELECT COUNT(*) AS n FROM history_state").get()).toEqual({ n: 1 });
      expect(ledger.db.prepare("SELECT COUNT(*) AS n FROM collector_runs").get()).toEqual({ n: 1 });
      expect(
        (ledger.db.prepare("SELECT result FROM collector_runs").get() as { result: string }).result,
      ).toBe("failed");
      ledger.db.exec("DROP TRIGGER fail_second_conversation");
      const replay = await collectIntoLedger(adapter, ledger, account, {
        mode: "backfill", range: EXPLICIT_RANGE, now: NOW,
      });
      expect(replay.ledger).toMatchObject({ attemptsInserted: 1, committed: true });
      expect(new SqliteCheckpointStore(ledger, LEDGER_SCOPE).loadDiscovery("active")?.status)
        .toBe("complete");
    } finally {
      ledger.close();
      await adapter.close();
    }
  });

  it("does not persist a failed page's mutated continuation", async () => {
    const directory = mkdtempSync(join(tmpdir(), "usage-capture-page-atomicity-"));
    temporaryDirectories.push(directory);
    const ledger = new Ledger(join(directory, "usage.sqlite"));
    const account = defaultConfig().accounts[0]!;
    Object.assign(account, {
      id: LEDGER_SCOPE.collectorAccountId,
      expectedProviderUserId: LEDGER_SCOPE.providerUserId,
      expectedWorkspaceId: LEDGER_SCOPE.workspaceId,
      quotaOwnerId: LEDGER_SCOPE.quotaOwnerId,
    });
    const reader = new ScriptedReader();
    const candidate = summary("conv-page-atomicity");
    addCompleteIndex(reader, [candidate], []);
    reader.details.set(candidate.conversationId, detail(candidate.conversationId));
    reader.messagePages.set(
      candidate.conversationId,
      new Map([
        [
          "latest",
          continuationPage([message(candidate.conversationId, "msg-page-1")], "cursor-1"),
        ],
        [
          "cursor-1",
          completePage([message(candidate.conversationId, "msg-page-2")]),
        ],
      ]),
    );

    try {
      const first = await collectIntoLedger(reader, ledger, account, {
        mode: "backfill",
        range: EXPLICIT_RANGE,
        now: NOW,
        maxMessagePagesPerConversation: 1,
      });
      expect(first.revisits[0]?.continuation).toBe("cursor-1");
      const persistedBeforeFailure = (
        ledger.db
          .prepare("SELECT state_json FROM history_state")
          .get() as { state_json: string }
      ).state_json;

      reader.messageErrors.set(
        `${candidate.conversationId}:cursor-1`,
        new Error("late_page_error"),
      );
      ledger.db.exec(`
        CREATE TRIGGER fail_history_page BEFORE INSERT ON observations
        WHEN NEW.source_kind='history_conversation'
        BEGIN SELECT RAISE(ABORT, 'injected failed page'); END
      `);

      await expect(
        collectIntoLedger(reader, ledger, account, {
          mode: "incremental",
          now: new Date("2026-09-07T13:00:00.000Z"),
        }),
      ).rejects.toThrow("injected failed page");
      expect(
        (
          ledger.db
            .prepare("SELECT state_json FROM history_state")
            .get() as { state_json: string }
        ).state_json,
      ).toBe(persistedBeforeFailure);
    } finally {
      ledger.close();
    }
  });
});

function testCheckpoint(
  scope: HistoryScope,
  overrides: Partial<DiscoveryCheckpoint> = {},
): DiscoveryCheckpoint {
  return {
    stateVersion: 1,
    accountId: "fixture-primary",
    scope,
    status: "complete",
    mode: "incremental",
    range: EXPLICIT_RANGE,
    candidateCutoff: EXPLICIT_RANGE.start,
    scanStartedAt: NOW.toISOString(),
    continuation: null,
    pagesFetched: 1,
    pageBudget: 500,
    lastCompleteDiscoveryStartedAt: null,
    lastPageAt: NOW.toISOString(),
    paginationState: "complete",
    warnings: [],
    updatedAt: NOW.toISOString(),
    ...overrides,
  };
}

class RecordingFixtureTransport implements HistoryTransport {
  constructor(
    private readonly inner: FixtureTransport,
    private readonly requests: Array<{ method: string; path: string }>,
  ) {}

  async request(
    method: string,
    path: string,
    params: Record<string, unknown> = {},
  ): Promise<Record<string, unknown>> {
    this.requests.push({ method, path });
    return this.inner.request(method, path, params);
  }
}
