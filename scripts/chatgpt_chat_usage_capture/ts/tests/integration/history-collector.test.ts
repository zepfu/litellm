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
  readonly details = new Map<string, ConversationDetailProjection>();
  readonly messagePages = new Map<
    string,
    Map<string, AdaptedPage<MessageRecord>>
  >();
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

    expect(result.status).toBe("complete");
    expect(result.conversations).toHaveLength(2);
    expect(
      result.conversations.find(
        (conversation) => conversation.summary.conversationId === "conv-late-visible",
      )?.scopes,
    ).toEqual(["active", "archived"]);
    expect(result.coverage.projects).toBe("validated_for_discovered_projects");
    expect(result.coverage.branches).toBe("version_metadata_observed");
    expect(result.coverage.overall).toBe("complete");
    expect(store.loadDiscovery("active")?.lastCompleteDiscoveryStartedAt).toBe(
      NOW.toISOString(),
    );
    expect(store.loadDiscovery("archived")?.lastCompleteDiscoveryStartedAt).toBe(
      NOW.toISOString(),
    );
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
    const result = await new HistoryCollector(reader, {
      accountId: "fixture-primary",
      store,
      clock: { now: () => NOW },
      maxMessagePagesPerConversation: 1,
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

    const result = await new HistoryCollector(reader, {
      accountId: "fixture-primary",
      store: new MemoryCheckpointStore(),
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
      pagesFetched: 2,
    });
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
        ["cursor-1", continuationPage([message("conv-incomplete", "msg-2")], "cursor-1")],
      ]),
    );

    const store = new MemoryCheckpointStore();
    const collector = new HistoryCollector(reader, {
      accountId: "fixture-primary",
      store,
      clock: { now: () => NOW },
    });
    const first = await collector.collect({
      mode: "backfill",
      range: EXPLICIT_RANGE,
    });
    expect(first.revisits).toHaveLength(1);
    expect(first.revisits[0]?.reason).toBe("repeated_cursor");
    expect(first.conversations[0]?.coverage).toBe("partial");

    reader.messagePages.set(
      "conv-incomplete",
      new Map([["latest", completePage([message("conv-incomplete", "msg-3")])]]),
    );
    const second = await collector.collect({
      mode: "incremental",
      now: new Date("2026-09-07T13:00:00.000Z"),
    });
    expect(second.revisits).toHaveLength(0);
    expect(second.conversations[0]?.coverage).toBe("complete");
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
    expect(explicit.conversations).toHaveLength(1);
    expect(explicit.conversations[0]?.summary.conversationId).toBe(
      "conv-outside-overlap",
    );
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
    };
    first.saveDiscovery(checkpoint);
    first.upsertRevisit(revisit);
    expect(new SqliteCheckpointStore(ledger, LEDGER_SCOPE).loadDiscovery("active"))
      .toBeNull();
    ledger.transaction(() => first.persist());
    ledger.close();

    const reopened = new Ledger(path);
    const second = new SqliteCheckpointStore(reopened, LEDGER_SCOPE);
    expect(second.loadDiscovery("active")).toEqual(checkpoint);
    expect(second.listRevisits()).toEqual([revisit]);
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

  it("rolls back evidence and discovery together when integrated ingestion fails", async () => {
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
      for (const table of ["accounts", "observations", "message_records", "attempts", "history_state", "collector_runs"]) {
        expect(ledger.db.prepare(`SELECT COUNT(*) AS n FROM ${table}`).get()).toEqual({ n: 0 });
      }
      ledger.db.exec("DROP TRIGGER fail_second_conversation");
      const replay = await collectIntoLedger(adapter, ledger, account, {
        mode: "backfill", range: EXPLICIT_RANGE, now: NOW,
      });
      expect(replay.ledger).toMatchObject({ attemptsInserted: 3, committed: true });
      expect(new SqliteCheckpointStore(ledger, LEDGER_SCOPE).loadDiscovery("active")?.status)
        .toBe("complete");
    } finally {
      ledger.close();
      await adapter.close();
    }
  });
});

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
