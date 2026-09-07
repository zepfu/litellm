import { describe, expect, it } from "vitest";

import {
  ChatGPTHistoryAdapter,
  AdapterError,
  CapabilityError,
  HttpStatusError,
  LegacyFallbackNotApprovedError,
  RateLimitedError,
  adaptConversationIndex,
  adaptMessagePage,
  assertAllowedRequest,
  isAllowedPath,
  INIT_ROUTE,
  LEGACY_DETAIL,
  MODERN_DETAIL,
  MODERN_INDEX,
  MODERN_MESSAGES,
  SESSION_ROUTE,
  type HistoryTransport,
} from "../../src/adapters/chatgpt/adapter.js";
import { FixtureTransport } from "../../src/adapters/chatgpt/fixture-transport.js";
import { PlaywrightTransport } from "../../src/browser/session.js";

const FIXTURE_ROOT = new URL("../fixtures/v1/", import.meta.url).pathname;

describe("route allowlist", () => {
  it("accepts allowlisted GET history/session routes", () => {
    expect(isAllowedPath(MODERN_INDEX)).toBe(true);
    expect(isAllowedPath(SESSION_ROUTE)).toBe(true);
    expect(isAllowedPath(MODERN_DETAIL.replace("{conversation_id}", "conv-001"))).toBe(true);
    expect(
      isAllowedPath(MODERN_MESSAGES.replace("{conversation_id}", "conv-001")),
    ).toBe(true);
    expect(isAllowedPath(LEGACY_DETAIL.replace("{conversation_id}", "conv-001"))).toBe(true);
  });

  it("rejects non-allowlisted paths", () => {
    expect(isAllowedPath("/backend-api/conversations/conv-001/delete")).toBe(false);
    expect(isAllowedPath("/backend-api/conversations/conv-001?include_deleted=true")).toBe(false);
    expect(isAllowedPath(INIT_ROUTE)).toBe(false);
    expect(isAllowedPath("/backend-api/me")).toBe(false);
    expect(isAllowedPath("/api/auth/session/extra")).toBe(false);
  });

  it("enforces the shared method and path boundary", () => {
    expect(() => assertAllowedRequest("POST", MODERN_INDEX)).toThrow(
      "method not allowlisted",
    );
    expect(() => assertAllowedRequest("GET", "/backend-api/me")).toThrow(
      "path not allowlisted",
    );
  });

  it("rejects invalid production Playwright requests before browser access", async () => {
    const transport = new PlaywrightTransport({
      adapter: "playwright_persistent_context",
      profilePath: "/stage1-missing-profile",
      headless: true,
      allowInteractiveLogin: false,
      requestTimeoutSeconds: 1,
    });

    await expect(transport.request("POST", MODERN_INDEX)).rejects.toThrow(
      "method not allowlisted",
    );
    await expect(transport.request("GET", "/backend-api/me")).rejects.toThrow(
      "path not allowlisted",
    );
    expect(transport.requests).toHaveLength(0);
  });

  it("rejects POST and other mutating methods", async () => {
    const transport = new FixtureTransport(FIXTURE_ROOT);
    const adapter = new ChatGPTHistoryAdapter(transport);
    await expect(
      adapter.listConversations({ archived: false }),
    ).resolves.toBeDefined();
    await expect(
      transport.request("POST", MODERN_INDEX, {}),
    ).rejects.toThrow(AdapterError);
    await expect(
      transport.request("DELETE", MODERN_INDEX, {}),
    ).rejects.toThrow(AdapterError);
  });

  it("rejects paths outside the allowlist", async () => {
    const transport = new FixtureTransport(FIXTURE_ROOT);
    await expect(
      transport.request("GET", "/backend-api/me", {}),
    ).rejects.toThrow(AdapterError);
  });

  it("never issues a request when the method or path is not allowlisted", async () => {
    const transport = new FixtureTransport(FIXTURE_ROOT);
    const adapter = new ChatGPTHistoryAdapter(transport);
    await adapter.listConversations({ archived: false });
    expect(transport.requests).toHaveLength(1);
    await expect(transport.request("POST", MODERN_INDEX, {})).rejects.toThrow();
    expect(transport.requests).toHaveLength(1);
  });

  it("reads fixture detail and message pages through safe conversation paths", async () => {
    const transport = new FixtureTransport(FIXTURE_ROOT);
    const adapter = new ChatGPTHistoryAdapter(transport);

    const conversation = await adapter.fetchConversation("conv-001");
    expect(conversation.coverage).toBe("validated_page");
    expect(conversation.messages).toHaveLength(3);
    expect(JSON.stringify(conversation)).not.toContain("PRIVATE");

    const messages = await adapter.fetchMessages("conv-001", {
      conversationSurface: "chat",
    });
    expect(messages.coverage).toBe("validated_page");
    expect(messages.items).toHaveLength(3);
    expect(messages.items[1]?.recordedFinalModelRaw).toBe("gpt-5.6-astra-pro");

    await expect(adapter.fetchConversation("conv/001")).rejects.toThrow(
      "conversation id is not a safe path token",
    );
  });

  it("does not treat a short page as complete when total says more remain", () => {
    const page = adaptConversationIndex(
      {
        items: [{ id: "conv-001", update_time: "2026-09-07T00:00:00Z" }],
        total: 3,
      },
      { archived: false, offset: 0, limit: 2 },
    );

    expect(page.exhausted).toBe(false);
    expect(page.continuation).toBe(1);
    expect(page.paginationState).toBe("contradictory");
    expect(page.coverage).toBe("partial");
    expect(page.warnings).toContain("short_page_before_reported_total");
  });

  it("does not treat a short page without pagination controls as exhausted", () => {
    const page = adaptConversationIndex(
      {
        items: [{ id: "conv-001", update_time: "2026-09-07T00:00:00Z" }],
      },
      { archived: false, offset: 0, limit: 2 },
    );

    expect(page.exhausted).toBe(false);
    expect(page.continuation).toBe(1);
    expect(page.paginationState).toBe("unknown");
    expect(page.coverage).toBe("partial");
    expect(page.warnings).toContain("missing_pagination_controls");
  });

  it("does not validate an exact-limit page without pagination controls", () => {
    const page = adaptConversationIndex(
      {
        items: [
          { id: "conv-001", update_time: "2026-09-07T00:00:00Z" },
          { id: "conv-002", update_time: "2026-09-07T00:01:00Z" },
        ],
      },
      { archived: false, offset: 0, limit: 2 },
    );

    expect(page.exhausted).toBe(false);
    expect(page.paginationState).toBe("unknown");
    expect(page.coverage).toBe("partial");
    expect(page.warnings).toContain("missing_pagination_controls");
  });

  it("rejects conflicting top-level and nested has-more controls", () => {
    const page = adaptConversationIndex(
      {
        items: [{ id: "conv-001", update_time: "2026-09-07T00:00:00Z" }],
        has_more: false,
        pagination: { has_more: true },
      },
      { archived: false, offset: 0, limit: 100 },
    );

    expect(page.exhausted).toBe(false);
    expect(page.continuation).toBeNull();
    expect(page.paginationState).toBe("contradictory");
    expect(page.coverage).toBe("partial");
    expect(page.warnings).toContain("conflicting_has_more");
  });

  it("keeps missing and invalid update timestamps unresolved", () => {
    const page = adaptConversationIndex(
      {
        items: [
          {
            id: "conv-missing-update",
            create_time: "2026-09-07T00:00:00Z",
          },
          {
            id: "conv-invalid-update",
            create_time: "2026-09-07T00:00:00Z",
            update_time: "not-a-timestamp",
          },
          {
            id: "conv-numeric-update",
            update_time: 1788739200,
          },
        ],
      },
      { archived: false, offset: 0, limit: 100 },
    );

    expect(page.items[0]?.updatedAt).toBeNull();
    expect(page.items[1]?.updatedAt).toBeNull();
    expect(page.items[2]?.updatedAt).toBe("2026-09-07T00:00:00.000Z");
    expect(page.items[0]?.coverage).toBe("partial");
    expect(page.items[1]?.coverage).toBe("partial");
    expect(page.coverage).toBe("partial");
    expect(page.warnings).toContain("conversation_missing_update_time");
    expect(page.warnings).toContain("conversation_invalid_update_time");
  });

  it("normalizes numeric message timestamps and marks invalid timestamps partial", () => {
    const page = adaptMessagePage(
      {
        messages: [
          {
            id: "msg-epoch",
            author: { role: "user" },
            create_time: 1788739200,
          },
          {
            id: "msg-string-epoch",
            author: { role: "assistant" },
            create_time: "1788739201000",
          },
          {
            id: "msg-invalid-time",
            author: { role: "assistant" },
            create_time: "not-a-timestamp",
          },
        ],
        page_info: { has_previous_page: false, start_cursor: null },
      },
      { conversationId: "conv-001" },
    );

    expect(page.items.map((item) => item.createdAt)).toEqual([
      "2026-09-07T00:00:00.000Z",
      "2026-09-07T00:00:01.000Z",
      null,
    ]);
    expect(page.coverage).toBe("partial");
    expect(page.warnings).toContain("message_invalid_created_at");
  });

  it("honors returned index pagination controls", () => {
    const page = adaptConversationIndex(
      {
        items: [{ id: "conv-001", update_time: "2026-09-07T00:00:00Z" }],
        offset: 0,
        total: 101,
        has_more: true,
        next_offset: 25,
      },
      { archived: false, offset: 0, limit: 100 },
    );

    expect(page.continuation).toBe(25);
    expect(page.exhausted).toBe(false);
    expect(page.paginationState).toBe("continuation");
    expect(page.coverage).toBe("validated_page");
  });

  it("marks a returned repeated index offset as contradictory", () => {
    const page = adaptConversationIndex(
      {
        items: [{ id: "conv-001", update_time: "2026-09-07T00:00:00Z" }],
        offset: 0,
        has_more: true,
        next_offset: 0,
      },
      { archived: false, offset: 0, limit: 100 },
    );

    expect(page.continuation).toBe(0);
    expect(page.exhausted).toBe(false);
    expect(page.paginationState).toBe("contradictory");
    expect(page.coverage).toBe("partial");
    expect(page.warnings).toContain("nonadvancing_next_offset");
  });

  it("requires capability approval before using legacy detail after 404", async () => {
    class StatusTransport implements HistoryTransport {
      readonly requests: string[] = [];

      async request(
        _method: string,
        path: string,
      ): Promise<Record<string, unknown>> {
        this.requests.push(path);
        if (path.endsWith("/conversations/conv-001")) {
          return { http_status: 404 };
        }
        return {
          http_status: 200,
          mapping: {
            "node-1": {
              id: "node-1",
              message: {
                id: "msg-1",
                author: { role: "assistant" },
                metadata: { model_slug: "gpt-5.6-astra-pro" },
              },
            },
          },
        };
      }
    }

    const unapprovedTransport = new StatusTransport();
    const unapproved = new ChatGPTHistoryAdapter(
      unapprovedTransport,
      {},
      { legacyFallbackApproved: false },
    );
    const unapprovedFailure = unapproved.fetchConversation("conv-001");
    await expect(unapprovedFailure).rejects.toBeInstanceOf(
      LegacyFallbackNotApprovedError,
    );
    await expect(unapprovedFailure).rejects.toBeInstanceOf(
      CapabilityError,
    );
    expect(unapprovedTransport.requests).toEqual([
      "/backend-api/conversations/conv-001",
    ]);

    const approvedTransport = new StatusTransport();
    const approved = new ChatGPTHistoryAdapter(
      approvedTransport,
      {},
      { legacyFallbackApproved: true },
    );
    const result = await approved.fetchConversation("conv-001");
    expect(result.detailRoute).toBe("legacy");
    expect(approvedTransport.requests).toEqual([
      "/backend-api/conversations/conv-001",
      "/backend-api/conversation/conv-001",
    ]);
  });

  it("surfaces typed HTTP and response-capability failures", async () => {
    class StatusTransport implements HistoryTransport {
      constructor(private readonly payload: Record<string, unknown>) {}

      async request(): Promise<Record<string, unknown>> {
        return this.payload;
      }
    }

    const httpAdapter = new ChatGPTHistoryAdapter(
      new StatusTransport({ http_status: 503 }),
    );
    await expect(httpAdapter.listConversations({ archived: false })).rejects.toMatchObject({
      name: "HttpStatusError",
      status: 503,
      path: MODERN_INDEX,
    } satisfies Partial<HttpStatusError>);

    const capabilityAdapter = new ChatGPTHistoryAdapter(
      new StatusTransport({ http_status: "not-a-status" }),
    );
    await expect(
      capabilityAdapter.listConversations({ archived: false }),
    ).rejects.toMatchObject({
      name: "CapabilityError",
      capability: "http_status",
      path: MODERN_INDEX,
    } satisfies Partial<CapabilityError>);
  });

  it("never falls back to legacy detail for a rate limit", async () => {
    class RateLimitTransport implements HistoryTransport {
      readonly requests: string[] = [];

      async request(
        _method: string,
        path: string,
      ): Promise<Record<string, unknown>> {
        this.requests.push(path);
        return {
          http_status: 429,
          retry_after: "120",
        };
      }
    }

    const transport = new RateLimitTransport();
    const adapter = new ChatGPTHistoryAdapter(
      transport,
      {},
      { legacyFallbackApproved: true },
    );
    await expect(adapter.fetchConversation("conv-001")).rejects.toBeInstanceOf(
      RateLimitedError,
    );
    expect(transport.requests).toEqual([
      "/backend-api/conversations/conv-001",
    ]);
  });

  it("marks missing-conversation uncertainty as explicit unknown state", () => {
    const page = adaptConversationIndex(
      {
        items: [],
        total: 0,
        has_missing_conversations: true,
      },
      { archived: false, offset: 0, limit: 100 },
    );

    expect(page.exhausted).toBe(false);
    expect(page.paginationState).toBe("unknown");
    expect(page.coverage).toBe("unrecognized");
  });
});
