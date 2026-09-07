import { describe, expect, it } from "vitest";

import {
  ChatGPTHistoryAdapter,
  AdapterError,
  isAllowedPath,
  INIT_ROUTE,
  LEGACY_DETAIL,
  MODERN_DETAIL,
  MODERN_INDEX,
  MODERN_MESSAGES,
  SESSION_ROUTE,
} from "../../src/adapters/chatgpt/adapter.js";
import { FixtureTransport } from "../../src/adapters/chatgpt/fixture-transport.js";

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
});
