import { describe, expect, it } from "vitest";

import {
  assertNoSecrets,
  classifySurface,
  sanitizeIdentity,
  sanitizeMapping,
  sanitizeMetadata,
  sanitizeToken,
} from "../../src/security/sanitizer.js";

describe("sanitizer boundary", () => {
  it("never persists credentials, tokens, cookies, or raw headers", () => {
    const payload = {
      authorization: "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
      session_token: "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
      cookie: "session=abc123",
      email: "user@example.com",
      metadata: {
        model_slug: "gpt-5.6-astra-pro",
        requested_model: "gpt-5.6-astra-pro",
      },
    };
    const sanitized = sanitizeMapping(payload);
    expect(sanitized.authorization).toBeUndefined();
    expect(sanitized.session_token).toBeUndefined();
    expect(sanitized.cookie).toBeUndefined();
    expect(sanitized.email).toBeUndefined();
    expect(sanitized.metadata).toEqual({
      model_slug: "gpt-5.6-astra-pro",
      requested_model: "gpt-5.6-astra-pro",
    });
  });

  it("drops content keys and never keeps prompt/answer text", () => {
    const payload = {
      content: "What is the weather today?",
      title: "Weather conversation",
      body: "The answer is...",
      prompt: "Tell me a joke",
      message: "Hello",
      metadata: {
        model_slug: "gpt-5.6-astra-pro",
      },
    };
    const sanitized = sanitizeMapping(payload);
    expect(sanitized.content).toBeUndefined();
    expect(sanitized.title).toBeUndefined();
    expect(sanitized.body).toBeUndefined();
    expect(sanitized.prompt).toBeUndefined();
    expect(sanitized.message).toBeUndefined();
    expect(sanitized.metadata).toEqual({ model_slug: "gpt-5.6-astra-pro" });
  });

  it("rejects payloads that contain secret-like values after sanitization", () => {
    expect(() =>
      assertNoSecrets({ token: "bearer abc123" }),
    ).toThrow(/secret-like value/);
    expect(() =>
      assertNoSecrets({ header: "authorization: bearer abc123" }),
    ).toThrow(/secret-like value/);
    expect(() =>
      assertNoSecrets({ email: "user@example.com" }),
    ).toThrow(/email survived/);
  });

  it("accepts sanitized metadata-only payloads", () => {
    expect(() =>
      assertNoSecrets({
        model_slug: "gpt-5.6-astra-pro",
        requested_model: "gpt-5.6-astra-pro",
        surface: "chat",
      }),
    ).not.toThrow();
  });

  it("classifies surface explicitly and never auto-promotes unknown to chat", () => {
    expect(classifySurface({ surface: "chat" })).toBe("chat");
    expect(classifySurface({ surface: "codex" })).toBe("codex");
    expect(classifySurface({ surface: "deep_research" })).toBe("deep_research");
    expect(classifySurface({ surface: "unknown" })).toBe("unknown");
    expect(classifySurface({ surface: "agent_mode" })).toBe("agent_mode");
    expect(classifySurface({ surface: "voice" })).toBe("voice");
    expect(classifySurface({ surface: "image_generation" })).toBe("image_generation");
    expect(classifySurface({ surface: null })).toBe("unknown");
    expect(classifySurface({})).toBe("unknown");
  });

  it("sanitizes metadata to an allowlisted projection", () => {
    const metadata = {
      model_slug: "gpt-5.6-astra-pro",
      requested_model: "gpt-5.6-astra-pro",
      requested_mode: "standard",
      reasoning_effort: "high",
      generation_id: "gen-123",
      request_id: "req-456",
      unknown_field: "should-be-dropped",
      content: "should-be-dropped",
    };
    const sanitized = sanitizeMetadata(metadata);
    expect(sanitized.model_slug).toBe("gpt-5.6-astra-pro");
    expect(sanitized.requested_model).toBe("gpt-5.6-astra-pro");
    expect(sanitized.unknown_field).toBeUndefined();
    expect(sanitized.content).toBeUndefined();
  });

  it("sanitizes identity records to a strict allowlist", () => {
    const identity = sanitizeIdentity({
      provider_user_id: "user-abc123",
      workspace_id: "ws-xyz",
      quota_owner_id: "user-abc123",
      surface: "chat",
      email: "user@example.com",
      authorization: "bearer token",
    });
    expect(identity.provider_user_id).toBe("user-abc123");
    expect(identity.workspace_id).toBe("ws-xyz");
    expect(identity.quota_owner_id).toBe("user-abc123");
    expect(identity.surface).toBe("chat");
    expect(identity.email).toBeUndefined();
    expect(identity.authorization).toBeUndefined();
  });

  it("rejects unsafe tokens", () => {
    expect(sanitizeToken("valid-token_123")).toBe("valid-token_123");
    expect(sanitizeToken("a" + "b".repeat(127))).toBe("a" + "b".repeat(127));
    expect(sanitizeToken("")).toBeNull();
    expect(sanitizeToken("   ")).toBeNull();
    expect(sanitizeToken("a b")).toBeNull();
    expect(sanitizeToken("a@b")).toBeNull();
  });
});
