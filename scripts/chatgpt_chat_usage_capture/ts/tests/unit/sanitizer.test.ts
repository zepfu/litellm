import { describe, expect, it } from "vitest";

import {
  assertNoSecrets,
  classifySurface,
  iterUnknownFields,
  observationProjection,
  sanitizeIdentity,
  sanitizeMapping,
  sanitizeMetadata,
  sanitizeDiagnosticKey,
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
      resolved_model: "gpt-5.6-astra-pro",
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
    expect(sanitized.resolved_model).toBe("gpt-5.6-astra-pro");
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

  it("should recursively project typed metadata and drop nested content, tools, and credentials", () => {
    const projected = observationProjection(
      {
        conversation_id: "conversation-1",
        messages: [
          {
            messageId: "message-1",
            role: "assistant",
            metadata: {
              model_slug: "gpt-5.6-astra-pro",
              content: "private prompt",
              nested: {
                authorization: "Bearer secret",
              },
            },
            content: "private answer",
            tool_calls: [
              {
                arguments: {
                  secret: "private tool input",
                },
              },
            ],
            toolCalls: [{ arguments: { privateValue: "private tool input" } }],
            credentials: { accessToken: "eyJsecret" },
          },
        ],
        nested: {
          content: "private nested content",
        },
      },
      {
        sourceKind: "conversation_detail",
        runId: "run-1",
        evidenceId: "evidence-1",
      },
    );

    expect(projected.messages).toEqual([
      {
        messageId: "message-1",
        role: "assistant",
        metadata: {
          model_slug: "gpt-5.6-astra-pro",
        },
      },
    ]);
    expect(JSON.stringify(projected)).not.toContain("private");
    expect(JSON.stringify(projected)).not.toContain("Bearer");
    expect(JSON.stringify(projected)).not.toContain("eyJ");
  });

  it("should stop unknown-field diagnostics beneath excluded subtrees", () => {
    const fields = [
      ...iterUnknownFields({
        authorization: {
          nested_private_field: {
            email: "user@example.com",
          },
        },
        unknown_field: {
          safe_child: true,
        },
      }),
    ];

    expect(fields.some((field) => field.includes("nested_private_field"))).toBe(false);
    expect(fields.some((field) => field.includes("user@example.com"))).toBe(false);
    expect(fields).toContain("[excluded]:object");
    expect(fields).toContain("unknown_field:object");
  });

  it("should hash unsafe diagnostic keys and keep them bounded", () => {
    const privateKey = "private prompt with user@example.com and " + "x".repeat(300);
    const sanitizedKey = sanitizeDiagnosticKey(privateKey, 32);
    const projected = observationProjection(
      {
        [privateKey]: {
          child: true,
        },
      },
      {
        sourceKind: "conversation_detail",
        runId: "run-1",
        evidenceId: "evidence-1",
      },
    );
    const provenance = projected.provenance as {
      unknown_fields: string[];
    };

    expect(sanitizedKey).toMatch(/^key_[a-f0-9]+$/);
    expect(sanitizedKey.length).toBeLessThanOrEqual(32);
    expect(provenance.unknown_fields.join("\n")).not.toContain(privateKey);
    expect(provenance.unknown_fields.every((field) => field.length <= 96)).toBe(true);
  });

  it("should report incomplete projection and omit unfinished data at depth and node budgets", () => {
    const deepPayload: Record<string, unknown> = {
      safe: "root",
    };
    let cursor = deepPayload;
    for (let index = 0; index < 20; index += 1) {
      const child: Record<string, unknown> = {
        safe: `level-${index}`,
      };
      cursor.child = child;
      cursor = child;
    }

    const depthLimited = observationProjection(
      deepPayload,
      {
        sourceKind: "conversation_detail",
        runId: "run-1",
        evidenceId: "evidence-1",
      },
      { maxDepth: 3, maxNodes: 1000 },
    );
    const depthSanitization = (
      depthLimited.provenance as {
        sanitization: { status: string };
      }
    ).sanitization;

    const nodeLimited = observationProjection(
      {
        messages: [
          { messageId: "message-1", role: "assistant" },
          { messageId: "message-2", role: "assistant" },
        ],
      },
      {
        sourceKind: "conversation_detail",
        runId: "run-1",
        evidenceId: "evidence-1",
      },
      { maxDepth: 16, maxNodes: 3 },
    );
    const nodeSanitization = (
      nodeLimited.provenance as {
        sanitization: { status: string };
      }
    ).sanitization;

    expect(depthSanitization.status).toBe("incomplete");
    expect(depthLimited.coverage).toBe("partial");
    expect(nodeSanitization.status).toBe("incomplete");
    expect(nodeLimited.coverage).toBe("partial");
  });

  it("should validate full and partial 100-message pages with generated provenance", () => {
    const payload = {
      conversation_id: "conversation-1",
      surface: "chat",
      messages: Array.from({ length: 100 }, (_, index) => ({
        conversationId: "conversation-1",
        messageId: `message-${index}`,
        nodeId: `node-${index}`,
        parentId: null,
        children: [],
        role: index % 2 === 0 ? "user" : "assistant",
        channel: "final",
        createdAt: "2026-09-07T00:00:00Z",
        status: "finished_successfully",
        endTurn: true,
        requestedModelRaw: "gpt-5.6-astra-pro",
        requestedModeRaw: "standard",
        requestedReasoningEffortRaw: "high",
        recordedFinalModelRaw: "gpt-5.6-astra-pro",
        generationId: `generation-${index}`,
        requestId: `request-${index}`,
        surface: "chat",
        origin: null,
        metadata: {
          model_slug: "gpt-5.6-astra-pro",
          requested_model: "gpt-5.6-astra-pro",
          surface: "chat",
        },
      })),
    };
    const provenance = {
      sourceKind: "conversation_detail",
      runId: "run-1",
      evidenceId: "evidence-1",
    };

    const projected = observationProjection(payload, provenance);
    const partial = observationProjection(payload, provenance, { maxNodes: 1024 });

    expect(projected.messages).toHaveLength(100);
    expect(() => assertNoSecrets(projected)).not.toThrow();
    expect(partial.coverage).toBe("partial");
    expect(() => assertNoSecrets(partial)).not.toThrow();
  });

  it("should fail closed with an explicit error for cyclic input", () => {
    const cyclic: Record<string, unknown> = {
      conversation_id: "conversation-1",
    };
    cyclic.self = cyclic;

    const projected = observationProjection(
      cyclic,
      {
        sourceKind: "conversation_detail",
        runId: "run-1",
        evidenceId: "evidence-1",
      },
    );
    const provenance = projected.provenance as {
      projection_status: string;
      projection_error: string | null;
    };

    expect(provenance.projection_status).toBe("error");
    expect(provenance.projection_error).toBe("cycle_detected");
    expect(projected.coverage).toBe("unrecognized");
    expect(JSON.stringify(projected)).not.toContain("[object Object]");
  });
});
