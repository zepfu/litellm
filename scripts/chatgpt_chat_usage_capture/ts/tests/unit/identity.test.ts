import { describe, expect, it } from "vitest";

import { inspectSession } from "../../src/normalize/identity.js";

describe("fail-closed identity verification", () => {
  const expected = {
    providerUserId: "user-abc123",
    workspaceId: "ws-xyz",
    quotaOwnerId: "user-abc123",
  };

  it("returns ready when all configured identity fields match", () => {
    const result = inspectSession(
      {
        user: { id: "user-abc123" },
        workspace_id: "ws-xyz",
        account: { quota_owner_id: "user-abc123" },
        quota_owner_id: "user-abc123",
        surface: "chat",
      },
      expected,
    );
    expect(result.authState).toBe("ready");
    expect(result.identityErrors).toEqual([]);
    expect(result.providerUserId).toBe("user-abc123");
    expect(result.workspaceId).toBe("ws-xyz");
    expect(result.quotaOwnerId).toBe("user-abc123");
  });

  it("fails closed on identity mismatch", () => {
    const result = inspectSession(
      {
        user: { id: "user-def456" },
        workspace_id: "ws-xyz",
        quota_owner_id: "user-def456",
        surface: "chat",
      },
      expected,
    );
    expect(result.authState).toBe("identity_mismatch");
    expect(result.identityErrors).toContain("provider_user_id_mismatch");
    expect(result.identityErrors).toContain("quota_owner_id_mismatch");
  });

  it("fails closed on missing observed identity fields", () => {
    const result = inspectSession(
      {
        user: { id: "user-abc123" },
        account: {},
        surface: "chat",
      },
      expected,
    );
    expect(result.authState).toBe("identity_mismatch");
    expect(result.identityErrors).toContain("missing_observed_workspace_id");
    expect(result.identityErrors).toContain("missing_observed_quota_owner_id");
  });

  it("fails closed on missing expected identity fields", () => {
    const result = inspectSession(
      {
        user: { id: "user-abc123" },
        workspace_id: "ws-xyz",
        quota_owner_id: "user-abc123",
        surface: "chat",
      },
      {},
    );
    expect(result.authState).toBe("unconfigured");
    expect(result.identityErrors).toContain("missing_expected_provider_user_id");
    expect(result.identityErrors).toContain("missing_expected_workspace_id");
    expect(result.identityErrors).toContain("missing_expected_quota_owner_id");
  });

  it("fails closed on unauthenticated session", () => {
    const result = inspectSession({}, expected);
    expect(result.authState).toBe("auth_required");
    expect(result.identityErrors).toEqual([]);
  });
});
