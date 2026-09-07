/**
 * Fail-closed identity, workspace, and quota-owner verification.
 *
 * Stage 1 never infers an account from an email fragment, page title, first
 * returned account, or a hard-coded fallback ID. Every configured identity
 * field must be explicitly bound and every observed value must match before
 * collection is allowed to proceed.
 */

import { ADAPTER_VERSION, SURFACE_CHAT } from "../contracts/records.js";
import type { CapabilityRecord, IdentityRecord, Surface } from "../contracts/records.js";
import { classifySurface, sanitizeIdentity, sanitizeToken } from "../security/sanitizer.js";

const REQUIRED_IDENTITY_FIELDS = [
  "providerUserId",
  "workspaceId",
  "quotaOwnerId",
] as const;

type RequiredIdentityField = (typeof REQUIRED_IDENTITY_FIELDS)[number];

export interface ExpectedIdentity {
  providerUserId?: string | null;
  workspaceId?: string | null;
  quotaOwnerId?: string | null;
}

export interface SessionPayload {
  user?: Record<string, unknown> | null;
  account?: Record<string, unknown> | null;
  [key: string]: unknown;
}

export function emptyCapabilities(): CapabilityRecord {
  return {
    adapterVersion: ADAPTER_VERSION,
    indexScopes: ["active", "archived"],
    archiveBehavior: "query_param",
    projectCoverage: "unknown",
    modernDetail: "available",
    pagination: "offset_and_cursor",
    legacySupport: "fallback_on_404_405",
    branchVisibility: "include_has_versions",
    modelMetadata: "message_metadata_projection",
    quotaMetadata: "not_collected_stage1",
    warnings: [],
  };
}

export function inspectSession(
  payload: SessionPayload,
  expected: ExpectedIdentity,
): IdentityRecord {
  const user = isRecord(payload.user) ? payload.user : {};
  const account = isRecord(payload.account) ? payload.account : {};

  const observed: Record<string, unknown> = {
    provider_user_id: firstPresent(user.id, payload.user_id, payload.id),
    workspace_id: firstPresent(payload.workspace_id, account.workspace_id),
    quota_owner_id: firstPresent(payload.quota_owner_id, account.quota_owner_id),
    surface: classifySurface(payload, { default: null }),
  };

  const identity = sanitizeIdentity(observed) as Record<string, unknown>;
  const surface = (identity.surface as Surface | undefined) ?? "unknown";

  const authenticated = Boolean(
    (isRecord(payload.user) && Object.keys(payload.user ?? {}).length > 0) ||
      payload.user_id ||
      payload.id,
  );

  if (!authenticated) {
    return {
      providerUserId: null,
      workspaceId: null,
      quotaOwnerId: null,
      surface,
      authState: "auth_required",
      identityErrors: [],
    };
  }

  const errors: string[] = [];
  for (const field of REQUIRED_IDENTITY_FIELDS) {
    const expectedValue = optionalString(expected[field]);
    const actualValue = optionalString(identity[toSnake(field)]);
    if (expectedValue === null) {
      errors.push(`missing_expected_${toSnake(field)}`);
    } else if (actualValue === null) {
      errors.push(`missing_observed_${toSnake(field)}`);
    } else if (actualValue !== expectedValue) {
      errors.push(`${toSnake(field)}_mismatch`);
    }
  }

  let authState: IdentityRecord["authState"];
  if (errors.some((error) => error.startsWith("missing_expected_"))) {
    authState = "unconfigured";
  } else if (errors.length > 0) {
    authState = "identity_mismatch";
  } else {
    authState = "ready";
  }

  return {
    providerUserId: optionalString(identity.provider_user_id),
    workspaceId: optionalString(identity.workspace_id),
    quotaOwnerId: optionalString(identity.quota_owner_id),
    surface,
    authState,
    identityErrors: errors,
  };
}

function toSnake(field: RequiredIdentityField): string {
  switch (field) {
    case "providerUserId":
      return "provider_user_id";
    case "workspaceId":
      return "workspace_id";
    case "quotaOwnerId":
      return "quota_owner_id";
  }
}

function firstPresent(...values: unknown[]): unknown {
  for (const value of values) {
    if (value !== null && value !== undefined && value !== "") {
      return value;
    }
  }
  return undefined;
}

function optionalString(value: unknown): string | null {
  if (value === null || value === undefined || value === "" || value === false) {
    return null;
  }
  if (value === true) {
    return "true";
  }
  return sanitizeToken(String(value));
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

export { SURFACE_CHAT };
