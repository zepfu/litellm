/**
 * Metadata-only sanitization boundary for the ChatGPT Chat usage collector.
 *
 * The privacy promise is persistence and reporting of allowlisted metadata,
 * not a claim that upstream responses never contain content. Credentials,
 * cookies, tokens, raw headers, browser storage, titles, and message bodies
 * are stripped before any ledger, log, fixture-export, or report write.
 */

import { createHash } from "node:crypto";

export const ADAPTER_VERSION = "chatgpt-chat-history-v1";
export const SURFACE_CHAT = "chat";

const SENSITIVE_KEY_RE =
  /(authorization|auth[_-]?token|bearer|cookie|csrf|credential|email|password|refresh[_-]?token|secret|session[_-]?token|set-cookie|access[_-]?token|id[_-]?token|api[_-]?key|x-auth|storage)/i;
const CONTENT_KEY_RE =
  /^(content|text|title|parts|body|prompt|answer|message|html|markdown|attachment|file_name|filename|tool_result|arguments|input_text|output_text)$/i;
const EMAIL_RE = /[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/i;
const REDACTED = "[redacted]";

const OBSERVATION_ALLOWLIST = new Set([
  "id",
  "conversation_id",
  "message_id",
  "node_id",
  "parent",
  "parent_id",
  "children",
  "author",
  "role",
  "channel",
  "create_time",
  "update_time",
  "created_at",
  "updated_at",
  "status",
  "end_turn",
  "weight",
  "recipient",
  "metadata",
  "model_slug",
  "requested_model",
  "requested_mode",
  "reasoning_effort",
  "default_model_slug",
  "generation_id",
  "request_id",
  "message_request_id",
  "parent_id_of_prompt",
  "is_archived",
  "is_starred",
  "gizmo_id",
  "workspace_id",
  "current_node",
  "mapping",
  "messages",
  "page_info",
  "has_previous_page",
  "start_cursor",
  "offset",
  "limit",
  "total",
  "items",
  "has_versions",
  "conversation_template_id",
  "surface",
  "origin",
  "shared",
  "imported",
  "copied",
  "error_type",
  "error_code",
]);

const METADATA_ALLOWLIST = new Set([
  "model_slug",
  "requested_model",
  "requested_model_slug",
  "requested_mode",
  "reasoning_effort",
  "default_model_slug",
  "generation_id",
  "request_id",
  "message_request_id",
  "parent_id",
  "is_complete",
  "is_visually_hidden_from_conversation",
  "timestamp_",
  "status",
  "gizmo_id",
  "conversation_id",
  "surface",
  "origin",
  "from_shared",
  "from_copy",
  "imported",
  "workspace_id",
]);

const IDENTITY_ALLOWLIST = new Set([
  "provider_user_id",
  "workspace_id",
  "quota_owner_id",
  "collector_account_id",
  "surface",
  "auth_state",
  "plan_label",
  "identity_errors",
]);

const METADATA_IDENTIFIER_KEYS = new Set([
  "model_slug",
  "requested_model",
  "requested_model_slug",
  "default_model_slug",
  "generation_id",
  "request_id",
  "message_request_id",
  "parent_id",
  "gizmo_id",
  "conversation_id",
  "workspace_id",
]);

const METADATA_TOKEN_KEYS = new Set([
  "requested_mode",
  "reasoning_effort",
  "status",
  "surface",
  "origin",
]);

const METADATA_BOOLEAN_KEYS = new Set([
  "is_complete",
  "is_visually_hidden_from_conversation",
  "from_shared",
  "from_copy",
  "imported",
]);

const METADATA_NUMBER_KEYS = new Set(["timestamp_"]);

const SAFE_METADATA_TOKEN_RE = /^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$/;

const UNKNOWN_TYPE_NAMES: Record<string, string> = {
  Object: "object",
  Array: "array",
  String: "string",
  Number: "number",
  Boolean: "boolean",
};

class PrivacyError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "PrivacyError";
  }
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  if (value === null || typeof value !== "object") {
    return false;
  }
  return !Array.isArray(value);
}

export function classifySurface(
  raw: unknown,
  options: { default?: string | null } = {},
): string {
  const candidates: unknown[] = [];
  if (isPlainObject(raw)) {
    const metadata = raw.metadata;
    candidates.push(
      raw.surface,
      raw.product,
      isPlainObject(metadata) ? metadata.surface : undefined,
      raw.conversation_template_id,
    );
  } else {
    candidates.push(raw);
  }
  for (const candidate of candidates) {
    if (candidate === null || candidate === undefined) {
      continue;
    }
    const normalized = String(candidate).trim().toLowerCase();
    if (!normalized) {
      continue;
    }
    if (["chat", "chatgpt", "chatgpt-chat", "ordinary_chat"].includes(normalized)) {
      return SURFACE_CHAT;
    }
    if (normalized.includes("codex")) {
      return "codex";
    }
    if (["work", "chatgpt-work"].includes(normalized)) {
      return "work";
    }
    if (normalized.includes("deep_research") || normalized === "deep-research") {
      return "deep_research";
    }
    if (normalized.includes("agent")) {
      return "agent_mode";
    }
    if (normalized.includes("voice")) {
      return "voice";
    }
    if (normalized.includes("image")) {
      return "image_generation";
    }
    if (normalized !== "unknown" && normalized !== "none") {
      return "unknown";
    }
  }
  if (options.default === SURFACE_CHAT) {
    return SURFACE_CHAT;
  }
  return "unknown";
}

export function schemaFingerprint(value: unknown): string {
  const names = [...iterUnknownFields(value)].sort();
  return sha256Hex(JSON.stringify(names)).slice(0, 32);
}

export function* iterUnknownFields(value: unknown, prefix = ""): Generator<string> {
  if (isPlainObject(value)) {
    for (const [key, child] of Object.entries(value)) {
      const path = prefix ? `${prefix}.${key}` : key;
      if (!OBSERVATION_ALLOWLIST.has(key) && !METADATA_ALLOWLIST.has(key)) {
        yield `${path}:${typeName(child)}`;
      }
      yield* iterUnknownFields(child, path);
    }
  } else if (Array.isArray(value)) {
    const limit = Math.min(value.length, 8);
    for (let index = 0; index < limit; index += 1) {
      yield* iterUnknownFields(value[index], prefix ? `${prefix}[]` : "[]");
    }
  }
}

function typeName(value: unknown): string {
  if (value === null) {
    return "null";
  }
  return UNKNOWN_TYPE_NAMES[Object.prototype.toString.call(value).slice(8, -1)] ??
    typeof value;
}

export function redactText(value: string): string {
  return value.replace(EMAIL_RE, REDACTED);
}

export function sanitizeValue(
  value: unknown,
  options: { key?: string; allowContent?: boolean } = {},
): unknown {
  const key = options.key ?? "";
  const allowContent = options.allowContent ?? false;
  if (isPlainObject(value)) {
    return sanitizeMapping(value, { allowContent });
  }
  if (Array.isArray(value)) {
    return value.map((item) => sanitizeValue(item, { key, allowContent }));
  }
  if (typeof value === "string") {
    if (!allowContent && CONTENT_KEY_RE.test(key)) {
      return null;
    }
    if (SENSITIVE_KEY_RE.test(key)) {
      return REDACTED;
    }
    return redactText(value);
  }
  return value;
}

export function sanitizeMapping(
  payload: Record<string, unknown>,
  options: { allowContent?: boolean } = {},
): Record<string, unknown> {
  const allowContent = options.allowContent ?? false;
  const out: Record<string, unknown> = {};
  for (const [rawKey, rawValue] of Object.entries(payload)) {
    const key = String(rawKey);
    if (SENSITIVE_KEY_RE.test(key)) {
      continue;
    }
    if (!allowContent && CONTENT_KEY_RE.test(key)) {
      continue;
    }
    if (key === "metadata" && isPlainObject(rawValue)) {
      out[key] = sanitizeMetadata(rawValue);
      continue;
    }
    if (key === "author" && isPlainObject(rawValue)) {
      out[key] = {
        role: rawValue.role ?? null,
        name: "[redacted]",
      };
      continue;
    }
    out[key] = sanitizeValue(rawValue, { key, allowContent });
  }
  return out;
}

export function sanitizeMetadata(
  metadata: Record<string, unknown>,
): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [rawKey, rawValue] of Object.entries(metadata)) {
    const key = String(rawKey);
    if (!METADATA_ALLOWLIST.has(key)) {
      continue;
    }
    const projected = projectMetadataValue(key, rawValue);
    if (projected !== undefined) {
      out[key] = projected;
    }
  }
  return out;
}

function projectMetadataValue(key: string, value: unknown): unknown {
  if (METADATA_BOOLEAN_KEYS.has(key)) {
    return typeof value === "boolean" ? value : undefined;
  }
  if (METADATA_NUMBER_KEYS.has(key)) {
    if (typeof value === "boolean" || typeof value !== "number" || !Number.isFinite(value)) {
      return undefined;
    }
    return value;
  }
  if (!METADATA_IDENTIFIER_KEYS.has(key) && !METADATA_TOKEN_KEYS.has(key)) {
    return undefined;
  }
  if (typeof value !== "string") {
    return undefined;
  }
  return sanitizeToken(value) ?? undefined;
}

export function sanitizeToken(value: unknown): string | null {
  if (typeof value !== "string") {
    return null;
  }
  const normalized = value.trim();
  if (!normalized || !SAFE_METADATA_TOKEN_RE.test(normalized)) {
    return null;
  }
  return normalized;
}

export function sanitizeIdentity(
  payload: Record<string, unknown>,
): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const key of IDENTITY_ALLOWLIST) {
    if (!(key in payload)) {
      continue;
    }
    const value = payload[key];
    if (SENSITIVE_KEY_RE.test(key)) {
      continue;
    }
    if (key === "identity_errors") {
      if (Array.isArray(value)) {
        out[key] = value
          .map((item) => sanitizeToken(String(item)))
          .filter((item): item is string => item !== null)
          .slice(0, 16);
      }
      continue;
    }
    if (typeof value === "string") {
      const normalized = sanitizeToken(value);
      if (normalized !== null) {
        out[key] = normalized;
      }
    }
  }
  if (!("surface" in out)) {
    out.surface = classifySurface(payload, { default: null });
  }
  return out;
}

export function observationProjection(
  payload: Record<string, unknown>,
  provenance: {
    sourceKind: string;
    runId: string;
    evidenceId: string;
  },
): Record<string, unknown> {
  const sanitized = sanitizeMapping(payload);
  const keep: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(sanitized)) {
    if (OBSERVATION_ALLOWLIST.has(key)) {
      keep[key] = value;
    }
  }
  keep.surface = classifySurface(payload, { default: null });
  keep.provenance = {
    adapter_version: ADAPTER_VERSION,
    source_kind: provenance.sourceKind,
    run_id: provenance.runId,
    evidence_id: provenance.evidenceId,
    schema_fingerprint: schemaFingerprint(payload),
    unknown_fields: [...iterUnknownFields(payload)].sort().slice(0, 32),
  };
  return keep;
}

export function assertNoSecrets(value: unknown, path = "root"): void {
  walkNoSecrets(value, path);
}

function walkNoSecrets(value: unknown, path: string): void {
  if (typeof value === "string") {
    const lowered = value.toLowerCase();
    for (const needle of ["bearer ", "set-cookie", "authorization", "eyj"]) {
      if (lowered.includes(needle)) {
        throw new PrivacyError(
          `secret-like value survived sanitization at ${path}`,
        );
      }
    }
    if (EMAIL_RE.test(value)) {
      throw new PrivacyError(`email survived sanitization at ${path}`);
    }
    return;
  }
  if (Array.isArray(value)) {
    value.forEach((child, index) => walkNoSecrets(child, `${path}[${index}]`));
    return;
  }
  if (isPlainObject(value)) {
    for (const [key, child] of Object.entries(value)) {
      walkNoSecrets(child, `${path}.${key}`);
    }
  }
}

export function evidenceIdentity(
  sourceKind: string,
  sourceId: string,
  revision: string,
): string {
  return sha256Hex(`${sourceKind}|${sourceId}|${revision}`);
}

function sha256Hex(input: string): string {
  return createHash("sha256").update(input, "utf8").digest("hex");
}
