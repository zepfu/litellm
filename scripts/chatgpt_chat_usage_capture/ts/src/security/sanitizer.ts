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

export const DEFAULT_MAX_SANITIZER_DEPTH = 16;
/**
 * Sized for the configured 100-message history page while remaining bounded.
 */
export const DEFAULT_MAX_SANITIZER_NODES = 4096;
export const DEFAULT_MAX_SANITIZER_DIAGNOSTICS = 32;
export const DEFAULT_MAX_DIAGNOSTIC_KEY_LENGTH = 96;

const MAX_SANITIZER_NODES = 100_000;
const MAX_SANITIZER_DIAGNOSTICS = 256;
const PROVENANCE_VALIDATION_NODE_RESERVE =
  32 + MAX_SANITIZER_DIAGNOSTICS * 2;

export type SanitizationStatus = "complete" | "incomplete" | "error";

export interface SanitizerOptions {
  /**
   * Retained for source compatibility. Content is never admitted across this
   * boundary, including when callers pass true.
   */
  allowContent?: boolean;
  maxDepth?: number;
  maxNodes?: number;
  maxDiagnostics?: number;
  maxDiagnosticKeyLength?: number;
}

export interface SanitizationDiagnostics {
  status: SanitizationStatus;
  error: string | null;
  nodesVisited: number;
  maxDepth: number;
  maxNodes: number;
  unknownFields: string[];
  diagnostics: string[];
  diagnosticsTruncated: boolean;
}

const SENSITIVE_KEY_RE =
  /(authorization|auth[_-]?token|bearer|cookie|csrf|credential|email|password|refresh[_-]?token|secret|session[_-]?token|set-cookie|access[_-]?token|id[_-]?token|api[_-]?key|x-auth|storage|token)/i;
const CONTENT_KEY_RE =
  /^(content|text|title|parts|body|prompt|answer|message|html|markdown|attachment|attachments|file|files|file[_-].*|tool|tools|tool[_-].*|function|functions|function[_-].*|argument|arguments|argument[_-].*|input|inputs|input[_-].*|output|outputs|output[_-].*|headers?|request[_-]?headers|response[_-]?headers)$/i;
const EMAIL_RE = /[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/i;
const EMAIL_GLOBAL_RE = /[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/gi;
const REDACTED = "[redacted]";
const DIAGNOSTIC_KEY_RE = /^[A-Za-z0-9_][A-Za-z0-9_.:/-]{0,63}$/;
const RESERVED_OBJECT_KEYS = new Set(["__proto__", "constructor", "prototype"]);

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
  "resolved_model",
  "resolved_model_slug",
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
  "detail_route",
  "pagination_state",
  "coverage",
  "warnings",
]);

const METADATA_ALLOWLIST = new Set([
  "model_slug",
  "requested_model",
  "requested_model_slug",
  "resolved_model",
  "resolved_model_slug",
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
  "resolved_model",
  "resolved_model_slug",
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

const MESSAGE_ALLOWLIST = new Set([
  ...OBSERVATION_ALLOWLIST,
  ...METADATA_ALLOWLIST,
  "conversationId",
  "messageId",
  "nodeId",
  "parentId",
  "createdAt",
  "updatedAt",
  "endTurn",
  "requestedModelRaw",
  "requestedModeRaw",
  "requestedReasoningEffortRaw",
  "recordedFinalModelRaw",
  "generationId",
  "requestId",
]);

const MAPPING_NODE_ALLOWLIST = new Set([
  ...MESSAGE_ALLOWLIST,
  "message",
]);

const PAGE_INFO_ALLOWLIST = new Set([
  "has_previous_page",
  "has_next_page",
  "start_cursor",
  "end_cursor",
  "next_cursor",
  "previous_cursor",
  "offset",
  "limit",
  "total",
]);

const DIAGNOSTIC_KNOWN_KEYS = new Set([
  ...OBSERVATION_ALLOWLIST,
  ...METADATA_ALLOWLIST,
  ...MESSAGE_ALLOWLIST,
  ...MAPPING_NODE_ALLOWLIST,
  ...PAGE_INFO_ALLOWLIST,
]);

const BOOLEAN_FIELDS = new Set([
  "is_archived",
  "is_starred",
  "shared",
  "imported",
  "copied",
  "has_versions",
  "has_previous_page",
  "has_next_page",
  "end_turn",
  "endTurn",
  "is_complete",
  "is_visually_hidden_from_conversation",
  "from_shared",
  "from_copy",
]);

const NUMBER_FIELDS = new Set(["weight", "offset", "limit", "total", "timestamp_"]);

const TIMESTAMP_FIELDS = new Set([
  "create_time",
  "update_time",
  "created_at",
  "updated_at",
]);

const TOKEN_ARRAY_FIELDS = new Set(["children"]);
const DIAGNOSTIC_ARRAY_FIELDS = new Set(["warnings"]);

const SAFE_METADATA_TOKEN_RE = /^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$/;

const UNKNOWN_TYPE_NAMES: Record<string, string> = {
  Object: "object",
  Array: "array",
  String: "string",
  Number: "number",
  Boolean: "boolean",
  Null: "null",
  Undefined: "undefined",
  Function: "function",
};

class PrivacyError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "PrivacyError";
  }
}

interface TraversalState {
  maxDepth: number;
  maxNodes: number;
  maxDiagnostics: number;
  maxDiagnosticKeyLength: number;
  nodesVisited: number;
  status: SanitizationStatus;
  errors: string[];
  schemaEntries: Set<string>;
  diagnostics: Set<string>;
  diagnosticsTruncated: boolean;
  active: WeakSet<object>;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  if (value === null || typeof value !== "object") {
    return false;
  }
  return !Array.isArray(value);
}

function isObjectLike(value: unknown): value is object {
  return value !== null && typeof value === "object";
}

function normalizeBudget(value: unknown, fallback: number, maximum: number): number {
  if (!Number.isInteger(value) || Number(value) < 0) {
    return fallback;
  }
  return Math.min(Number(value), maximum);
}

function createTraversalState(
  options: SanitizerOptions = {},
  nodeReserve = 0,
): TraversalState {
  const configuredMaxNodes = normalizeBudget(
    options.maxNodes,
    DEFAULT_MAX_SANITIZER_NODES,
    MAX_SANITIZER_NODES,
  );
  return {
    maxDepth: normalizeBudget(options.maxDepth, DEFAULT_MAX_SANITIZER_DEPTH, 256),
    maxNodes: configuredMaxNodes + nodeReserve,
    maxDiagnostics: normalizeBudget(
      options.maxDiagnostics,
      DEFAULT_MAX_SANITIZER_DIAGNOSTICS,
      MAX_SANITIZER_DIAGNOSTICS,
    ),
    maxDiagnosticKeyLength: normalizeBudget(
      options.maxDiagnosticKeyLength,
      DEFAULT_MAX_DIAGNOSTIC_KEY_LENGTH,
      256,
    ),
    nodesVisited: 0,
    status: "complete",
    errors: [],
    schemaEntries: new Set<string>(),
    diagnostics: new Set<string>(),
    diagnosticsTruncated: false,
    active: new WeakSet<object>(),
  };
}

function effectiveDiagnosticKeyLength(state: TraversalState): number {
  return Math.max(16, state.maxDiagnosticKeyLength);
}

function boundedDiagnosticValue(value: string, maxLength: number, prefix: string): string {
  if (value.length <= maxLength) {
    return value;
  }
  const digestLength = Math.max(1, maxLength - prefix.length);
  return `${prefix}${sha256Hex(value).slice(0, digestLength)}`.slice(0, maxLength);
}

export function sanitizeDiagnosticKey(
  value: unknown,
  maxLength = DEFAULT_MAX_DIAGNOSTIC_KEY_LENGTH,
): string {
  let normalized: string;
  try {
    normalized = String(value).trim();
  } catch {
    normalized = "";
  }
  const length = Math.max(16, normalizeBudget(maxLength, DEFAULT_MAX_DIAGNOSTIC_KEY_LENGTH, 256));
  if (
    normalized.length > 0 &&
    normalized.length <= length &&
    DIAGNOSTIC_KEY_RE.test(normalized) &&
    !RESERVED_OBJECT_KEYS.has(normalized) &&
    !isExcludedKey(normalized)
  ) {
    return normalized;
  }
  return boundedDiagnosticValue(`key_${sha256Hex(normalized)}`, length, "key_");
}

function boundedDiagnosticPath(path: string, state: TraversalState): string {
  return boundedDiagnosticValue(
    path,
    effectiveDiagnosticKeyLength(state),
    "path_",
  );
}

function diagnosticPath(
  parent: string,
  key: string,
  state: TraversalState,
): string {
  const safeKey = sanitizeDiagnosticKey(key, effectiveDiagnosticKeyLength(state));
  return boundedDiagnosticPath(parent ? `${parent}.${safeKey}` : safeKey, state);
}

function dynamicDiagnosticPath(parent: string, state: TraversalState): string {
  return boundedDiagnosticPath(`${parent}[]`, state);
}

function addBounded(
  target: Set<string>,
  value: string,
  state: TraversalState,
): void {
  if (target.has(value)) {
    return;
  }
  if (target.size >= state.maxDiagnostics) {
    state.diagnosticsTruncated = true;
    return;
  }
  target.add(
    boundedDiagnosticValue(
      value,
      effectiveDiagnosticKeyLength(state),
      "diag_",
    ),
  );
}

function recordSchemaEntry(
  state: TraversalState,
  path: string,
  value: unknown,
): void {
  addBounded(
    state.schemaEntries,
    `${boundedDiagnosticPath(path, state)}:${typeName(value)}`,
    state,
  );
}

function recordDiagnostic(state: TraversalState, value: string): void {
  addBounded(state.diagnostics, value, state);
}

function recordExcludedSubtree(
  state: TraversalState,
  parent: string,
  value: unknown,
): void {
  const path = boundedDiagnosticPath(
    parent ? `${parent}.[excluded]` : "[excluded]",
    state,
  );
  recordSchemaEntry(state, path, value);
  recordDiagnostic(state, `${path}:excluded`);
}

function markIncomplete(
  state: TraversalState,
  reason: "depth_budget" | "node_budget" | "invalid_type" | "invalid_value",
  path: string,
): void {
  if (state.status !== "error") {
    state.status = "incomplete";
  }
  recordDiagnostic(
    state,
    `[projection]:${reason}:${boundedDiagnosticPath(path || "root", state)}`,
  );
}

function markError(
  state: TraversalState,
  error: "cycle_detected" | "projection_exception" | "invalid_root",
): void {
  state.status = "error";
  if (!state.errors.includes(error)) {
    state.errors.push(error);
  }
  recordDiagnostic(state, `[projection-error]:${error}`);
}

function enterNode(
  state: TraversalState,
  value: unknown,
  depth: number,
  path: string,
): boolean {
  if (depth > state.maxDepth) {
    markIncomplete(state, "depth_budget", path);
    return false;
  }
  if (state.nodesVisited >= state.maxNodes) {
    markIncomplete(state, "node_budget", path);
    return false;
  }
  state.nodesVisited += 1;
  if (isObjectLike(value)) {
    if (state.active.has(value)) {
      markError(state, "cycle_detected");
      return false;
    }
    state.active.add(value);
  }
  return true;
}

function leaveNode(state: TraversalState, value: unknown): void {
  if (isObjectLike(value)) {
    state.active.delete(value);
  }
}

function visitNode<T>(
  state: TraversalState,
  value: unknown,
  depth: number,
  path: string,
  visitor: () => T,
): T | undefined {
  if (!enterNode(state, value, depth, path)) {
    return undefined;
  }
  try {
    return visitor();
  } catch {
    markError(state, "projection_exception");
    return undefined;
  } finally {
    leaveNode(state, value);
  }
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

export function schemaFingerprint(
  value: unknown,
  options: SanitizerOptions = {},
): string {
  const names = [...iterUnknownFields(value, "", options)].sort();
  return sha256Hex(JSON.stringify(names)).slice(0, 32);
}

export function* iterUnknownFields(
  value: unknown,
  prefix = "",
  options: SanitizerOptions = {},
): Generator<string> {
  const state = createTraversalState(options);
  walkUnknownValue(value, state, 0, prefix, false);
  for (const entry of [...state.schemaEntries].sort()) {
    yield entry;
  }
  if (state.status !== "complete") {
    yield `[diagnostic]:${state.status}`;
  }
}

function walkUnknownValue(
  value: unknown,
  state: TraversalState,
  depth: number,
  path: string,
  dynamicKeys: boolean,
): void {
  if (!enterNode(state, value, depth, path)) {
    return;
  }
  try {
    if (Array.isArray(value)) {
      for (const child of value) {
        walkUnknownValue(
          child,
          state,
          depth + 1,
          dynamicDiagnosticPath(path, state),
          false,
        );
      }
      return;
    }
    if (!isPlainObject(value)) {
      return;
    }
    for (const [rawKey, child] of Object.entries(value)) {
      const key = String(rawKey);
      if (isExcludedKey(key)) {
        recordExcludedSubtree(state, path, child);
        continue;
      }
      const childPath = dynamicKeys
        ? dynamicDiagnosticPath(path, state)
        : diagnosticPath(path, key, state);
      if (!DIAGNOSTIC_KNOWN_KEYS.has(key)) {
        recordSchemaEntry(state, childPath, child);
      }
      if (key === "mapping" && isPlainObject(child)) {
        walkUnknownValue(child, state, depth + 1, childPath, true);
      } else {
        walkUnknownValue(child, state, depth + 1, childPath, false);
      }
    }
  } catch {
    markError(state, "projection_exception");
  } finally {
    leaveNode(state, value);
  }
}

function isExcludedKey(key: string): boolean {
  const normalized = key
    .replace(/([a-z])([A-Z])/g, "$1_$2")
    .replace(/[\s-]+/g, "_")
    .toLowerCase();
  return SENSITIVE_KEY_RE.test(normalized) || CONTENT_KEY_RE.test(normalized);
}

function typeName(value: unknown): string {
  if (value === null) {
    return "null";
  }
  return UNKNOWN_TYPE_NAMES[Object.prototype.toString.call(value).slice(8, -1)] ??
    typeof value;
}

export function redactText(value: string): string {
  return value.replace(EMAIL_GLOBAL_RE, REDACTED);
}

export function sanitizeValue(
  value: unknown,
  options: SanitizerOptions & { key?: string } = {},
): unknown {
  const state = createTraversalState(options);
  const key = options.key ?? "";
  if (key === "metadata") {
    return projectMetadataObject(value, state, 0, key);
  }
  if (isExcludedKey(key)) {
    recordExcludedSubtree(state, "", value);
    return undefined;
  }
  if (OBSERVATION_ALLOWLIST.has(key) || MESSAGE_ALLOWLIST.has(key)) {
    return projectField(key, value, state, 0, key);
  }
  if (isPlainObject(value)) {
    return projectObject(value, state, 0, key, OBSERVATION_ALLOWLIST);
  }
  if (Array.isArray(value)) {
    return projectTokenArray(value, state, 0, key);
  }
  if (typeof value === "string") {
    return projectTokenValue(value, state, 0, key);
  }
  if (typeof value === "number" || typeof value === "boolean" || value === null) {
    if (!enterNode(state, value, 0, key)) {
      return undefined;
    }
    leaveNode(state, value);
    return value;
  }
  return undefined;
}

export function sanitizeMapping(
  payload: Record<string, unknown>,
  options: SanitizerOptions = {},
): Record<string, unknown> {
  const state = createTraversalState(options);
  return projectObject(payload, state, 0, "", OBSERVATION_ALLOWLIST) ?? {};
}

export function sanitizeMetadata(
  metadata: Record<string, unknown>,
  options: SanitizerOptions = {},
): Record<string, unknown> {
  const state = createTraversalState(options);
  return projectMetadataObject(metadata, state, 0, "metadata") ?? {};
}

export function sanitizeMetadataWithDiagnostics(
  metadata: Record<string, unknown>,
  options: SanitizerOptions = {},
): {
  metadata: Record<string, unknown>;
  diagnostics: SanitizationDiagnostics;
} {
  const state = createTraversalState(options);
  const projected = projectMetadataObject(metadata, state, 0, "metadata") ?? {};
  return {
    metadata: projected,
    diagnostics: diagnosticsFromState(state),
  };
}

function projectObject(
  value: unknown,
  state: TraversalState,
  depth: number,
  path: string,
  allowlist: Set<string>,
): Record<string, unknown> | undefined {
  if (!isPlainObject(value)) {
    markIncomplete(state, "invalid_type", path);
    return undefined;
  }
  return visitNode(state, value, depth, path, () => {
    const entries = Object.entries(value);
    const out: Record<string, unknown> = {};

    // Project allowlisted evidence before unknown-field diagnostics so
    // arbitrary provider fields cannot exhaust the budget first.
    for (const [rawKey, rawValue] of entries) {
      const key = String(rawKey);
      if (isExcludedKey(key) || !allowlist.has(key)) {
        continue;
      }
      const childPath = diagnosticPath(path, key, state);
      const projected = projectField(key, rawValue, state, depth + 1, childPath);
      if (projected !== undefined) {
        out[key] = projected;
      }
    }

    for (const [rawKey, rawValue] of entries) {
      const key = String(rawKey);
      if (isExcludedKey(key)) {
        recordExcludedSubtree(state, path, rawValue);
        continue;
      }
      if (allowlist.has(key)) {
        continue;
      }
      const childPath = diagnosticPath(path, key, state);
      recordSchemaEntry(state, childPath, rawValue);
      if (key === "mapping" && isPlainObject(rawValue)) {
        walkUnknownValue(rawValue, state, depth + 1, childPath, true);
      } else {
        walkUnknownValue(rawValue, state, depth + 1, childPath, false);
      }
    }

    return out;
  });
}

function projectField(
  key: string,
  value: unknown,
  state: TraversalState,
  depth: number,
  path: string,
): unknown {
  if (isExcludedKey(key)) {
    recordExcludedSubtree(state, path, value);
    return undefined;
  }
  if (key === "metadata") {
    return value === null
      ? null
      : projectMetadataObject(value, state, depth, path);
  }
  if (key === "author") {
    return projectAuthor(value, state, depth, path);
  }
  if (key === "messages") {
    return projectObjectArray(value, state, depth, path, MESSAGE_ALLOWLIST);
  }
  if (key === "items") {
    return projectObjectArray(value, state, depth, path, OBSERVATION_ALLOWLIST);
  }
  if (key === "mapping") {
    return projectMapping(value, state, depth, path);
  }
  if (key === "page_info") {
    return projectObject(value, state, depth, path, PAGE_INFO_ALLOWLIST);
  }
  if (TOKEN_ARRAY_FIELDS.has(key)) {
    return projectTokenArray(value, state, depth, path);
  }
  if (DIAGNOSTIC_ARRAY_FIELDS.has(key)) {
    return projectDiagnosticArray(value, state, depth, path);
  }
  if (BOOLEAN_FIELDS.has(key)) {
    return projectBoolean(value, state, depth, path);
  }
  if (NUMBER_FIELDS.has(key)) {
    return projectNumber(value, state, depth, path);
  }
  if (TIMESTAMP_FIELDS.has(key)) {
    return projectTimestamp(value, state, depth, path);
  }
  return projectTokenValue(value, state, depth, path);
}

function projectMetadataObject(
  value: unknown,
  state: TraversalState,
  depth: number,
  path: string,
): Record<string, unknown> | undefined {
  if (!isPlainObject(value)) {
    markIncomplete(state, "invalid_type", path);
    return undefined;
  }
  return visitNode(state, value, depth, path, () => {
    const entries = Object.entries(value);
    const out: Record<string, unknown> = {};

    // Process safety-critical allowlisted metadata before unknown-field
    // diagnostics so a large unknown subtree cannot consume the traversal
    // budget and silently drop model/generation evidence.
    for (const [rawKey, rawValue] of entries) {
      const key = String(rawKey);
      if (isExcludedKey(key) || !METADATA_ALLOWLIST.has(key)) {
        continue;
      }
      const childPath = diagnosticPath(path, key, state);
      const projected = projectMetadataValue(
        key,
        rawValue,
        state,
        depth + 1,
        childPath,
      );
      if (projected !== undefined) {
        out[key] = projected;
      }
    }

    for (const [rawKey, rawValue] of entries) {
      const key = String(rawKey);
      if (isExcludedKey(key)) {
        recordExcludedSubtree(state, path, rawValue);
        continue;
      }
      if (METADATA_ALLOWLIST.has(key)) {
        continue;
      }
      const childPath = diagnosticPath(path, key, state);
      recordSchemaEntry(state, childPath, rawValue);
      walkUnknownValue(rawValue, state, depth + 1, childPath, false);
    }

    return out;
  });
}

function projectAuthor(
  value: unknown,
  state: TraversalState,
  depth: number,
  path: string,
): Record<string, unknown> | null | undefined {
  if (value === null) {
    return null;
  }
  if (!isPlainObject(value)) {
    markIncomplete(state, "invalid_type", path);
    return undefined;
  }
  return visitNode(state, value, depth, path, () => {
    const role = value.role;
    const projectedRole =
      typeof role === "string"
        ? projectTokenValue(role, state, depth + 1, `${path}.role`)
        : null;
    if (typeof role !== "undefined" && projectedRole === undefined) {
      markIncomplete(state, "invalid_value", `${path}.role`);
    }
    return { role: projectedRole };
  });
}

function projectObjectArray(
  value: unknown,
  state: TraversalState,
  depth: number,
  path: string,
  allowlist: Set<string>,
): unknown[] | null | undefined {
  if (value === null) {
    return null;
  }
  if (!Array.isArray(value)) {
    markIncomplete(state, "invalid_type", path);
    return undefined;
  }
  return visitNode(state, value, depth, path, () => {
    const out: unknown[] = [];
    for (let index = 0; index < value.length; index += 1) {
      const child = value[index];
      const childPath = dynamicDiagnosticPath(path, state);
      if (!isPlainObject(child)) {
        markIncomplete(state, "invalid_type", childPath);
        continue;
      }
      const projected = projectObject(child, state, depth + 1, childPath, allowlist);
      if (projected !== undefined) {
        out.push(projected);
      }
    }
    return out;
  });
}

function projectMapping(
  value: unknown,
  state: TraversalState,
  depth: number,
  path: string,
): Record<string, unknown> | null | undefined {
  if (value === null) {
    return null;
  }
  if (!isPlainObject(value)) {
    markIncomplete(state, "invalid_type", path);
    return undefined;
  }
  return visitNode(state, value, depth, path, () => {
    const out: Record<string, unknown> = {};
    for (const [rawKey, rawValue] of Object.entries(value)) {
      if (isExcludedKey(rawKey)) {
        recordExcludedSubtree(state, path, rawValue);
        continue;
      }
      const safeKey = sanitizeDynamicKey(rawKey, state);
      const childPath = dynamicDiagnosticPath(path, state);
      if (!isPlainObject(rawValue)) {
        markIncomplete(state, "invalid_type", childPath);
        continue;
      }
      const projected = projectObject(
        rawValue,
        state,
        depth + 1,
        childPath,
        MAPPING_NODE_ALLOWLIST,
      );
      if (projected !== undefined) {
        Object.defineProperty(out, safeKey, {
          configurable: true,
          enumerable: true,
          value: projected,
          writable: true,
        });
      }
    }
    return out;
  });
}

function sanitizeDynamicKey(value: string, state: TraversalState): string {
  const token = sanitizeToken(value);
  if (
    token !== null &&
    !RESERVED_OBJECT_KEYS.has(token) &&
    !isExcludedKey(token)
  ) {
    return token;
  }
  return sanitizeDiagnosticKey(value, effectiveDiagnosticKeyLength(state));
}

function projectTokenArray(
  value: unknown,
  state: TraversalState,
  depth: number,
  path: string,
): string[] | null | undefined {
  if (value === null) {
    return null;
  }
  if (!Array.isArray(value)) {
    markIncomplete(state, "invalid_type", path);
    return undefined;
  }
  return visitNode(state, value, depth, path, () => {
    const out: string[] = [];
    for (const child of value) {
      const projected = projectTokenValue(
        child,
        state,
        depth + 1,
        dynamicDiagnosticPath(path, state),
      );
      if (typeof projected === "string") {
        out.push(projected);
      }
    }
    return out;
  });
}

function projectDiagnosticArray(
  value: unknown,
  state: TraversalState,
  depth: number,
  path: string,
): string[] | null | undefined {
  if (value === null) {
    return null;
  }
  if (!Array.isArray(value)) {
    markIncomplete(state, "invalid_type", path);
    return undefined;
  }
  return visitNode(state, value, depth, path, () => {
    const out: string[] = [];
    for (const child of value) {
      const childPath = dynamicDiagnosticPath(path, state);
      if (!enterNode(state, child, depth + 1, childPath)) {
        continue;
      }
      if (typeof child !== "string") {
        markIncomplete(state, "invalid_type", childPath);
        leaveNode(state, child);
        continue;
      }
      const normalized = sanitizeDiagnosticText(child);
      if (normalized !== null) {
        out.push(normalized);
      }
      leaveNode(state, child);
    }
    return out;
  });
}

function sanitizeDiagnosticText(value: string): string | null {
  const normalized = value.trim();
  if (!normalized) {
    return null;
  }
  if (
    normalized.length <= DEFAULT_MAX_DIAGNOSTIC_KEY_LENGTH &&
    DIAGNOSTIC_KEY_RE.test(normalized) &&
    !isExcludedKey(normalized)
  ) {
    return normalized;
  }
  return `diag_${sha256Hex(normalized).slice(0, 16)}`;
}

function projectTokenValue(
  value: unknown,
  state: TraversalState,
  depth: number,
  path: string,
): string | null | undefined {
  if (value === null) {
    if (!enterNode(state, value, depth, path)) {
      return undefined;
    }
    leaveNode(state, value);
    return null;
  }
  if (typeof value !== "string") {
    markIncomplete(state, "invalid_type", path);
    return undefined;
  }
  if (!enterNode(state, value, depth, path)) {
    return undefined;
  }
  const token = sanitizeToken(value);
  leaveNode(state, value);
  if (token === null) {
    markIncomplete(state, "invalid_value", path);
    return undefined;
  }
  return token;
}

function projectBoolean(
  value: unknown,
  state: TraversalState,
  depth: number,
  path: string,
): boolean | null | undefined {
  if (value === null) {
    if (!enterNode(state, value, depth, path)) {
      return undefined;
    }
    leaveNode(state, value);
    return null;
  }
  if (typeof value !== "boolean") {
    markIncomplete(state, "invalid_type", path);
    return undefined;
  }
  if (!enterNode(state, value, depth, path)) {
    return undefined;
  }
  leaveNode(state, value);
  return value;
}

function projectNumber(
  value: unknown,
  state: TraversalState,
  depth: number,
  path: string,
): number | null | undefined {
  if (value === null) {
    if (!enterNode(state, value, depth, path)) {
      return undefined;
    }
    leaveNode(state, value);
    return null;
  }
  if (typeof value !== "number" || !Number.isFinite(value)) {
    markIncomplete(state, "invalid_type", path);
    return undefined;
  }
  if (!enterNode(state, value, depth, path)) {
    return undefined;
  }
  leaveNode(state, value);
  return value;
}

function projectTimestamp(
  value: unknown,
  state: TraversalState,
  depth: number,
  path: string,
): string | number | null | undefined {
  if (value === null) {
    if (!enterNode(state, value, depth, path)) {
      return undefined;
    }
    leaveNode(state, value);
    return null;
  }
  if (typeof value === "number") {
    return projectNumber(value, state, depth, path);
  }
  return projectTokenValue(value, state, depth, path);
}

function projectMetadataValue(
  key: string,
  value: unknown,
  state: TraversalState,
  depth: number,
  path: string,
): unknown {
  if (METADATA_BOOLEAN_KEYS.has(key)) {
    if (typeof value !== "boolean") {
      markIncomplete(state, "invalid_type", path);
      return undefined;
    }
    if (!enterNode(state, value, depth, path)) {
      return undefined;
    }
    leaveNode(state, value);
    return value;
  }
  if (METADATA_NUMBER_KEYS.has(key)) {
    return projectNumber(value, state, depth, path);
  }
  if (!METADATA_IDENTIFIER_KEYS.has(key) && !METADATA_TOKEN_KEYS.has(key)) {
    return undefined;
  }
  return projectTokenValue(value, state, depth, path);
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
  options: SanitizerOptions = {},
): Record<string, unknown> {
  const state = createTraversalState(options);
  const projected = isPlainObject(payload)
    ? projectObject(payload, state, 0, "", OBSERVATION_ALLOWLIST)
    : undefined;
  if (!isPlainObject(payload)) {
    markError(state, "invalid_root");
  }
  const keep: Record<string, unknown> = projected ?? {};
  keep.surface = classifySurface(payload, { default: null });
  if (state.status === "error") {
    keep.coverage = "unrecognized";
  } else if (state.status === "incomplete") {
    keep.coverage = "partial";
  }
  const diagnostics = diagnosticsFromState(state);
  const safeSourceKind = safeProvenanceToken(provenance.sourceKind);
  const safeRunId = safeProvenanceToken(provenance.runId);
  const safeEvidenceId = safeProvenanceToken(provenance.evidenceId);
  keep.provenance = {
    adapter_version: ADAPTER_VERSION,
    source_kind: safeSourceKind,
    run_id: safeRunId,
    evidence_id: safeEvidenceId,
    schema_fingerprint: sha256Hex(JSON.stringify(diagnostics.unknownFields)).slice(0, 32),
    unknown_fields: diagnostics.unknownFields.slice(0, DEFAULT_MAX_SANITIZER_DIAGNOSTICS),
    projection_status: diagnostics.status,
    projection_error: diagnostics.error,
    sanitization: {
      status: diagnostics.status,
      error: diagnostics.error,
      nodes_visited: diagnostics.nodesVisited,
      max_depth: diagnostics.maxDepth,
      max_nodes: diagnostics.maxNodes,
      diagnostic_keys: diagnostics.diagnostics,
      diagnostics_truncated: diagnostics.diagnosticsTruncated,
    },
  };
  return keep;
}

function diagnosticsFromState(state: TraversalState): SanitizationDiagnostics {
  return {
    status: state.status,
    error: state.errors[0] ?? null,
    nodesVisited: state.nodesVisited,
    maxDepth: state.maxDepth,
    maxNodes: state.maxNodes,
    unknownFields: [...state.schemaEntries].sort(),
    diagnostics: [...state.diagnostics].sort(),
    diagnosticsTruncated: state.diagnosticsTruncated,
  };
}

function safeProvenanceToken(value: unknown): string {
  const token = sanitizeToken(value);
  return token ?? `id_${sha256Hex(String(value)).slice(0, 16)}`;
}

export function assertNoSecrets(
  value: unknown,
  path = "root",
  options: SanitizerOptions = {},
): void {
  const state = createTraversalState(options, PROVENANCE_VALIDATION_NODE_RESERVE);
  walkNoSecrets(value, path, state, 0);
  if (state.status !== "complete") {
    throw new PrivacyError(`secret check incomplete (${state.status})`);
  }
}

function walkNoSecrets(
  value: unknown,
  path: string,
  state: TraversalState,
  depth: number,
): void {
  if (!enterNode(state, value, depth, path)) {
    return;
  }
  try {
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
      value.forEach((child, index) =>
        walkNoSecrets(child, `${path}[${index}]`, state, depth + 1),
      );
      return;
    }
    if (isPlainObject(value)) {
      for (const [key, child] of Object.entries(value)) {
        walkNoSecrets(child, `${path}.${key}`, state, depth + 1);
      }
    }
  } finally {
    leaveNode(state, value);
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
