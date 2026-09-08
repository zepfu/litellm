/**
 * Read-only ChatGPT history adapter.
 *
 * All application-issued requests are GET-only and restricted to an explicit
 * allowlist of history and session routes. The adapter never submits prompts,
 * never archives/renames/deletes conversations, and never issues any mutating
 * call. Responses are adapted into sanitized projections before leaving this
 * module.
 */

import { ADAPTER_VERSION } from "../../contracts/records.js";
import type {
  AdaptedPage,
  CapabilityRecord,
  ConversationDetailProjection,
  ConversationSummary,
  MessageRecord,
  PaginationState,
  Surface,
} from "../../contracts/records.js";
import {
  classifySurface,
  sanitizeToken,
  sanitizeMetadataWithDiagnostics,
} from "../../security/sanitizer.js";
import { emptyCapabilities, inspectSession } from "../../normalize/identity.js";
import type {
  ExpectedIdentity,
  SessionPayload,
} from "../../normalize/identity.js";

export const MODERN_INDEX = "/backend-api/conversations";
export const MODERN_DETAIL = "/backend-api/conversations/{conversation_id}";
export const MODERN_MESSAGES = "/backend-api/conversations/{conversation_id}/messages";
export const LEGACY_DETAIL = "/backend-api/conversation/{conversation_id}";
export const SESSION_ROUTE = "/api/auth/session";
export const INIT_ROUTE = "/backend-api/conversation/init";

export const ALLOWED_METHODS = new Set(["GET"]);
export const ALLOWED_PATH_PREFIXES = [
  "/backend-api/conversations",
  "/backend-api/conversation/",
  "/api/auth/session",
] as const;

const RETRY_AFTER_HEADER_KEYS = ["Retry-After", "retry-after"];

export class AdapterError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "AdapterError";
  }
}

export class HttpStatusError extends AdapterError {
  readonly status: number;
  readonly path: string | null;
  readonly retryAfter: string | null;

  constructor(
    message: string,
    options: {
      status: number;
      path?: string | null;
      retryAfter?: string | null;
    },
  ) {
    super(message);
    this.name = "HttpStatusError";
    this.status = options.status;
    this.path = options.path ?? null;
    this.retryAfter = options.retryAfter ?? null;
  }
}

export class CapabilityError extends AdapterError {
  readonly capability: string;
  readonly path: string | null;
  readonly status: number | null;

  constructor(
    message: string,
    options: {
      capability: string;
      path?: string | null;
      status?: number | null;
    },
  ) {
    super(message);
    this.name = "CapabilityError";
    this.capability = options.capability;
    this.path = options.path ?? null;
    this.status = options.status ?? null;
  }
}

export class AuthenticationRequiredError extends HttpStatusError {
  constructor(message: string, options: { status: number; path: string }) {
    super(message, options);
    this.name = "AuthenticationRequiredError";
  }
}

export class RateLimitedError extends HttpStatusError {
  constructor(
    message: string,
    options: { status?: number; retryAfter?: string | null; path?: string | null } = {},
  ) {
    super(message, {
      status: options.status ?? 429,
      ...(options.retryAfter !== undefined
        ? { retryAfter: options.retryAfter }
        : {}),
      ...(options.path !== undefined ? { path: options.path } : {}),
    });
    this.name = "RateLimitedError";
  }
}

export class LegacyFallbackNotApprovedError extends CapabilityError {
  constructor(message: string, options: { status: number; path: string }) {
    super(message, {
      capability: "legacy_detail_fallback",
      status: options.status,
      path: options.path,
    });
    this.name = "LegacyFallbackNotApprovedError";
  }
}

interface IndexPaginationControls {
  nextOffset: number | null;
  nextOffsetPresent: boolean;
  hasMore: boolean | null;
  hasMorePresent: boolean;
  returnedOffset: number | null;
  returnedOffsetPresent: boolean;
  invalid: boolean;
  contradictory: boolean;
  warnings: string[];
}

interface TimestampResult {
  value: string | null;
  warning: "conversation_missing_update_time" | "conversation_invalid_update_time" | null;
}

export interface HistoryTransport {
  request(
    method: string,
    path: string,
    params?: Record<string, unknown>,
  ): Promise<Record<string, unknown>>;
}

export function assertAllowedRequest(method: string, path: string): void {
  const normalizedMethod = method.toUpperCase();
  if (!ALLOWED_METHODS.has(normalizedMethod)) {
    throw new AdapterError(`method not allowlisted: ${normalizedMethod} ${path}`);
  }
  if (!isAllowedPath(path)) {
    throw new AdapterError(`path not allowlisted: ${path}`);
  }
}

export function isAllowedPath(path: string): boolean {
  if (path === MODERN_INDEX || path === SESSION_ROUTE || path === INIT_ROUTE) {
    return path !== INIT_ROUTE;
  }
  const segments = path.split("/");
  const conversationId = segments[3];
  if (!conversationId || sanitizeToken(conversationId) === null) {
    return false;
  }
  if (
    segments.length === 4 &&
    segments[1] === "backend-api" &&
    segments[2] === "conversations"
  ) {
    return true;
  }
  if (
    segments.length === 5 &&
    segments[1] === "backend-api" &&
    segments[2] === "conversations" &&
    segments[4] === "messages"
  ) {
    return true;
  }
  return (
    segments.length === 4 &&
    segments[1] === "backend-api" &&
    segments[2] === "conversation"
  );
}

export class ChatGPTHistoryAdapter {
  readonly schemaVersion = ADAPTER_VERSION;
  readonly capabilities: CapabilityRecord = emptyCapabilities();

  constructor(
    private readonly transport: HistoryTransport,
    private readonly expectedIdentity: ExpectedIdentity = {},
    private readonly options: { legacyFallbackApproved?: boolean } = {},
  ) {}

  async inspectSessionIdentity() {
    let payload: Record<string, unknown>;
    try {
      payload = await this.request("GET", SESSION_ROUTE);
      raiseIfHttpError(payload, SESSION_ROUTE);
    } catch (error) {
      if (error instanceof AuthenticationRequiredError) {
        return {
          providerUserId: null,
          workspaceId: null,
          quotaOwnerId: null,
          surface: "unknown" as Surface,
          authState: "auth_required" as const,
          identityErrors: [],
        };
      }
      throw error;
    }
    return inspectSession(payload as SessionPayload, this.expectedIdentity);
  }

  async listConversations(options: {
    archived: boolean;
    offset?: number;
    limit?: number;
    order?: string;
  }): Promise<AdaptedPage<ConversationSummary>> {
    const offset = options.offset ?? 0;
    const limit = options.limit ?? 100;
    const order = options.order ?? "updated";
    const payload = await this.request("GET", MODERN_INDEX, {
      offset,
      limit,
      order,
      is_archived: String(options.archived),
    });
    raiseIfHttpError(payload, MODERN_INDEX);
    return adaptConversationIndex(payload, {
      archived: options.archived,
      offset,
      limit,
    });
  }

  async fetchConversation(
    conversationId: string,
    options: { allowLegacyFallback?: boolean } = {},
  ): Promise<ConversationDetailProjection> {
    const modernPath = conversationPath(MODERN_DETAIL, conversationId);
    const payload = await this.request("GET", modernPath, {
      include_has_versions: "true",
      num_turns: 100,
    });
    const status = httpStatus(payload, modernPath);
    if (status === 404 || status === 405) {
      const allowLegacyFallback =
        options.allowLegacyFallback ??
        this.options.legacyFallbackApproved ??
        this.capabilities.legacySupport === "fallback_on_404_405";
      if (!allowLegacyFallback) {
        throw new LegacyFallbackNotApprovedError(
          `modern detail returned ${status} for ${conversationId}; legacy fallback is not capability-approved`,
          { status, path: modernPath },
        );
      }
      const legacyPath = conversationPath(LEGACY_DETAIL, conversationId);
      const legacyPayload = await this.request("GET", legacyPath);
      raiseIfHttpError(legacyPayload, legacyPath);
      return adaptConversationDetail(legacyPayload, conversationId, {
        detailRoute: "legacy",
      });
    }
    raiseIfHttpError(payload, modernPath);
    return adaptConversationDetail(payload, conversationId, {
      detailRoute: "modern",
    });
  }

  async fetchMessages(
    conversationId: string,
    options: { before?: string | null; numTurns?: number; conversationSurface?: Surface } = {},
  ): Promise<AdaptedPage<MessageRecord>> {
    const params: Record<string, unknown> = {
      include_has_versions: "true",
      num_turns: options.numTurns ?? 100,
    };
    if (options.before) {
      params.before = options.before;
    }
    const path = conversationPath(MODERN_MESSAGES, conversationId);
    const payload = await this.request("GET", path, params);
    raiseIfHttpError(payload, path);
    return adaptMessagePage(payload, {
      conversationId,
      conversationSurface: options.conversationSurface ?? "unknown",
    });
  }

  async close(): Promise<void> {
    const closer = (this.transport as { close?: () => Promise<void> | void }).close;
    if (typeof closer === "function") {
      await closer.call(this.transport);
    }
  }

  private async request(
    method: string,
    path: string,
    params: Record<string, unknown> = {},
  ): Promise<Record<string, unknown>> {
    assertAllowedRequest(method, path);
    return this.transport.request(method.toUpperCase(), path, params);
  }
}

export function adaptConversationIndex(
  payload: Record<string, unknown>,
  options: { archived: boolean; offset: number; limit: number },
): AdaptedPage<ConversationSummary> {
  const { archived, offset, limit } = options;
  raiseIfHttpError(payload, MODERN_INDEX);
  if (payload.content_type === "text/html" || typeof payload.items === "string") {
    raiseIfAuthenticationRequired(payload, MODERN_INDEX);
    throw new AdapterError("unrecognized conversation index: HTML or non-list items");
  }
  const itemsRaw = Array.isArray(payload.items)
    ? payload.items
    : Array.isArray(payload.conversations)
      ? payload.conversations
      : null;
  if (!itemsRaw) {
    return {
      items: [],
      continuation: null,
      exhausted: false,
      paginationState: "unknown",
      schemaVersion: ADAPTER_VERSION,
      coverage: "unrecognized",
      warnings: ["missing items array"],
    };
  }
  const summaries: ConversationSummary[] = [];
  const warnings: string[] = [];
  for (const item of itemsRaw) {
    if (!isRecord(item)) {
      warnings.push("non-object conversation item");
      continue;
    }
    const conversationId = sanitizeToken(
      item.id ?? item.conversation_id ?? "",
    );
    if (!conversationId) {
      warnings.push("conversation missing id");
      continue;
    }
    const updatedAt = conversationUpdateTime(item, warnings);
    summaries.push({
      conversationId,
      createdAt: optionalString(item.create_time ?? item.created_at),
      updatedAt: updatedAt.value,
      isArchived: typeof item.is_archived === "boolean" ? item.is_archived : archived,
      workspaceId: optionalString(item.workspace_id),
      projectId: optionalString(item.gizmo_id ?? item.project_id),
      surface: classifySurface(item, { default: null }) as Surface,
      origin: summaryOrigin(item, warnings),
      hasVersions:
        typeof item.has_versions === "boolean" ? item.has_versions : null,
      currentNode: optionalString(item.current_node),
      coverage: updatedAt.warning === null ? "validated_page" : "partial",
    });
  }

  const totalValue = payload.total;
  const total =
    typeof totalValue === "number" &&
    Number.isInteger(totalValue) &&
    totalValue >= 0
      ? totalValue
      : null;
  const invalidTotal = totalValue !== undefined && total === null;
  if (invalidTotal) {
    warnings.push("invalid_total");
  }
  let continuation: string | number | null = null;
  let exhausted = false;
  let paginationState: PaginationState = "unknown";
  const pageEnd = offset + itemsRaw.length;
  const hasMissingConversations = payload.has_missing_conversations === true;
  const controls = readIndexPaginationControls(payload);
  warnings.push(...controls.warnings);
  if (hasMissingConversations) {
    warnings.push("index reported missing conversations");
  }
  if (controls.returnedOffsetPresent && controls.returnedOffset !== offset) {
    warnings.push("returned_index_offset_mismatch");
    paginationState = "contradictory";
    continuation = controls.returnedOffset ?? offset;
  } else if (controls.invalid) {
    paginationState = "unknown";
    continuation =
      controls.nextOffsetPresent && controls.nextOffset !== null
        ? controls.nextOffset
        : pageEnd > offset
          ? pageEnd
          : offset;
  } else if (controls.contradictory) {
    warnings.push("conflicting_pagination_controls");
    paginationState = "contradictory";
    continuation =
      controls.nextOffsetPresent && controls.nextOffset !== null
        ? controls.nextOffset
        : null;
  } else if (total !== null && total < offset) {
    warnings.push("total_before_offset");
    paginationState = "contradictory";
  } else if (total !== null && pageEnd > total) {
    warnings.push("page_exceeds_total");
    paginationState = "contradictory";
  } else if (controls.nextOffsetPresent || controls.hasMorePresent) {
    if (controls.hasMore === false) {
      if (controls.nextOffset !== null) {
        warnings.push("terminal_page_has_next_offset");
        continuation = controls.nextOffset;
        paginationState = "contradictory";
      } else if (total !== null && pageEnd < total) {
        warnings.push("has_more_false_before_reported_total");
        continuation = pageEnd;
        paginationState = "contradictory";
      } else {
        exhausted = !hasMissingConversations;
        paginationState = exhausted ? "complete" : "unknown";
      }
    } else if (controls.hasMore === true) {
      if (controls.nextOffset === null) {
        warnings.push("has_more_without_next_offset");
        continuation = pageEnd > offset ? pageEnd : offset;
        paginationState = "unknown";
      } else if (controls.nextOffset <= offset) {
        warnings.push("nonadvancing_next_offset");
        continuation = controls.nextOffset;
        paginationState = "contradictory";
      } else if (total !== null && controls.nextOffset >= total) {
        warnings.push("next_offset_after_reported_total");
        continuation = controls.nextOffset;
        paginationState = "contradictory";
      } else {
        continuation = controls.nextOffset;
        paginationState = "continuation";
      }
    } else if (controls.nextOffset === null) {
      if (total !== null && pageEnd < total) {
        warnings.push("missing_next_offset_before_reported_total");
        continuation = pageEnd;
        paginationState = "contradictory";
      } else {
        exhausted = !hasMissingConversations;
        paginationState = exhausted ? "complete" : "unknown";
      }
    } else if (controls.nextOffset <= offset) {
      warnings.push("nonadvancing_next_offset");
      continuation = controls.nextOffset;
      paginationState = "contradictory";
    } else if (total !== null && controls.nextOffset >= total) {
      warnings.push("next_offset_after_reported_total");
      continuation = controls.nextOffset;
      paginationState = "contradictory";
    } else {
      continuation = controls.nextOffset;
      paginationState = "continuation";
    }
  } else if (invalidTotal) {
    paginationState = "unknown";
  } else if (total !== null && itemsRaw.length < limit && pageEnd < total) {
    // A short page cannot prove exhaustion while the server says more items
    // remain. Keep a resumable offset and expose the contradiction.
    warnings.push("short_page_before_reported_total");
    continuation = pageEnd;
    paginationState = "contradictory";
  } else if (total !== null && pageEnd >= total) {
    exhausted = !hasMissingConversations;
    paginationState = exhausted ? "complete" : "unknown";
  } else if (itemsRaw.length < limit) {
    warnings.push("missing_pagination_controls");
    continuation = pageEnd > offset ? pageEnd : offset;
    paginationState = "unknown";
  } else {
    warnings.push("missing_pagination_controls");
    paginationState = "unknown";
    continuation = pageEnd;
  }

  let coverage: AdaptedPage<ConversationSummary>["coverage"] = "validated_page";
  if (paginationState === "unknown") {
    coverage = hasMissingConversations ? "unrecognized" : "partial";
  } else if (
    paginationState === "contradictory" ||
    warnings.length > 0
  ) {
    coverage = "partial";
  }

  return {
    items: summaries,
    continuation,
    exhausted,
    paginationState,
    schemaVersion: ADAPTER_VERSION,
    coverage,
    warnings,
  };
}

export function adaptConversationDetail(
  payload: Record<string, unknown>,
  conversationId: string,
  options: { detailRoute?: "modern" | "legacy" } = {},
): ConversationDetailProjection {
  const surface = classifySurface(payload, { default: null }) as Surface;
  const page = adaptMessagePage(payload, {
    conversationId,
    conversationSurface: surface,
  });
  const warnings = [...page.warnings];
  const updatedAt = conversationUpdateTime(payload, warnings);
  return {
    conversationId,
    createdAt: optionalString(payload.create_time ?? payload.created_at),
    updatedAt: updatedAt.value,
    currentNode: optionalString(payload.current_node),
    surface,
    detailRoute: options.detailRoute ?? "modern",
    messages: page.items,
    continuation:
      typeof page.continuation === "string" ? page.continuation : null,
    paginationState: page.paginationState,
    coverage:
      updatedAt.warning === null ? page.coverage : "partial",
    warnings,
  };
}

export function adaptMessagePage(
  payload: Record<string, unknown>,
  options: { conversationId: string; conversationSurface?: Surface },
): AdaptedPage<MessageRecord> {
  const { conversationId, conversationSurface = "unknown" } = options;
  const warnings: string[] = [];
  const records: MessageRecord[] = [];
  const hasMapping = isRecord(payload.mapping);
  const hasMessages = "messages" in payload;
  const messagesRaw = payload.messages;

  if (hasMapping) {
    records.push(
      ...iterMappingMessages(payload.mapping as Record<string, unknown>, {
        conversationId,
        warnings,
        conversationSurface: classifySurface(payload, {
          default: conversationSurface,
        }) as Surface,
      }),
    );
  } else if (hasMessages && Array.isArray(messagesRaw)) {
    for (const item of messagesRaw) {
      if (isRecord(item)) {
        const record = messageFromNode(item, {
          conversationId,
          warnings,
          conversationSurface,
        });
        if (record) {
          records.push(record);
        } else {
          warnings.push("message item missing id");
        }
      } else {
        warnings.push("non-object message item");
      }
    }
  } else {
    warnings.push("unrecognized_detail_shape");
    if (hasMessages) {
      warnings.push("messages_not_array");
    }
    if ("mapping" in payload) {
      warnings.push("mapping_not_object");
    }
    return {
      items: records,
      continuation: null,
      exhausted: false,
      paginationState: "unknown",
      schemaVersion: ADAPTER_VERSION,
      coverage: "unrecognized",
      warnings,
    };
  }

  if (hasMapping && !hasMessages && !("page_info" in payload)) {
    warnings.push("missing_pagination_controls");
    return {
      items: records,
      continuation: null,
      exhausted: false,
      paginationState: "unknown",
      schemaVersion: ADAPTER_VERSION,
      coverage: "partial",
      warnings,
    };
  }

  const pageInfo = payload.page_info;
  if (!isRecord(pageInfo)) {
    warnings.push("missing_pagination_controls");
    return {
      items: records,
      continuation: null,
      exhausted: false,
      paginationState: "unknown",
      schemaVersion: ADAPTER_VERSION,
      coverage: "unrecognized",
      warnings,
    };
  }

  const cursor = pageInfo.start_cursor;
  const hasPrevious = pageInfo.has_previous_page;
  let exhausted = false;
  let continuation: string | number | null = null;
  let paginationState: PaginationState = "unknown";
  let invalidPagination = false;
  if (typeof hasPrevious !== "boolean") {
    warnings.push("non_boolean_has_previous_page");
    invalidPagination = true;
  } else if (hasPrevious) {
    if (typeof cursor !== "string" || !cursor.trim()) {
      warnings.push("has_previous_page_without_start_cursor");
      invalidPagination = true;
    } else {
      continuation = cursor.trim();
      paginationState = "continuation";
    }
  } else if (cursor !== null && cursor !== undefined && cursor !== "") {
    warnings.push("terminal_page_has_cursor");
    invalidPagination = true;
  } else {
    exhausted = true;
    paginationState = "complete";
  }
  if (payload.repeated_cursor) {
    warnings.push("repeated_cursor");
    exhausted = false;
    continuation = null;
    paginationState = "repeated_cursor";
  }

  const coverage = invalidPagination
    ? "unrecognized"
    : warnings.length > 0
      ? "partial"
      : "validated_page";

  return {
    items: records,
    continuation,
    exhausted,
    paginationState: invalidPagination ? "contradictory" : paginationState,
    schemaVersion: ADAPTER_VERSION,
    coverage,
    warnings,
  };
}

function iterMappingMessages(
  mapping: Record<string, unknown>,
  options: {
    conversationId: string;
    warnings: string[];
    conversationSurface: Surface;
  },
): MessageRecord[] {
  const records: MessageRecord[] = [];
  for (const [nodeId, node] of Object.entries(mapping)) {
    if (!isRecord(node)) {
      options.warnings.push("non-object mapping node");
      continue;
    }
    const record = messageFromNode(node, {
      conversationId: options.conversationId,
      nodeId: String(nodeId),
      warnings: options.warnings,
      conversationSurface: options.conversationSurface,
    });
    if (record) {
      records.push(record);
    }
  }
  return records;
}

function messageFromNode(
  node: Record<string, unknown>,
  options: {
    conversationId: string;
    nodeId?: string;
    warnings: string[];
    conversationSurface: Surface;
  },
): MessageRecord | null {
  const messageRaw = node.message;
  const message: Record<string, unknown> = isRecord(messageRaw) ? messageRaw : node;
  const messageId =
    sanitizeToken(message.id) ??
    sanitizeToken(node.id) ??
    sanitizeToken(options.nodeId);
  if (!messageId) {
    options.warnings.push("message missing id");
    return null;
  }

  const authorRaw = message.author;
  const author: Record<string, unknown> = isRecord(authorRaw) ? authorRaw : {};
  const authorRole = sanitizeToken(author.role);
  const metadataRaw = message.metadata;
  const metadataProjection = sanitizeMetadataWithDiagnostics(
    isRecord(metadataRaw) ? metadataRaw : {},
  );
  const metadata = metadataProjection.metadata;
  if (metadataProjection.diagnostics.status !== "complete") {
    options.warnings.push(
      `metadata_projection_${metadataProjection.diagnostics.status}`,
    );
  }

  const childrenRaw = Array.isArray(node.children)
    ? node.children
    : Array.isArray(message.children)
      ? message.children
      : [];
  const children = childrenRaw
    .map((item) => sanitizeToken(item))
    .filter((item): item is string => item !== null);

  let requested: unknown;
  if (authorRole === "user") {
    requested =
      metadata.requested_model ??
      metadata.requested_model_slug ??
      metadata.model_slug;
  } else {
    requested = metadata.requested_model;
  }

  let recorded: unknown = null;
  if (authorRole === "assistant") {
    recorded =
      metadata.model_slug ??
      sanitizeToken(message.model_slug);
  }

  const conversationSurface = classifySurface(message, {
    default: classifySurface(metadata, { default: options.conversationSurface }),
  }) as Surface;

  return {
    conversationId: options.conversationId,
    messageId,
    nodeId: sanitizeToken(node.id) ?? sanitizeToken(options.nodeId) ?? messageId,
    parentId:
      sanitizeToken(node.parent) ??
      sanitizeToken(message.parent) ??
      sanitizeToken(metadata.parent_id),
    children,
    role: authorRole,
    channel: sanitizeToken(message.channel) ?? sanitizeToken(metadata.channel),
    createdAt: messageCreatedAt(message, node, options.warnings),
    status: sanitizeToken(message.status) ?? sanitizeToken(metadata.status),
    endTurn: typeof message.end_turn === "boolean" ? message.end_turn : null,
    requestedModelRaw:
      authorRole === "user"
        ? sanitizeToken(requested)
        : sanitizeToken(metadata.requested_model),
    requestedModeRaw: sanitizeToken(metadata.requested_mode),
    requestedReasoningEffortRaw: sanitizeToken(metadata.reasoning_effort),
    recordedFinalModelRaw: sanitizeToken(recorded),
    generationId:
      sanitizeToken(metadata.generation_id) ??
      sanitizeToken(metadata.message_request_id),
    requestId: sanitizeToken(metadata.request_id),
    surface: conversationSurface,
    origin:
      sanitizeToken(metadata.origin) ??
      (metadata.from_shared === true ? "shared" : null),
    metadata,
  };
}

export function raiseIfRateLimited(
  payload: Record<string, unknown>,
  path: string,
): void {
  const status = httpStatus(payload, path);
  if (status !== 429) {
    return;
  }
  throw new RateLimitedError(
    `rate limited (429) for ${path}; no legacy fallback and not quota exhaustion`,
    {
      status: 429,
      retryAfter: retryAfterValue(payload),
      path,
    },
  );
}

export function raiseIfHttpError(
  payload: Record<string, unknown>,
  path: string,
): void {
  const status = httpStatus(payload, path);
  if (status >= 200 && status < 300) {
    return;
  }
  if (status === 401 || status === 403) {
    raiseIfAuthenticationRequired(payload, path);
  }
  if (status === 429) {
    raiseIfRateLimited(payload, path);
  }
  throw new HttpStatusError(`HTTP ${status} for ${path}`, {
    status,
    path,
    retryAfter: retryAfterValue(payload),
  });
}

export function raiseIfAuthenticationRequired(
  payload: Record<string, unknown>,
  path: string,
): void {
  const status = httpStatus(payload, path);
  const contentType = String(payload.content_type ?? "").toLowerCase();
  if (
    ![401, 403].includes(status) &&
    !(status >= 200 && status < 300 && contentType.startsWith("text/html"))
  ) {
    return;
  }
  throw new AuthenticationRequiredError(
    `authentication required (${status}) for ${path}; legacy fallback is disabled`,
    {
      status: [401, 403].includes(status) ? status : 401,
      path,
    },
  );
}

function conversationUpdateTime(
  payload: Record<string, unknown>,
  warnings: string[],
): TimestampResult {
  let sawTimestamp = false;
  let invalidTimestamp = false;
  for (const key of ["update_time", "updated_at"]) {
    if (!Object.prototype.hasOwnProperty.call(payload, key)) {
      continue;
    }
    const raw = payload[key];
    if (raw === null || raw === undefined || raw === "") {
      continue;
    }
    sawTimestamp = true;
    const value = normalizeTimestamp(raw);
    if (value !== null) {
      if (invalidTimestamp) {
        warnings.push("conversation_invalid_update_time");
      }
      return {
        value,
        warning: invalidTimestamp ? "conversation_invalid_update_time" : null,
      };
    }
    invalidTimestamp = true;
  }
  const warning = sawTimestamp && invalidTimestamp
    ? "conversation_invalid_update_time"
    : "conversation_missing_update_time";
  warnings.push(warning);
  return { value: null, warning };
}

function messageCreatedAt(
  message: Record<string, unknown>,
  node: Record<string, unknown>,
  warnings: string[],
): string | null {
  let invalidTimestamp = false;
  for (const raw of [message.create_time, node.create_time]) {
    if (raw === null || raw === undefined || raw === "") {
      continue;
    }
    const value = normalizeTimestamp(raw);
    if (value !== null) {
      if (invalidTimestamp) {
        warnings.push("message_invalid_created_at");
      }
      return value;
    }
    invalidTimestamp = true;
  }
  if (invalidTimestamp) {
    warnings.push("message_invalid_created_at");
  }
  return null;
}

function readIndexPaginationControls(
  payload: Record<string, unknown>,
): IndexPaginationControls {
  const sources = [payload];
  for (const key of ["pagination", "page_info"]) {
    const value = payload[key];
    if (isRecord(value)) {
      sources.push(value);
    }
  }

  const nextOffsetRaw = readControlValues(sources, [
    "next_offset",
    "nextOffset",
    "next",
  ]);
  const hasMoreRaw = readControlValues(sources, [
    "has_more",
    "hasMore",
    "has_next_page",
    "hasNextPage",
  ]);
  const returnedOffsetRaw = readControlValues(sources, [
    "offset",
    "current_offset",
    "currentOffset",
  ]);
  const warnings: string[] = [];
  let invalid = false;
  let contradictory = false;

  let nextOffset: number | null = null;
  if (nextOffsetRaw.present) {
    const parsedValues = nextOffsetRaw.values.map((value) =>
      value === null || value === undefined ? null : nonNegativeInteger(value),
    );
    if (
      nextOffsetRaw.values.some(
        (value, index) =>
          value !== null &&
          value !== undefined &&
          parsedValues[index] === null,
      )
    ) {
      warnings.push("invalid_next_offset");
      invalid = true;
    } else if (hasConflictingControlValues(parsedValues)) {
      warnings.push("conflicting_next_offset");
      contradictory = true;
    } else {
      nextOffset = parsedValues[0] ?? null;
    }
  }

  let hasMore: boolean | null = null;
  if (hasMoreRaw.present) {
    const values = hasMoreRaw.values;
    if (values.some((value) => typeof value !== "boolean")) {
      warnings.push("invalid_has_more");
      invalid = true;
    } else if (hasConflictingControlValues(values)) {
      warnings.push("conflicting_has_more");
      contradictory = true;
    } else {
      hasMore = values[0] as boolean;
    }
  }

  let returnedOffset: number | null = null;
  if (returnedOffsetRaw.present) {
    const parsedValues = returnedOffsetRaw.values.map((value) =>
      value === null || value === undefined ? null : nonNegativeInteger(value),
    );
    if (
      returnedOffsetRaw.values.some(
        (value, index) =>
          value !== null &&
          value !== undefined &&
          parsedValues[index] === null,
      )
    ) {
      warnings.push("invalid_returned_index_offset");
      invalid = true;
    } else if (hasConflictingControlValues(parsedValues)) {
      warnings.push("conflicting_returned_index_offset");
      contradictory = true;
    } else {
      returnedOffset = parsedValues[0] ?? null;
    }
  }

  return {
    nextOffset,
    nextOffsetPresent: nextOffsetRaw.present,
    hasMore,
    hasMorePresent: hasMoreRaw.present,
    returnedOffset,
    returnedOffsetPresent: returnedOffsetRaw.present,
    invalid,
    contradictory,
    warnings,
  };
}

function readControlValues(
  sources: Array<Record<string, unknown>>,
  keys: string[],
): { present: boolean; value: unknown; values: unknown[] } {
  const values: unknown[] = [];
  for (const source of sources) {
    for (const key of keys) {
      if (Object.prototype.hasOwnProperty.call(source, key)) {
        values.push(source[key]);
      }
    }
  }
  return {
    present: values.length > 0,
    value: values[0],
    values,
  };
}

function hasConflictingControlValues(values: unknown[]): boolean {
  if (values.length < 2) {
    return false;
  }
  return values.slice(1).some((value) => !Object.is(value, values[0]));
}

function httpStatus(payload: Record<string, unknown>, path: string): number {
  const raw = payload.http_status;
  if (raw === null || raw === undefined) {
    return 200;
  }
  const parsed = nonNegativeInteger(raw);
  if (parsed === null || parsed < 100 || parsed > 599) {
    throw new CapabilityError(`invalid HTTP status for ${path}`, {
      capability: "http_status",
      path,
    });
  }
  return parsed;
}

function retryAfterValue(payload: Record<string, unknown>): string | null {
  const headers = isRecord(payload.headers) ? payload.headers : {};
  let retryAfter: unknown = payload.retry_after;
  if (retryAfter === null || retryAfter === undefined) {
    for (const key of RETRY_AFTER_HEADER_KEYS) {
      if (headers[key] !== undefined) {
        retryAfter = headers[key];
        break;
      }
    }
  }
  return retryAfter === null || retryAfter === undefined
    ? null
    : String(retryAfter);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function conversationPath(template: string, conversationId: string): string {
  const safeId = sanitizeToken(conversationId);
  if (safeId === null || safeId.includes("/")) {
    throw new AdapterError("conversation id is not a safe path token");
  }
  return template.replace("{conversation_id}", safeId);
}

function optionalString(value: unknown): string | null {
  if (value === null || value === undefined || value === "" || value === false) {
    return null;
  }
  if (value === true) {
    return "true";
  }
  return String(value);
}

function summaryOrigin(
  item: Record<string, unknown>,
  warnings: string[],
): string | null {
  const origin = optionalString(item.origin);
  if (origin !== null) {
    return origin;
  }
  const metadataRaw = item.metadata;
  if (!isRecord(metadataRaw)) {
    return null;
  }
  const metadataProjection = sanitizeMetadataWithDiagnostics(metadataRaw);
  if (metadataProjection.diagnostics.status !== "complete") {
    warnings.push(
      `metadata_projection_${metadataProjection.diagnostics.status}`,
    );
  }
  const metadata = metadataProjection.metadata;
  return (
    optionalString(metadata.origin) ??
    (metadata.imported === true ? "imported" : null) ??
    (metadata.from_copy === true ? "copied" : null)
  );
}

function normalizeTimestamp(value: unknown): string | null {
  if (typeof value === "number") {
    return epochTimestamp(value);
  }
  if (typeof value !== "string") {
    return null;
  }
  const text = value.trim();
  if (!text) {
    return null;
  }
  if (/^[+-]?(?:\d+|\d+\.\d+)$/.test(text)) {
    const numeric = Number(text);
    return epochTimestamp(numeric);
  }
  const parsed = Date.parse(text);
  return Number.isFinite(parsed) ? new Date(parsed).toISOString() : null;
}

function epochTimestamp(value: number): string | null {
  if (!Number.isFinite(value) || value < 0) {
    return null;
  }
  const milliseconds = value < 1_000_000_000_000 ? value * 1000 : value;
  const date = new Date(milliseconds);
  return Number.isFinite(date.getTime()) ? date.toISOString() : null;
}

function nonNegativeInteger(value: unknown): number | null {
  if (typeof value === "number") {
    return Number.isInteger(value) && value >= 0 ? value : null;
  }
  if (typeof value === "string" && /^\d+$/.test(value.trim())) {
    const parsed = Number(value);
    return Number.isSafeInteger(parsed) ? parsed : null;
  }
  return null;
}
