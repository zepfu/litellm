import { randomUUID } from "node:crypto";

import type {
  AdaptedPage,
  CapabilityRecord,
  ConversationDetailProjection,
  ConversationSummary,
  IdentityRecord,
  MessageRecord,
  PaginationState,
} from "../contracts/records.js";
import type {
  HistoryCollectionRequest,
  HistoryReader,
  RevisitEntry,
} from "../contracts/history.js";
import type {
  LedgerScope,
  ModelMappingVersion,
} from "../ledger/types.js";
import {
  AuthenticationRequiredError,
  RateLimitedError,
} from "../adapters/chatgpt/adapter.js";
import {
  ScheduleBridgeStore,
} from "../scheduler/bridge-store.js";
import type {
  CompleteTriggerRequest,
  ScheduleOptions,
  ScheduleScope,
} from "../scheduler/types.js";
import { runBoundedCollector, BoundedWorkerError, typedReadResult } from "./collect.js";
import {
  DEFAULT_WORKER_BOUNDS,
  MAX_REQUEST_ID_LENGTH,
  WORKER_PROTOCOL_VERSION,
  type CollectorRunEnvelope,
  type OpaquePageRequest,
  type PreparedHistory,
  type WorkerBounds,
  type WorkerControlMessage,
  type WorkerResponse,
  type WorkerStartRun,
} from "./contracts.js";
import type { WorkerBridge, WorkerOperation } from "./contracts.js";
import type {
  BridgeStateEnvelope,
  BridgePageMutation,
} from "../history/bridge-checkpoints.js";

const MAX_QUEUE_PAGES = 16;

export async function runStdioWorker(
  input: NodeJS.ReadableStream = process.stdin,
  output: NodeJS.WritableStream = process.stdout,
): Promise<void> {
  const inbox = new FrameInbox(input);
  const first = await inbox.next();
  const start = parseStartRun(first);
  inbox.setBounds(start.bounds);
  const app = new WorkerClientApplication(start, inbox, output);
  await app.run();
}

class WorkerClientApplication {
  private readonly controller = new AbortController();
  private readonly bridge: ParentWorkerBridge;
  private triggerId: string | null = null;
  private stateVersion = 0;

  constructor(
    private readonly start: WorkerStartRun,
    private readonly inbox: FrameInbox,
    private readonly output: NodeJS.WritableStream,
  ) {
    this.bridge = new ParentWorkerBridge(
      start.envelope,
      start.bounds,
      inbox,
      output,
      (message) => this.handleControl(message),
    );
  }

  async run(): Promise<void> {
    try {
      const schedule = this.createScheduleStore();
      const scope = scheduleScope(this.start.envelope);
      const at = Date.now();
      let scheduleState = await schedule.ensureSchedule(
        scope,
        this.start.scheduleOptions as ScheduleOptions,
        at,
      );
      if (this.start.authenticationRecoveryRequested === true) {
        const recovered = await schedule.recoverAuthentication(scope, Date.now());
        if (recovered.state) {
          scheduleState = recovered.state;
        }
      }
      const claim = await schedule.claimTrigger(scope, Date.now());
      if (!claim) {
        throw new BoundedWorkerError("state_conflict", true, true);
      }
      this.triggerId = claim.trigger.triggerId;
      scheduleState = claim.state;

      const prepared = await this.bridge.prepareHistory(this.triggerId);
      const reader = new ParentHistoryReader(this.bridge, prepared);
      const request = this.start.collectionRequest as unknown as HistoryCollectionRequest;
      const mapping = this.start.mapping as ModelMappingVersion;
      const collection = await runBoundedCollector({
        envelope: this.start.envelope,
        bridge: this.bridge,
        reader,
        scope: prepared.scope,
        mapping,
        request,
        bounds: {
          maxFrameBytes: this.start.bounds.maxFrameBytes,
          maxTotalBytes: this.start.bounds.maxTotalBytes,
          maxRequests: this.start.bounds.maxRequests,
          deadlineAt: monotonicNow() + this.start.bounds.remainingMs,
        },
        signal: this.controller.signal,
      });
      const completion = completionFor(collection.result, Date.now());
      const completed = await schedule.completeTrigger(
        scope,
        Date.now(),
        completion,
      );
      scheduleState = completed.state ?? scheduleState;
      this.stateVersion = this.bridge.stateVersion;
      await this.bridge.finishRun({
        expectedVersion: this.stateVersion,
        triggerId: this.triggerId,
        outcome: completion.outcome,
        summary: {
          status: collection.result.status,
          coverageIncomplete: collection.coverageIncomplete,
          warnings: collection.warnings,
          scheduleState,
        },
      });
    } catch (error) {
      if (this.triggerId !== null) {
        try {
          await this.bridge.cancel({
            expectedVersion: this.bridge.stateVersion,
            triggerId: this.triggerId,
            outcome: "cancelled_after_start",
            summary: {
              errorCode: errorCode(error),
              coverageIncomplete: true,
            },
          });
        } catch {
          // Parent owns the final cancellation fallback and lease cleanup.
        }
      }
      throw error;
    }
  }

  private createScheduleStore(): ScheduleBridgeStore {
    const transport = {
      request: async (
        operation: "loadState" | "compareAndSetState",
        payload: unknown,
      ) => {
        const result = await this.bridge.requestRaw(operation, payload);
        const stateVersion =
          typeof result.stateVersion === "number"
            ? result.stateVersion
            : this.bridge.stateVersion;
        return {
          ok: true as const,
          result: {
            state: (result.state ?? null) as never,
            stateVersion,
          },
        };
      },
    };
    return new ScheduleBridgeStore(transport, {
      runId: this.start.envelope.runId,
      bindingGeneration: this.start.envelope.bindingGeneration,
      leaseFencingToken: this.start.envelope.leaseFencingToken,
    });
  }

  private handleControl(message: WorkerControlMessage): void {
    if (message.control !== "cancelRun") {
      throw new BoundedWorkerError("protocol_invalid", false, true);
    }
    validateEnvelope(controlEnvelope(message), this.start.envelope);
    this.controller.abort();
  }
}

class ParentWorkerBridge implements WorkerBridge {
  private requestSequence = 0;
  stateVersion = 0;

  constructor(
    private readonly envelope: CollectorRunEnvelope,
    private readonly bounds: WorkerStartRun["bounds"],
    private readonly inbox: FrameInbox,
    private readonly output: NodeJS.WritableStream,
    private readonly onControl: (message: WorkerControlMessage) => void,
  ) {}

  async requestRaw(
    operation: WorkerOperation,
    payload: unknown,
  ): Promise<Record<string, unknown>> {
    const requestId = this.nextRequestId();
    await writeFrame(
      this.output,
      {
        protocolVersion: WORKER_PROTOCOL_VERSION,
        requestId,
        ...this.envelope,
        operation,
        payload,
      },
      this.bounds,
    );
    while (true) {
      const message = await this.inbox.next();
      if (isControlMessage(message)) {
        this.onControl(message);
        continue;
      }
      const response = parseWorkerResponse(message);
      if (response.requestId !== requestId) {
        throw new BoundedWorkerError("protocol_invalid", false, true);
      }
      if (!response.ok) {
        throw new BoundedWorkerError(
          response.error.code,
          response.error.retryable,
          response.error.coverageIncomplete,
        );
      }
      const result = isRecord(response.result) ? response.result : {};
      if (typeof result.stateVersion === "number") {
        this.stateVersion = result.stateVersion;
      }
      return result;
    }
  }

  async prepareHistory(triggerId: string): Promise<PreparedHistory> {
    const result = await this.requestControl(
      "prepareHistory",
      { claimedTriggerId: triggerId },
    );
    return parsePreparedHistory(result, this.envelope.collectorAccountId);
  }

  async loadState(
    request: {
      kind?: "header" | "schedule" | "candidates";
      cursor?: string | null;
      limit?: number;
      expectedStateVersion?: number;
    } = {},
  ): Promise<BridgeStateEnvelope | null> {
    const headerResult = await this.requestRaw("loadState", {
      kind: "header",
    });
    const stateVersion =
      typeof headerResult.stateVersion === "number"
        ? headerResult.stateVersion
        : 0;
    const state = stateFromHeader(
      headerResult.header,
      this.envelope.collectorAccountId,
      stateVersion,
    );
    const candidates = await this.readCandidateQueue(stateVersion);
    mergeQueueItems(state, candidates.items);
    state.queueCoverage = candidates.complete ? "complete" : "partial";
    this.stateVersion = stateVersion;
    return state;
  }

  async compareAndSetState(envelope: {
    expectedStateVersion: number;
    next: BridgeStateEnvelope;
  }): Promise<{ stateVersion: number }> {
    const result = await this.requestRaw("compareAndSetState", {
      kind: "history",
      expectedStateVersion: envelope.expectedStateVersion,
      state: envelope.next,
    });
    const stateVersion = Number(result.stateVersion);
    if (!Number.isSafeInteger(stateVersion)) {
      throw new BoundedWorkerError("protocol_invalid", false, true);
    }
    this.stateVersion = stateVersion;
    return { stateVersion };
  }

  async readHistory(request: OpaquePageRequest) {
    const result = await this.requestRaw("readHistory", request);
    if (result.status === "blocked" || result.status === "unavailable") {
      if (result.reason === "authentication") {
        throw new AuthenticationRequiredError(
          "native Chat history requires authentication",
          { status: 401, path: "history" },
        );
      }
      if (result.reason === "cooldown") {
        const retryAfterMs =
          typeof result.retryAfterMs === "number" &&
          Number.isFinite(result.retryAfterMs) &&
          result.retryAfterMs >= 0
            ? result.retryAfterMs
            : null;
        throw new RateLimitedError(
          "native Chat history is rate limited",
          {
            ...(retryAfterMs === null
              ? {}
              : { retryAfter: new Date(Date.now() + retryAfterMs).toISOString() }),
            path: "history",
          },
        );
      }
      throw new BoundedWorkerError(
        "history_contract_unavailable",
        true,
        true,
      );
    }
    return typedReadResultFromWire(result);
  }

  async loadConversationMetadata(request: {
    conversationId: string;
    cursor?: string | null;
    limit?: number;
    snapshotId?: string | null;
  }) {
    const result = await this.requestRaw("loadConversationMetadata", request);
    return parseConversationMetadata(result, request.conversationId);
  }

  async commitPage(payload: {
    pageCommitId: string;
    expectedStateVersion?: number;
    source?: unknown;
    mutations: BridgePageMutation;
  }): Promise<{ acknowledged: true; stateVersion?: number }> {
    const result = await this.requestRaw("commitPage", payload);
    return {
      acknowledged: true,
      ...(typeof result.stateVersion === "number"
        ? { stateVersion: result.stateVersion }
        : {}),
    };
  }

  async loadReportSnapshot(request: {
    cursor?: string | null;
    limit?: number;
    snapshotId?: string | null;
  } = {}): Promise<Record<string, unknown>> {
    return this.requestRaw("loadReportSnapshot", request);
  }

  async finishRun(payload: {
    expectedVersion: number;
    triggerId: string;
    outcome: "success" | "failure" | "authentication";
    summary?: Record<string, unknown>;
  }): Promise<{ finished: true; stateVersion?: number }> {
    const result = await this.requestRaw("finishRun", payload);
    return {
      finished: true,
      ...(typeof result.stateVersion === "number"
        ? { stateVersion: result.stateVersion }
        : {}),
    };
  }

  async cancel(payload: {
    expectedVersion: number;
    triggerId: string;
    outcome?: string;
    summary?: Record<string, unknown>;
  }): Promise<{ cancelled: true; stateVersion?: number }> {
    const result = await this.requestRaw("cancel", payload);
    return {
      cancelled: true,
      ...(typeof result.stateVersion === "number"
        ? { stateVersion: result.stateVersion }
        : {}),
    };
  }

  private async readCandidateQueue(
    expectedStateVersion: number,
  ): Promise<{ items: Record<string, unknown>[]; complete: boolean }> {
    const items: Record<string, unknown>[] = [];
    let cursor: string | null = null;
    for (let page = 0; page < MAX_QUEUE_PAGES; page += 1) {
      const result = await this.requestRaw("loadState", {
        kind: "candidates",
        cursor,
        limit: 64,
        expectedStateVersion,
      });
      const pageItems = Array.isArray(result.items)
        ? result.items.filter(isRecord)
        : [];
      items.push(...pageItems);
      if (result.hasMore !== true) {
        return { items, complete: true };
      }
      cursor = typeof result.nextCursor === "string" ? result.nextCursor : null;
      if (cursor === null) {
        return { items, complete: false };
      }
    }
    return { items, complete: false };
  }

  private async requestControl(
    control: "prepareHistory",
    payload: unknown,
  ): Promise<Record<string, unknown>> {
    const requestId = this.nextRequestId();
    await writeFrame(
      this.output,
      {
        protocolVersion: WORKER_PROTOCOL_VERSION,
        control,
        requestId,
        envelope: this.envelope,
        payload,
      },
      this.bounds,
    );
    while (true) {
      const message = await this.inbox.next();
      if (isControlMessage(message)) {
        this.onControl(message);
        continue;
      }
      const response = parseWorkerResponse(message);
      if (response.requestId !== requestId) {
        throw new BoundedWorkerError("protocol_invalid", false, true);
      }
      if (!response.ok) {
        throw new BoundedWorkerError(
          response.error.code,
          response.error.retryable,
          response.error.coverageIncomplete,
        );
      }
      return isRecord(response.result) ? response.result : {};
    }
  }

  private nextRequestId(): string {
    this.requestSequence += 1;
    const suffix = `${this.requestSequence}-${randomUUID()}`;
    return suffix.slice(0, MAX_REQUEST_ID_LENGTH);
  }
}

class ParentHistoryReader implements HistoryReader {
  readonly capabilities: CapabilityRecord;

  constructor(
    private readonly bridge: ParentWorkerBridge,
    private readonly prepared: PreparedHistory,
  ) {
    this.capabilities = prepared.capabilities;
  }

  async inspectSessionIdentity(): Promise<IdentityRecord> {
    return this.prepared.identity;
  }

  async listConversations(options: {
    archived: boolean;
    offset?: number;
    limit?: number;
    order?: string;
  }): Promise<AdaptedPage<ConversationSummary>> {
    this.requireCapability("index");
    const result = await this.bridge.readHistory({
      kind: "index",
      requiredCapability: "index",
      ...options,
    });
    return pageFromWire(result, (item) => item as ConversationSummary);
  }

  async fetchConversation(
    conversationId: string,
    options?: { allowLegacyFallback?: boolean },
  ): Promise<ConversationDetailProjection> {
    this.requireCapability("modern_detail");
    const result = await this.bridge.readHistory({
      kind: "detail",
      requiredCapability: "modern_detail",
      conversationId,
      ...options,
    });
    if (!result.detail || !isRecord(result.detail)) {
      throw new BoundedWorkerError("history_reader_failed", true, true);
    }
    return result.detail as ConversationDetailProjection;
  }

  async fetchMessages(
    conversationId: string,
    options?: {
      before?: string | null;
      numTurns?: number;
      conversationSurface?: ConversationSummary["surface"];
    },
  ): Promise<AdaptedPage<MessageRecord>> {
    this.requireCapability("messages");
    const result = await this.bridge.readHistory({
      kind: "messages",
      requiredCapability: "messages",
      conversationId,
      ...options,
    });
    return pageFromWire(result, (item) => item as MessageRecord);
  }

  private requireCapability(
    capability: "index" | "modern_detail" | "messages",
  ): void {
    const capabilities = this.capabilities;
    const available =
      capability === "index"
        ? capabilities.indexScopes.length > 0
        : capability === "modern_detail"
          ? capabilities.modernDetail !== "unavailable"
          : capabilities.pagination !== "unavailable";
    if (!available) {
      throw new BoundedWorkerError(
        "history_contract_unavailable",
        true,
        true,
      );
    }
  }
}

class FrameInbox {
  private maxFrameBytes = DEFAULT_WORKER_BOUNDS.maxFrameBytes;
  private maxTotalBytes = DEFAULT_WORKER_BOUNDS.maxTotalBytes;
  private totalBytes = 0;
  private buffer = Buffer.alloc(0);
  private readonly frames: unknown[] = [];
  private readonly waiters: Array<{
    resolve: (value: unknown) => void;
    reject: (error: unknown) => void;
  }> = [];
  private closedError: Error | null = null;

  constructor(input: NodeJS.ReadableStream) {
    void this.pump(input);
  }

  setBounds(bounds: WorkerStartRun["bounds"]): void {
    this.maxFrameBytes = Math.min(
      this.maxFrameBytes,
      positiveBound(bounds.maxFrameBytes, DEFAULT_WORKER_BOUNDS.maxFrameBytes),
    );
    this.maxTotalBytes = Math.min(
      this.maxTotalBytes,
      positiveBound(bounds.maxTotalBytes, DEFAULT_WORKER_BOUNDS.maxTotalBytes),
    );
  }

  next(): Promise<unknown> {
    if (this.frames.length > 0) {
      return Promise.resolve(this.frames.shift());
    }
    if (this.closedError) {
      return Promise.reject(this.closedError);
    }
    return new Promise((resolve, reject) => {
      this.waiters.push({ resolve, reject });
    });
  }

  private async pump(input: NodeJS.ReadableStream): Promise<void> {
    try {
      for await (const chunk of input as AsyncIterable<Buffer | string>) {
        const bytes = Buffer.isBuffer(chunk)
          ? chunk
          : Buffer.from(chunk, "utf8");
        this.totalBytes += bytes.length;
        if (this.totalBytes > this.maxTotalBytes) {
          throw new BoundedWorkerError("bounds_exceeded", true, true);
        }
        this.buffer = Buffer.concat([this.buffer, bytes]);
        this.drainFrames();
      }
      if (this.buffer.length > 0) {
        throw new BoundedWorkerError("protocol_invalid", false, true);
      }
      this.fail(new BoundedWorkerError("history_reader_failed", true, true));
    } catch (error) {
      this.fail(error instanceof Error ? error : new Error(String(error)));
    }
  }

  private drainFrames(): void {
    while (true) {
      const newline = this.buffer.indexOf(0x0a);
      if (newline < 0) {
        if (this.buffer.length >= this.maxFrameBytes) {
          throw new BoundedWorkerError("bounds_exceeded", true, true);
        }
        return;
      }
      const frame = this.buffer.subarray(0, newline);
      this.buffer = this.buffer.subarray(newline + 1);
      if (frame.length + 1 > this.maxFrameBytes) {
        throw new BoundedWorkerError("bounds_exceeded", true, true);
      }
      let value: unknown;
      try {
        value = JSON.parse(frame.toString("utf8")) as unknown;
      } catch {
        throw new BoundedWorkerError("protocol_invalid", false, true);
      }
      this.push(value);
    }
  }

  private push(value: unknown): void {
    const waiter = this.waiters.shift();
    if (waiter) {
      waiter.resolve(value);
    } else {
      this.frames.push(value);
    }
  }

  private fail(error: Error): void {
    if (this.closedError) {
      return;
    }
    this.closedError = error;
    for (const waiter of this.waiters.splice(0)) {
      waiter.reject(error);
    }
  }
}

async function writeFrame(
  output: NodeJS.WritableStream,
  value: Record<string, unknown>,
  bounds: WorkerStartRun["bounds"],
): Promise<void> {
  const encoded = `${JSON.stringify(value)}\n`;
  if (
    Buffer.byteLength(encoded, "utf8") >
    positiveBound(bounds.maxFrameBytes, DEFAULT_WORKER_BOUNDS.maxFrameBytes)
  ) {
    throw new BoundedWorkerError("bounds_exceeded", true, true);
  }
  const writable = output as NodeJS.WritableStream & {
    write(chunk: string): boolean;
    once(event: string, listener: () => void): void;
  };
  if (!writable.write(encoded)) {
    await new Promise<void>((resolve) => writable.once("drain", resolve));
  }
}

function parseStartRun(value: unknown): WorkerStartRun {
  if (
    !isRecord(value) ||
    value.protocolVersion !== WORKER_PROTOCOL_VERSION ||
    value.control !== "startRun" ||
    !isRecord(value.envelope) ||
    !isRecord(value.bounds) ||
    !isRecord(value.scheduleOptions) ||
    !isRecord(value.collectionRequest) ||
    !isRecord(value.mapping)
  ) {
    throw new BoundedWorkerError("protocol_invalid", false, true);
  }
  const envelope = parseEnvelope(value.envelope);
  const remainingMs = Number(value.bounds.remainingMs);
  if (!Number.isFinite(remainingMs) || remainingMs <= 0) {
    throw new BoundedWorkerError("bounds_exceeded", true, true);
  }
  return {
    protocolVersion: WORKER_PROTOCOL_VERSION,
    control: "startRun",
    envelope,
    scheduleOptions: value.scheduleOptions,
    collectionRequest: value.collectionRequest,
    mapping: value.mapping as unknown as ModelMappingVersion,
    ...(value.authenticationRecoveryRequested === true
      ? { authenticationRecoveryRequested: true }
      : {}),
    bounds: {
      maxFrameBytes: positiveBound(
        value.bounds.maxFrameBytes,
        DEFAULT_WORKER_BOUNDS.maxFrameBytes,
      ),
      maxTotalBytes: positiveBound(
        value.bounds.maxTotalBytes,
        DEFAULT_WORKER_BOUNDS.maxTotalBytes,
      ),
      maxRequests: positiveBound(
        value.bounds.maxRequests,
        DEFAULT_WORKER_BOUNDS.maxRequests,
      ),
      remainingMs,
    },
  };
}

function parseWorkerResponse(value: unknown): WorkerResponse {
  if (
    !isRecord(value) ||
    value.protocolVersion !== WORKER_PROTOCOL_VERSION ||
    typeof value.requestId !== "string" ||
    typeof value.ok !== "boolean"
  ) {
    throw new BoundedWorkerError("protocol_invalid", false, true);
  }
  return value as unknown as WorkerResponse;
}

function parseEnvelope(value: unknown): CollectorRunEnvelope {
  const bindingGeneration = isRecord(value) ? value.bindingGeneration : undefined;
  const leaseFencingToken = isRecord(value)
    ? value.leaseFencingToken
    : undefined;
  if (
    !isRecord(value) ||
    typeof value.runId !== "string" ||
    value.runId.trim() === "" ||
    typeof value.collectorAccountId !== "string" ||
    value.collectorAccountId.trim() === "" ||
    typeof value.profileId !== "string" ||
    value.profileId.trim() === "" ||
    typeof bindingGeneration !== "number" ||
    !Number.isSafeInteger(bindingGeneration) ||
    bindingGeneration < 0 ||
    typeof leaseFencingToken !== "number" ||
    !Number.isSafeInteger(leaseFencingToken) ||
    leaseFencingToken < 0
  ) {
    throw new BoundedWorkerError("protocol_invalid", false, true);
  }
  return {
    runId: value.runId,
    collectorAccountId: value.collectorAccountId,
    profileId: value.profileId,
    bindingGeneration,
    leaseFencingToken,
  };
}

function parsePreparedHistory(
  value: Record<string, unknown>,
  accountId: string,
): PreparedHistory {
  if (
    !isRecord(value.identity) ||
    !isRecord(value.capabilities) ||
    !isRecord(value.capabilityManifest) ||
    !isRecord(value.scope)
  ) {
    throw new BoundedWorkerError(
      "history_contract_unavailable",
      true,
      true,
    );
  }
  const identity = value.identity as unknown as IdentityRecord;
  const scope = scopeFromWire(value.scope, accountId);
  if (
    identity.authState !== "ready" ||
    identity.surface !== "chat" ||
    scope.collectorAccountId !== accountId
  ) {
    throw new BoundedWorkerError(
      "history_contract_unavailable",
      true,
      true,
    );
  }
  const capabilities = value.capabilities as unknown as CapabilityRecord;
  if (!capabilities.adapterVersion || capabilities.indexScopes.length === 0) {
    throw new BoundedWorkerError(
      "history_contract_unavailable",
      true,
      true,
    );
  }
  return {
    identity,
    capabilities,
    capabilityManifest: value.capabilityManifest,
    scope,
  };
}

function stateFromHeader(
  rawHeader: unknown,
  accountId: string,
  stateVersionCounter: number,
): BridgeStateEnvelope {
  const empty: BridgeStateEnvelope = {
    stateVersion: 1,
    collectorAccountId: accountId,
    stateVersionCounter,
    discovery: {},
    revisits: [],
    queueCoverage: "complete",
  };
  if (!isRecord(rawHeader) || !isRecord(rawHeader.checkpoint)) {
    return empty;
  }
  const checkpoint = rawHeader.checkpoint;
  if (
    checkpoint.stateVersion === 1 &&
    checkpoint.collectorAccountId === accountId &&
    isRecord(checkpoint.discovery) &&
    Array.isArray(checkpoint.revisits)
  ) {
    return {
      ...empty,
      ...(checkpoint as unknown as BridgeStateEnvelope),
      stateVersionCounter,
    };
  }
  if (
    (checkpoint.scope === "active" || checkpoint.scope === "archived") &&
    typeof checkpoint.accountId === "string"
  ) {
    return {
      ...empty,
      discovery: {
        [checkpoint.scope]: checkpoint as never,
      },
    };
  }
  return empty;
}

function mergeQueueItems(
  state: BridgeStateEnvelope,
  items: readonly Record<string, unknown>[],
): void {
  const revisits = new Map(
    state.revisits.map((entry) => [entry.conversationId, entry]),
  );
  for (const item of items) {
    const payload = isRecord(item.payload) ? item.payload : item;
    const queueKind =
      payload.queueKind ??
      (String(item.candidateKey).startsWith("revisit:")
        ? "revisit"
        : "candidate");
    if (queueKind === "revisit" && isRecord(payload.entry)) {
      const entry = payload.entry as unknown as RevisitEntry;
      revisits.set(entry.conversationId, entry);
      continue;
    }
    const candidate = isRecord(payload.candidate)
      ? payload.candidate
      : payload;
    const candidateSummary = isRecord(candidate.summary)
      ? candidate.summary
      : null;
    const candidateConversationId =
      candidateSummary &&
      typeof candidateSummary.conversationId === "string"
        ? candidateSummary.conversationId
        : null;
    const scope =
      candidate.scope === "archived" ||
      candidateSummary?.isArchived === true
        ? "archived"
        : candidate.scope === "active"
          ? "active"
          : null;
    if (!scope || !candidateSummary || candidateConversationId === null) {
      continue;
    }
    const checkpoint = state.discovery[scope];
    if (!checkpoint) {
      continue;
    }
    const queue = checkpoint.candidateQueue ?? [];
    if (
      !queue.some(
        (existing) =>
          existing.summary.conversationId === candidateConversationId,
      )
    ) {
      checkpoint.candidateQueue = [
        ...queue,
        candidate as never,
      ];
    }
  }
  state.revisits = [...revisits.values()];
}

function typedReadResultFromWire(
  result: Record<string, unknown>,
): ReturnType<typeof typedReadResult> {
  const items = Array.isArray(result.items)
    ? result.items.filter(isRecord)
    : [];
  const normalized = {
    items,
    continuation:
      typeof result.continuation === "string" ||
      typeof result.continuation === "number"
        ? result.continuation
        : null,
    exhausted: result.exhausted === true,
    paginationState: isPaginationState(result.paginationState)
      ? result.paginationState
      : "unknown",
    schemaVersion:
      typeof result.schemaVersion === "string"
        ? result.schemaVersion
        : "chatgpt-chat-history-v1",
    coverage: isCoverage(result.coverage) ? result.coverage : "unrecognized",
    warnings: Array.isArray(result.warnings)
      ? result.warnings.filter((item): item is string => typeof item === "string")
      : [],
  } as unknown as Awaited<ReturnType<HistoryReader["listConversations"]>>;
  return typedReadResult(normalized);
}

function pageFromWire<T>(
  result: ReturnType<typeof typedReadResultFromWire>,
  project: (item: ConversationSummary | MessageRecord) => T,
): AdaptedPage<T> {
  const items = result.records.map(project);
  return {
    items,
    continuation: result.continuation,
    exhausted: result.exhausted,
    paginationState: result.paginationState,
    schemaVersion: result.schemaVersion ?? "chatgpt-chat-history-v1",
    coverage: result.coverage,
    warnings: result.warnings,
  };
}

function parseConversationMetadata(
  result: Record<string, unknown>,
  conversationId: string,
): {
  summary: ConversationSummary;
  messages?: MessageRecord[];
  attempts?: import("../ledger/types.js").ReconstructedAttempt[];
} {
  const summary = isRecord(result.summary)
    ? (result.summary as unknown as ConversationSummary)
    : null;
  const items = Array.isArray(result.items)
    ? result.items.filter(isRecord)
    : [];
  const firstSummary =
    summary ??
    (items.find(
      (item) => "conversationId" in item && "isArchived" in item,
    ) as unknown as ConversationSummary | undefined);
  if (!firstSummary) {
    throw new BoundedWorkerError("conversation_not_found", true, true);
  }
  const messages = Array.isArray(result.messages)
    ? result.messages.filter(isRecord)
    : items.flatMap((item) =>
        Array.isArray(item.messages)
          ? item.messages.filter(isRecord)
          : [],
      );
  const attempts = Array.isArray(result.attempts)
    ? result.attempts.filter(isRecord)
    : undefined;
  return {
    summary: {
      ...firstSummary,
      conversationId:
        firstSummary.conversationId === conversationId
          ? conversationId
          : firstSummary.conversationId,
    },
    ...(messages.length > 0
      ? { messages: messages as unknown as MessageRecord[] }
      : {}),
    ...(attempts
      ? {
          attempts:
            attempts as unknown as import("../ledger/types.js").ReconstructedAttempt[],
        }
      : {}),
  };
}

function scheduleScope(envelope: CollectorRunEnvelope): ScheduleScope {
  return {
    collectorAccountId: envelope.collectorAccountId,
    profileId: envelope.profileId,
  };
}

function completionFor(
  result: {
    status: "complete" | "partial" | "blocked";
    accountState: {
      status: "ready" | "paused";
      reason: "authentication" | "cooldown" | null;
      cooldownUntil: string | null;
    };
  },
  now: number,
): CompleteTriggerRequest {
  const retryAfterMs =
    result.accountState.cooldownUntil === null
      ? null
      : Math.max(
          0,
          new Date(result.accountState.cooldownUntil).getTime() - now,
        );
  return {
    outcome:
      result.accountState.reason === "authentication"
        ? "authentication"
        : result.status === "complete"
          ? "success"
          : "failure",
    retryAfterMs,
  };
}

function scopeFromWire(
  value: Record<string, unknown>,
  accountId: string,
): LedgerScope {
  return {
    collectorAccountId: accountId,
    provider: String(value.provider ?? ""),
    providerUserId: stringOrNull(value.providerUserId ?? value.provider_user_id),
    workspaceId: stringOrNull(value.workspaceId ?? value.workspace_id),
    quotaOwnerId: stringOrNull(value.quotaOwnerId ?? value.quota_owner_id),
    surface: String(value.surface ?? "unknown") as LedgerScope["surface"],
  };
}

function validateEnvelope(
  actual: CollectorRunEnvelope,
  expected: CollectorRunEnvelope,
): void {
  if (
    actual.runId !== expected.runId ||
    actual.collectorAccountId !== expected.collectorAccountId ||
    actual.profileId !== expected.profileId ||
    actual.bindingGeneration !== expected.bindingGeneration ||
    actual.leaseFencingToken !== expected.leaseFencingToken
  ) {
    throw new BoundedWorkerError("fence_invalid", false, true);
  }
}

function controlEnvelope(message: WorkerControlMessage): CollectorRunEnvelope {
  return parseEnvelope(
    message.envelope ?? {
      runId: message.runId,
      collectorAccountId: message.collectorAccountId,
      profileId: message.profileId,
      bindingGeneration: message.bindingGeneration,
      leaseFencingToken: message.leaseFencingToken,
    },
  );
}

function isControlMessage(value: unknown): value is WorkerControlMessage {
  return (
    isRecord(value) &&
    value.protocolVersion === WORKER_PROTOCOL_VERSION &&
    (value.control === "cancelRun" || value.control === "prepareHistory")
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isPaginationState(value: unknown): value is PaginationState {
  return (
    value === "complete" ||
    value === "continuation" ||
    value === "contradictory" ||
    value === "unknown" ||
    value === "repeated_cursor" ||
    value === "budget_exhausted"
  );
}

function isCoverage(
  value: unknown,
): "validated_page" | "partial" | "unrecognized" {
  return value === "validated_page" || value === "partial" || value === "unrecognized"
    ? value
    : "unrecognized";
}

function stringOrNull(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null;
}

function positiveBound(value: unknown, fallback: number): number {
  const parsed = Number(value);
  return Number.isSafeInteger(parsed) && parsed > 0 ? parsed : fallback;
}

function monotonicNow(): number {
  return typeof performance !== "undefined" ? performance.now() : Date.now();
}

function errorCode(error: unknown): string {
  if (error instanceof BoundedWorkerError) {
    return error.code;
  }
  if (error instanceof Error && error.name) {
    return error.name.toLowerCase().replace(/[^a-z0-9]+/g, "_");
  }
  return "worker_error";
}

if (process.argv.includes("--stdio-v1")) {
  runStdioWorker().catch((error: unknown) => {
    process.exitCode = error instanceof BoundedWorkerError ? 1 : 1;
  });
}
