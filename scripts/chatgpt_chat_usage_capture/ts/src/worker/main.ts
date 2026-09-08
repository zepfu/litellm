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
  HistoryMetadataPage,
  HistoryReader,
  RevisitEntry,
} from "../contracts/history.js";
import type {
  IngestContext,
  LedgerScope,
  ModelMappingVersion,
} from "../ledger/types.js";
import {
  buildUsageReport,
  type UsageReportRequest,
  type UsageReportSnapshot,
} from "../counting/report.js";
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
  type CommitPagePayload,
  type OpaquePageRequest,
  type PreparedHistory,
  type WorkerControlMessage,
  type WorkerErrorCode,
  type WorkerResponse,
  type WorkerStartRun,
} from "./contracts.js";
import type { WorkerBridge, WorkerOperation } from "./contracts.js";
import type {
  BridgeStateEnvelope,
} from "../history/bridge-checkpoints.js";

const MAX_REPORT_PAGES = 100;
const TERMINAL_REQUEST_RESERVE = 3;
const TERMINAL_FRAME_RESERVE = 6;
// Terminal state/summary objects share PostgreSQL's 64 KiB state-field bound.
const MAX_TERMINAL_FRAME_BYTES = 64 * 1024 + 4 * 1024;

export async function runStdioWorker(
  input: NodeJS.ReadableStream = process.stdin,
  output: NodeJS.WritableStream = process.stdout,
): Promise<void> {
  const budget = new WireBudget(DEFAULT_WORKER_BOUNDS);
  const inbox = new FrameInbox(input, budget);
  let app: WorkerClientApplication | null = null;
  try {
    const first = await inbox.next();
    budget.consumeRequest();
    const start = parseStartRun(first);
    const deadlineAt = monotonicNow() + start.bounds.remainingMs;
    budget.configure(start.bounds, deadlineAt);
    inbox.setBounds();
    app = new WorkerClientApplication(
      start,
      deadlineAt,
      budget,
      inbox,
      output,
    );
    await app.run();
  } finally {
    app?.close();
    if (app === null) {
      inbox.close();
    }
  }
}

class WorkerClientApplication {
  private readonly controller = new AbortController();
  private readonly bridge: ParentWorkerBridge;
  private triggerId: string | null = null;
  private finishRequested = false;

  constructor(
    private readonly start: WorkerStartRun,
    private readonly deadlineAt: number,
    budget: WireBudget,
    private readonly inbox: FrameInbox,
    private readonly output: NodeJS.WritableStream,
  ) {
    this.bridge = new ParentWorkerBridge(
      start.envelope,
      budget,
      inbox,
      output,
      this.controller.signal,
      (message) => this.handleControl(message),
    );
  }

  async run(): Promise<void> {
    const schedule = this.createScheduleStore();
    const scope = scheduleScope(this.start.envelope);
    try {
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
      const triggerId = claim.trigger.triggerId;
      this.triggerId = triggerId;
      scheduleState = claim.state;

      const prepared = await this.bridge.prepareHistory(triggerId);
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
          deadlineAt: this.deadlineAt,
        },
        signal: this.controller.signal,
        authenticationRecoveryRequested:
          this.start.authenticationRecoveryRequested === true,
      });
      const report = await this.collectReport(
        this.start.reportRequest ??
          defaultReportRequest(collection.result),
      );
      const completion = completionFor(
        collection.result,
        Date.now(),
        report?.truncated === true,
      );
      this.bridge.enterTerminalPhase();
      const completed = await schedule.completeTrigger(
        scope,
        Date.now(),
        completion,
      );
      if (!completed.applied) {
        throw new BoundedWorkerError("state_conflict", true, true);
      }
      scheduleState = completed.state ?? scheduleState;
      const finished = await this.bridge.finishRun({
        expectedVersion: this.bridge.stateVersion,
        triggerId,
        outcome: completion.outcome,
        summary: {
          status: collection.result.status,
          coverageIncomplete:
            collection.coverageIncomplete || report?.truncated === true,
          warnings: [
            ...collection.warnings,
            ...(report?.warnings ?? []),
          ],
          scheduleState,
          ...(report?.report === undefined
            ? {}
            : { report: report.report }),
        },
      });
      if (!finished.finished) {
        throw new BoundedWorkerError("protocol_invalid", false, true);
      }
      this.finishRequested = true;
    } catch (error) {
      if (this.triggerId !== null && !this.finishRequested && !this.bridge.remoteCancelled) {
        if (isAccountBlockingError(error)) {
          try {
            this.bridge.enterTerminalPhase();
            const completion = completionForError(error, Date.now());
            const completed = await schedule.completeTrigger(
              scope,
              Date.now(),
              completion,
            );
            if (completed.applied) {
              const finished = await this.bridge.finishRun({
                expectedVersion: this.bridge.stateVersion,
                triggerId: this.triggerId,
                outcome: completion.outcome,
                summary: {
                  errorCode: errorCode(error),
                  coverageIncomplete: true,
                },
              });
              if (finished.finished) {
                this.finishRequested = true;
              }
            }
          } catch {
            // Fall through to the bounded cancellation path below.
          }
        }
        if (this.finishRequested) {
          return;
        }
        try {
          const triggerId = this.triggerId;
          if (triggerId === null) {
            throw error;
          }
          this.bridge.enterTerminalPhase();
          await this.bridge.cancel({
            expectedVersion: this.bridge.stateVersion,
            triggerId,
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
    } finally {
      this.close();
    }
  }

  close(): void {
    this.bridge.close();
  }

  private async collectReport(
    request: Record<string, unknown>,
  ): Promise<{
    report?: Record<string, unknown>;
    truncated: boolean;
    warnings: string[];
  }> {
    const attempts: UsageReportSnapshot["attempts"][number][] = [];
    let snapshotId: string | null = null;
    let cursor: string | null = null;
    let first: Record<string, unknown> | null = null;
    let truncated = false;
    const warnings: string[] = [];
    const seenCursors = new Set<string>();
    let expectedSnapshotId: string | null | undefined;
    let snapshotInitialized = false;

    for (let page = 0; page < MAX_REPORT_PAGES; page += 1) {
      let current: Record<string, unknown>;
      try {
        current = await this.bridge.loadReportSnapshot({
          cursor,
          snapshotId,
          limit: 256,
        });
      } catch (error) {
        if (error instanceof BoundedWorkerError && error.code === "bounds_exceeded") {
          truncated = true;
          warnings.push("report_snapshot_truncated");
          break;
        }
        throw error;
      }
      first ??= current;
      const terminalPage =
        current.hasMore === false &&
        (current.nextCursor === undefined || current.nextCursor === null);
      if (!snapshotInitialized) {
        expectedSnapshotId =
          typeof current.snapshotId === "string" || current.snapshotId === null
            ? current.snapshotId
            : undefined;
        snapshotInitialized = true;
      } else if (
        current.snapshotId !== expectedSnapshotId &&
        !(current.snapshotId === null && terminalPage)
      ) {
        truncated = true;
        warnings.push("report_snapshot_changed");
        break;
      }
      if (
        !Array.isArray(current.attempts) ||
        current.attempts.some((attempt) => !isRecord(attempt))
      ) {
        truncated = true;
        warnings.push("report_snapshot_invalid_attempts");
        break;
      }
      const pageAttempts = current.attempts;
      attempts.push(...pageAttempts as UsageReportSnapshot["attempts"][number][]);
      if (terminalPage && current.truncated === false) {
        truncated = false;
      } else {
        truncated ||= current.truncated === true;
      }
      if (terminalPage) {
        if (
          current.nextCursor !== undefined &&
          current.nextCursor !== null
        ) {
          truncated = true;
          warnings.push("report_snapshot_contradictory_continuation");
        }
        break;
      }
      if (current.hasMore !== true) {
        truncated = true;
        warnings.push("report_snapshot_missing_continuation_state");
        break;
      }
      const nextCursor =
        typeof current.nextCursor === "string" ? current.nextCursor : null;
      const nextSnapshot: string | null =
        typeof current.snapshotId === "string"
          ? current.snapshotId
          : snapshotId;
      if (
        nextCursor === null ||
        nextSnapshot === null ||
        seenCursors.has(nextCursor)
      ) {
        truncated = true;
        warnings.push("report_snapshot_invalid_continuation");
        break;
      }
      seenCursors.add(nextCursor);
      cursor = nextCursor;
      snapshotId = nextSnapshot;
      if (page === MAX_REPORT_PAGES - 1) {
        truncated = true;
        warnings.push("report_snapshot_page_budget_exhausted");
      }
    }

    if (!first) {
      return {
        truncated: true,
        warnings: [...warnings, "report_snapshot_unavailable"],
      };
    }
    const snapshot = reportSnapshotFromWire(
      first,
      this.start.envelope.collectorAccountId,
      attempts,
      truncated,
      warnings,
    );
    const report = buildUsageReport(
      snapshot,
      request as unknown as UsageReportRequest,
    );
    return {
      report: report as unknown as Record<string, unknown>,
      truncated,
      warnings,
    };
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
    this.bridge.markRemoteCancellation();
    this.controller.abort();
  }
}

class ParentWorkerBridge implements WorkerBridge {
  private requestSequence = 0;
  private pendingRequestId: string | null = null;
  private closed = false;
  remoteCancelled = false;
  stateVersion = 0;

  constructor(
    private readonly envelope: CollectorRunEnvelope,
    private readonly budget: WireBudget,
    private readonly inbox: FrameInbox,
    private readonly output: NodeJS.WritableStream,
    private readonly signal: AbortSignal,
    private readonly onControl: (message: WorkerControlMessage) => void,
  ) {
    this.inbox.setAbortSignal(signal);
    this.inbox.setControlHandler(onControl);
  }

  async requestRaw(
    operation: WorkerOperation,
    payload: unknown,
  ): Promise<Record<string, unknown>> {
    this.assertRequestAvailable();
    const requestId = this.nextRequestId();
    this.beginRequest(requestId);
    try {
      this.budget.consumeRequest();
      await writeFrame(
        this.output,
        {
          protocolVersion: WORKER_PROTOCOL_VERSION,
          requestId,
          ...this.envelope,
          operation,
          payload,
        },
        this.budget,
        this.signal,
      );
      const result = await this.readResponse(requestId);
      if (typeof result.stateVersion === "number") {
        this.stateVersion = result.stateVersion;
      }
      return result;
    } finally {
      this.endRequest(requestId);
    }
  }

  async prepareHistory(triggerId: string): Promise<PreparedHistory> {
    this.assertRequestAvailable();
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
    const kind = request.kind ?? "header";
    if (kind === "header") {
      const headerResult = await this.requestRaw("loadState", { kind });
      const stateVersion = requiredStateVersion(
        headerResult.stateVersion ?? 0,
      );
      const state = stateFromHeader(
        headerResult.header,
        this.envelope.collectorAccountId,
        stateVersion,
      );
      if (state === null && stateVersion !== 0) {
        throw new BoundedWorkerError("protocol_invalid", false, true);
      }
      this.headerState = state;
      this.candidateState = null;
      this.candidateStateVersion = null;
      this.stateVersion = stateVersion;
      return state;
    }
    if (kind !== "candidates") {
      throw new BoundedWorkerError("operation_unsupported", false, true);
    }
    const expectedStateVersion = requiredStateVersion(
      request.expectedStateVersion,
    );
    if (
      this.headerState === null ||
      this.headerState.stateVersionCounter !== expectedStateVersion
    ) {
      throw new BoundedWorkerError("state_conflict", true, true);
    }
    const cursor = request.cursor ?? null;
    if (
      cursor === null ||
      this.candidateState === null ||
      this.candidateStateVersion !== expectedStateVersion
    ) {
      this.candidateState = structuredClone(this.headerState);
      for (const checkpoint of Object.values(this.candidateState.discovery)) {
        checkpoint.candidateQueue = [];
      }
      this.candidateStateVersion = expectedStateVersion;
    }
    const state = this.candidateState;
    const candidates = await this.readCandidateQueue(
      expectedStateVersion,
      cursor,
      request.limit,
    );
    const merged = mergeQueueItems(state, candidates.items);
    state.queueCoverage =
      !candidates.hasMore && merged ? "complete" : "partial";
    state.nextCursor = candidates.nextCursor;
    state.hasMore = candidates.hasMore;
    this.candidateState = state;
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
    const stateVersion = requiredStateVersion(result.stateVersion);
    this.stateVersion = stateVersion;
    return { stateVersion };
  }

  async readHistory(request: OpaquePageRequest) {
    const result = await this.requestRaw("readHistory", request);
    throwIfDomainBlocked(result, "history");
    return typedReadResultFromWire(result, request.kind);
  }

  async loadConversationMetadata(request: {
    conversationId: string;
    cursor?: string | null;
    limit?: number;
    snapshotId?: string | null;
  }): Promise<HistoryMetadataPage> {
    const result = await this.requestRaw("loadConversationMetadata", request);
    throwIfDomainBlocked(result, "history_metadata");
    return parseConversationMetadata(result, request.conversationId);
  }

  async commitPage(
    payload: CommitPagePayload,
  ): Promise<{ acknowledged: true; stateVersion: number; pageCommitId: string }> {
    const result = await this.requestRaw("commitPage", payload);
    const stateVersion = requiredStateVersion(result.stateVersion);
    if (
      result.acknowledged !== true ||
      typeof result.pageCommitId !== "string" ||
      result.pageCommitId !== payload.pageCommitId ||
      stateVersion <= payload.expectedStateVersion
    ) {
      throw new BoundedWorkerError("protocol_invalid", false, true);
    }
    return {
      acknowledged: true,
      stateVersion,
      pageCommitId: result.pageCommitId,
    };
  }

  async loadReportSnapshot(request: {
    cursor?: string | null;
    limit?: number;
    snapshotId?: string | null;
  } = {}): Promise<Record<string, unknown>> {
    const result = await this.requestRaw("loadReportSnapshot", request);
    throwIfDomainBlocked(result, "history_report");
    return result;
  }

  async finishRun(payload: {
    expectedVersion: number;
    triggerId: string;
    outcome: "success" | "failure" | "authentication";
    summary?: Record<string, unknown>;
  }): Promise<{ finished: true; stateVersion: number }> {
    const result = await this.requestRaw("finishRun", payload);
    const stateVersion = requiredStateVersion(result.stateVersion);
    if (result.finished !== true || stateVersion <= payload.expectedVersion) {
      throw new BoundedWorkerError("protocol_invalid", false, true);
    }
    return {
      finished: true,
      stateVersion,
    };
  }

  async cancel(payload: {
    expectedVersion: number;
    triggerId: string;
    outcome?: string;
    summary?: Record<string, unknown>;
  }): Promise<{ cancelled: true; stateVersion: number }> {
    const result = await this.requestRaw("cancel", payload);
    const stateVersion = requiredStateVersion(result.stateVersion);
    if (result.cancelled !== true || stateVersion <= payload.expectedVersion) {
      throw new BoundedWorkerError("protocol_invalid", false, true);
    }
    return {
      cancelled: true,
      stateVersion,
    };
  }

  markRemoteCancellation(): void {
    this.remoteCancelled = true;
  }

  enterTerminalPhase(): void {
    this.budget.enterTerminalPhase();
  }

  close(): void {
    if (this.closed) {
      return;
    }
    this.closed = true;
    this.inbox.close();
  }

  private headerState: BridgeStateEnvelope | null = null;
  private candidateState: BridgeStateEnvelope | null = null;
  private candidateStateVersion: number | null = null;

  private async readCandidateQueue(
    expectedStateVersion: number,
    cursor: string | null,
    limit = 64,
  ): Promise<{
    items: Record<string, unknown>[];
    nextCursor: string | null;
    hasMore: boolean;
  }> {
    const result = await this.requestRaw("loadState", {
      kind: "candidates",
      cursor,
      limit,
      expectedStateVersion,
    });
    if (requiredStateVersion(result.stateVersion) !== expectedStateVersion) {
      throw new BoundedWorkerError("state_conflict", true, true);
    }
    const pageItems = Array.isArray(result.items)
      ? result.items.filter(isRecord)
      : [];
    if (result.hasMore !== true && result.hasMore !== false) {
      throw new BoundedWorkerError("protocol_invalid", false, true);
    }
    const nextCursor =
      typeof result.nextCursor === "string" ? result.nextCursor : null;
    if (result.hasMore === true && nextCursor === null) {
      throw new BoundedWorkerError("protocol_invalid", false, true);
    }
    if (result.hasMore === false && nextCursor !== null) {
      throw new BoundedWorkerError("protocol_invalid", false, true);
    }
    return { items: pageItems, nextCursor, hasMore: result.hasMore };
  }

  private async requestControl(
    control: "prepareHistory",
    payload: unknown,
  ): Promise<Record<string, unknown>> {
    this.assertRequestAvailable();
    const requestId = this.nextRequestId();
    this.beginRequest(requestId);
    try {
      this.budget.consumeRequest();
      await writeFrame(
        this.output,
        {
          protocolVersion: WORKER_PROTOCOL_VERSION,
          control,
          requestId,
          envelope: this.envelope,
          payload,
        },
        this.budget,
        this.signal,
      );
      return await this.readResponse(requestId);
    } finally {
      this.endRequest(requestId);
    }
  }

  private async readResponse(
    requestId: string,
  ): Promise<Record<string, unknown>> {
    while (true) {
      const message = await this.inbox.next(this.signal);
      if (isControlMessage(message)) {
        this.onControl(message);
        this.assertRequestAvailable();
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

  private beginRequest(requestId: string): void {
    if (this.pendingRequestId !== null) {
      throw new BoundedWorkerError("protocol_invalid", false, true);
    }
    this.pendingRequestId = requestId;
  }

  private endRequest(requestId: string): void {
    if (this.pendingRequestId === requestId) {
      this.pendingRequestId = null;
    }
  }

  private assertRequestAvailable(): void {
    if (this.closed || this.remoteCancelled || this.signal.aborted) {
      throw new BoundedWorkerError("cancelled", false, true);
    }
    this.budget.assertActive(this.signal);
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

class WireBudget {
  private maxFrameBytes: number;
  private maxTotalBytes: number;
  private maxRequests: number;
  private terminalBytesReserve: number;
  private deadlineAt = Number.POSITIVE_INFINITY;
  private totalBytes = 0;
  private requestCount = 0;
  private largestFrameBytes = 0;
  private terminalPhase = false;

  constructor(bounds: {
    maxFrameBytes: number;
    maxTotalBytes: number;
    maxRequests: number;
  }) {
    this.maxFrameBytes = bounds.maxFrameBytes;
    this.maxTotalBytes = bounds.maxTotalBytes;
    this.maxRequests = bounds.maxRequests;
    this.terminalBytesReserve =
      TERMINAL_FRAME_RESERVE *
      Math.min(this.maxFrameBytes, MAX_TERMINAL_FRAME_BYTES);
  }

  configure(
    bounds: WorkerStartRun["bounds"],
    deadlineAt: number,
  ): void {
    this.maxFrameBytes = boundedValue(
      bounds.maxFrameBytes,
      DEFAULT_WORKER_BOUNDS.maxFrameBytes,
      "maxFrameBytes",
    );
    this.maxTotalBytes = boundedValue(
      bounds.maxTotalBytes,
      DEFAULT_WORKER_BOUNDS.maxTotalBytes,
      "maxTotalBytes",
    );
    this.maxRequests = boundedValue(
      bounds.maxRequests,
      DEFAULT_WORKER_BOUNDS.maxRequests,
      "maxRequests",
    );
    this.terminalBytesReserve =
      TERMINAL_FRAME_RESERVE *
      Math.min(this.maxFrameBytes, MAX_TERMINAL_FRAME_BYTES);
    this.deadlineAt = deadlineAt;
    this.terminalPhase = false;
    if (
      this.totalBytes > this.maxTotalBytes ||
      this.largestFrameBytes > this.maxFrameBytes
    ) {
      throw new BoundedWorkerError("bounds_exceeded", true, true);
    }
    this.assertActive();
  }

  consumeRequest(): void {
    this.assertActive();
    if (this.requestCount >= this.requestLimit()) {
      throw new BoundedWorkerError("bounds_exceeded", true, true);
    }
    this.requestCount += 1;
  }

  consumeIncomingControl(): void {
    this.assertActive();
    if (this.requestCount >= this.maxRequests) {
      throw new BoundedWorkerError("bounds_exceeded", true, true);
    }
    this.requestCount += 1;
  }

  consumeBytes(bytes: number): void {
    if (!Number.isSafeInteger(bytes) || bytes < 0) {
      throw new BoundedWorkerError("protocol_invalid", false, true);
    }
    if (this.totalBytes + bytes > this.byteLimit()) {
      throw new BoundedWorkerError("bounds_exceeded", true, true);
    }
    this.totalBytes += bytes;
    this.assertActive();
  }

  assertFrameSize(bytes: number): void {
    this.largestFrameBytes = Math.max(this.largestFrameBytes, bytes);
    if (bytes > this.maxFrameBytes) {
      throw new BoundedWorkerError("bounds_exceeded", true, true);
    }
  }

  enterTerminalPhase(): void {
    this.terminalPhase = true;
    this.assertActive();
  }

  assertActive(signal?: AbortSignal): void {
    if (signal?.aborted) {
      throw new BoundedWorkerError("cancelled", false, true);
    }
    if (monotonicNow() >= this.deadlineAt) {
      throw new BoundedWorkerError("bounds_exceeded", true, true);
    }
  }

  remainingMs(): number {
    return Math.max(0, this.deadlineAt - monotonicNow());
  }

  private requestLimit(): number {
    return this.terminalPhase
      ? this.maxRequests
      : Math.max(0, this.maxRequests - TERMINAL_REQUEST_RESERVE);
  }

  private byteLimit(): number {
    return this.terminalPhase
      ? this.maxTotalBytes
      : Math.max(0, this.maxTotalBytes - this.terminalBytesReserve);
  }
}

class FrameInbox {
  private readonly frames: unknown[] = [];
  private buffer = Buffer.alloc(0);
  private readonly waiters: Array<{
    resolve: (value: unknown) => void;
    reject: (error: unknown) => void;
  }> = [];
  private closedError: Error | null = null;
  private closed = false;
  private controlHandler: ((message: WorkerControlMessage) => void) | null =
    null;

  constructor(
    private readonly input: NodeJS.ReadableStream,
    private readonly budget: WireBudget,
  ) {
    void this.pump(input);
  }

  setBounds(): void {
    this.budget.assertFrameSize(this.buffer.length + 1);
    this.budget.assertActive();
  }

  setAbortSignal(_signal: AbortSignal): void {
    // The bridge passes its signal to each bounded read; no stream-wide
    // listener is retained here, which keeps cancellation cleanup deterministic.
  }

  setControlHandler(handler: (message: WorkerControlMessage) => void): void {
    this.controlHandler = handler;
    const queuedControls = this.frames.filter(isControlMessage);
    if (queuedControls.length === 0) {
      return;
    }
    this.frames.splice(
      0,
      this.frames.length,
      ...this.frames.filter((frame) => !isControlMessage(frame)),
    );
    for (const control of queuedControls) {
      handler(control);
    }
  }

  next(signal?: AbortSignal): Promise<unknown> {
    this.budget.assertActive(signal);
    if (this.frames.length > 0) {
      return Promise.resolve(this.frames.shift());
    }
    if (this.closedError) {
      return Promise.reject(this.closedError);
    }
    return new Promise((resolve, reject) => {
      let entry:
        | {
            resolve: (value: unknown) => void;
            reject: (error: unknown) => void;
        }
        | undefined;
      let timer: ReturnType<typeof setTimeout> | undefined;
      const cleanup = (): void => {
        signal?.removeEventListener("abort", onAbort);
        if (timer !== undefined) {
          clearTimeout(timer);
        }
      };
      const onAbort = (): void => {
        const index = entry === undefined ? -1 : this.waiters.indexOf(entry);
        if (index >= 0) {
          this.waiters.splice(index, 1);
        }
        cleanup();
        reject(new BoundedWorkerError("cancelled", false, true));
      };
      if (signal) {
        signal.addEventListener("abort", onAbort, { once: true });
        if (signal.aborted) {
          onAbort();
          return;
        }
      }
      entry = {
        resolve: (value) => {
          cleanup();
          resolve(value);
        },
        reject: (error) => {
          cleanup();
          reject(error);
        },
      };
      this.waiters.push(entry);
      const remaining = this.budget.remainingMs();
      if (Number.isFinite(remaining)) {
        timer = setTimeout(() => {
          const index = this.waiters.indexOf(entry!);
          if (index >= 0) {
            this.waiters.splice(index, 1);
          }
          cleanup();
          reject(new BoundedWorkerError("bounds_exceeded", true, true));
        }, Math.max(1, remaining));
      }
    });
  }

  private async pump(input: NodeJS.ReadableStream): Promise<void> {
    try {
      for await (const chunk of input as AsyncIterable<Buffer | string>) {
        if (this.closed) {
          return;
        }
        const bytes = Buffer.isBuffer(chunk)
          ? chunk
          : Buffer.from(chunk, "utf8");
        this.budget.consumeBytes(bytes.length);
        this.buffer = Buffer.concat([this.buffer, bytes]);
        this.drainFrames();
      }
      if (this.closed) {
        return;
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
        this.budget.assertFrameSize(this.buffer.length + 1);
        if (this.buffer.length >= DEFAULT_WORKER_BOUNDS.maxFrameBytes) {
          throw new BoundedWorkerError("bounds_exceeded", true, true);
        }
        return;
      }
      const frame = this.buffer.subarray(0, newline);
      this.buffer = this.buffer.subarray(newline + 1);
      this.budget.assertFrameSize(frame.length + 1);
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
    if (isControlMessage(value)) {
      this.budget.consumeIncomingControl();
      if (this.controlHandler) {
        this.controlHandler(value);
        return;
      }
    }
    const waiter = this.waiters.shift();
    if (waiter) {
      waiter.resolve(value);
    } else {
      this.frames.push(value);
    }
  }

  private fail(error: Error): void {
    if (this.closed || this.closedError) {
      return;
    }
    this.closedError = error;
    for (const waiter of this.waiters.splice(0)) {
      waiter.reject(error);
    }
  }

  close(): void {
    if (this.closed) {
      return;
    }
    this.frames.splice(0);
    this.fail(new BoundedWorkerError("cancelled", false, true));
    this.closed = true;
    const destroyable = this.input as NodeJS.ReadableStream & {
      destroy?: (error?: Error) => void;
    };
    destroyable.destroy?.();
  }
}

async function writeFrame(
  output: NodeJS.WritableStream,
  value: Record<string, unknown>,
  budget: WireBudget,
  signal: AbortSignal,
): Promise<void> {
  budget.assertActive(signal);
  const encoded = `${JSON.stringify(value)}\n`;
  const bytes = Buffer.byteLength(encoded, "utf8");
  budget.assertFrameSize(bytes);
  budget.consumeBytes(bytes);
  const writable = output as NodeJS.WritableStream & {
    write(chunk: string): boolean;
    once(event: string, listener: () => void): void;
    removeListener?(event: string, listener: () => void): void;
  };
  if (!writable.write(encoded)) {
    await new Promise<void>((resolve, reject) => {
      let timer: ReturnType<typeof setTimeout> | undefined;
      let settled = false;
      const cleanup = (): void => {
        signal.removeEventListener("abort", onAbort);
        writable.removeListener?.("drain", onDrain);
        if (timer !== undefined) {
          clearTimeout(timer);
        }
      };
      const onAbort = (): void => {
        if (settled) {
          return;
        }
        settled = true;
        cleanup();
        reject(new BoundedWorkerError("cancelled", false, true));
      };
      const onDrain = (): void => {
        if (settled) {
          return;
        }
        settled = true;
        cleanup();
        try {
          budget.assertActive(signal);
          resolve();
        } catch (error) {
          reject(error);
        }
      };
      signal.addEventListener("abort", onAbort, { once: true });
      if (signal.aborted) {
        onAbort();
        return;
      }
      const remaining = budget.remainingMs();
      if (Number.isFinite(remaining)) {
        timer = setTimeout(() => {
          if (settled) {
            return;
          }
          settled = true;
          cleanup();
          reject(new BoundedWorkerError("bounds_exceeded", true, true));
        }, Math.max(1, remaining));
      }
      writable.once("drain", onDrain);
    });
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
  if (!Number.isSafeInteger(remainingMs) || remainingMs <= 0) {
    throw new BoundedWorkerError("bounds_exceeded", true, true);
  }
  const reportRequest =
    value.reportRequest === undefined
      ? undefined
      : isRecord(value.reportRequest)
        ? value.reportRequest
        : null;
  if (reportRequest === null) {
    throw new BoundedWorkerError("protocol_invalid", false, true);
  }
  return {
    protocolVersion: WORKER_PROTOCOL_VERSION,
    control: "startRun",
    envelope,
    scheduleOptions: value.scheduleOptions,
    collectionRequest: value.collectionRequest,
    mapping: value.mapping as unknown as ModelMappingVersion,
    ...(reportRequest === undefined ? {} : { reportRequest }),
    ...(value.authenticationRecoveryRequested === true
      ? { authenticationRecoveryRequested: true }
      : {}),
    bounds: {
      maxFrameBytes: boundedValue(
        value.bounds.maxFrameBytes,
        DEFAULT_WORKER_BOUNDS.maxFrameBytes,
        "maxFrameBytes",
      ),
      maxTotalBytes: boundedValue(
        value.bounds.maxTotalBytes,
        DEFAULT_WORKER_BOUNDS.maxTotalBytes,
        "maxTotalBytes",
      ),
      maxRequests: boundedValue(
        value.bounds.maxRequests,
        DEFAULT_WORKER_BOUNDS.maxRequests,
        "maxRequests",
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
  if (
    value.ok === false &&
    (!isRecord(value.error) ||
      typeof value.error.code !== "string" ||
      typeof value.error.retryable !== "boolean" ||
      typeof value.error.coverageIncomplete !== "boolean" ||
      !isWorkerErrorCode(value.error.code))
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
  const identity = parseIdentity(value.identity);
  const scope = scopeFromWire(value.scope);
  if (
    identity.authState !== "ready" ||
    identity.surface !== "chat" ||
    scope.collectorAccountId !== accountId ||
    scope.providerUserId !== identity.providerUserId ||
    scope.workspaceId !== identity.workspaceId ||
    scope.quotaOwnerId !== identity.quotaOwnerId ||
    scope.surface !== identity.surface
  ) {
    throw new BoundedWorkerError(
      "fence_invalid",
      false,
      true,
    );
  }
  const capabilities = parseCapabilities(value.capabilities);
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

function parseIdentity(value: unknown): IdentityRecord {
  if (
    !isRecord(value) ||
    !("providerUserId" in value) ||
    !("workspaceId" in value) ||
    !("quotaOwnerId" in value) ||
    !isSurface(value.surface) ||
    !isAuthState(value.authState) ||
    !Array.isArray(value.identityErrors) ||
    value.identityErrors.some((item) => typeof item !== "string")
  ) {
    throw new BoundedWorkerError("history_contract_unavailable", true, true);
  }
  const fields = ["providerUserId", "workspaceId", "quotaOwnerId"] as const;
  if (
    fields.some(
      (field) =>
        value[field] !== null && typeof value[field] !== "string",
    )
  ) {
    throw new BoundedWorkerError("history_contract_unavailable", true, true);
  }
  return {
    providerUserId: value.providerUserId as string | null,
    workspaceId: value.workspaceId as string | null,
    quotaOwnerId: value.quotaOwnerId as string | null,
    surface: value.surface,
    authState: value.authState,
    identityErrors: [...value.identityErrors],
  };
}

function parseCapabilities(value: unknown): CapabilityRecord {
  if (
    !isRecord(value) ||
    typeof value.adapterVersion !== "string" ||
    !Array.isArray(value.indexScopes) ||
    value.indexScopes.some((item) => typeof item !== "string") ||
    typeof value.archiveBehavior !== "string" ||
    typeof value.projectCoverage !== "string" ||
    typeof value.modernDetail !== "string" ||
    typeof value.pagination !== "string" ||
    typeof value.legacySupport !== "string" ||
    typeof value.branchVisibility !== "string" ||
    typeof value.modelMetadata !== "string" ||
    typeof value.quotaMetadata !== "string" ||
    !Array.isArray(value.warnings) ||
    value.warnings.some((item) => typeof item !== "string")
  ) {
    throw new BoundedWorkerError("history_contract_unavailable", true, true);
  }
  return {
    adapterVersion: value.adapterVersion,
    indexScopes: [...value.indexScopes],
    archiveBehavior: value.archiveBehavior,
    projectCoverage: value.projectCoverage as CapabilityRecord["projectCoverage"],
    modernDetail: value.modernDetail,
    pagination: value.pagination,
    legacySupport: value.legacySupport,
    branchVisibility: value.branchVisibility,
    modelMetadata: value.modelMetadata,
    quotaMetadata: value.quotaMetadata,
    warnings: [...value.warnings],
  };
}

function stateFromHeader(
  rawHeader: unknown,
  accountId: string,
  stateVersionCounter: number,
): BridgeStateEnvelope | null {
  const empty: BridgeStateEnvelope = {
    stateVersion: 1,
    collectorAccountId: accountId,
    stateVersionCounter,
    discovery: {},
    queueCoverage: "partial",
  };
  if (rawHeader === null || rawHeader === undefined) {
    return stateVersionCounter === 0 ? null : empty;
  }
  if (!isRecord(rawHeader)) {
    throw new BoundedWorkerError("protocol_invalid", false, true);
  }
  if (rawHeader.checkpoint === null || rawHeader.checkpoint === undefined) {
    return stateVersionCounter === 0 ? null : empty;
  }
  if (!isRecord(rawHeader.checkpoint)) {
    throw new BoundedWorkerError("protocol_invalid", false, true);
  }
  const checkpoint = rawHeader.checkpoint;
  if (
    checkpoint.stateVersion === 1 &&
    checkpoint.collectorAccountId === accountId &&
    isRecord(checkpoint.discovery)
  ) {
    const discovery: BridgeStateEnvelope["discovery"] = {};
    for (const scope of ["active", "archived"] as const) {
      const rawCheckpoint = checkpoint.discovery[scope];
      if (!isRecord(rawCheckpoint)) {
        continue;
      }
      const { candidateQueue: _candidateQueue, ...headerCheckpoint } =
        rawCheckpoint;
      discovery[scope] = headerCheckpoint as unknown as NonNullable<
        BridgeStateEnvelope["discovery"][typeof scope]
      >;
    }
    const accountState = isRecord(checkpoint.accountState)
      ? (checkpoint.accountState as unknown as BridgeStateEnvelope["accountState"])
      : undefined;
    return {
      ...empty,
      discovery,
      ...(accountState ? { accountState } : {}),
      nextCursor:
        typeof checkpoint.nextCursor === "string"
          ? checkpoint.nextCursor
          : null,
      stateVersionCounter,
    };
  }
  if (
    (checkpoint.scope === "active" || checkpoint.scope === "archived") &&
    typeof checkpoint.accountId === "string"
  ) {
    const { candidateQueue: _candidateQueue, ...headerCheckpoint } = checkpoint;
    return {
      ...empty,
      discovery: {
        [checkpoint.scope]: headerCheckpoint as never,
      },
    };
  }
  throw new BoundedWorkerError("protocol_invalid", false, true);
}

function mergeQueueItems(
  state: BridgeStateEnvelope,
  items: readonly Record<string, unknown>[],
): boolean {
  const revisits = new Map(
    (state.revisits ?? []).map((entry) => [entry.conversationId, entry]),
  );
  let complete = true;
  for (const item of items) {
    const candidateKey =
      typeof item.candidateKey === "string" ? item.candidateKey : null;
    const payload = isRecord(item.payload) ? item.payload : item;
    if (item.operation === "remove") {
      const queueKind =
        item.queueKind ??
        payload.queueKind ??
        (candidateKey?.startsWith("revisit:") ? "revisit" : "candidate");
      if (queueKind === "revisit") {
        const conversationId =
          candidateKey?.startsWith("revisit:")
            ? candidateKey.slice("revisit:".length)
            : isRecord(payload.entry) &&
                typeof payload.entry.conversationId === "string"
              ? payload.entry.conversationId
              : null;
        if (conversationId === null) {
          complete = false;
        } else {
          revisits.delete(conversationId);
        }
      } else {
        const conversationId = candidateConversationIdFromKey(candidateKey);
        if (conversationId === null) {
          complete = false;
        } else {
          for (const checkpoint of Object.values(state.discovery)) {
            checkpoint?.candidateQueue &&
              (checkpoint.candidateQueue = checkpoint.candidateQueue.filter(
                (candidate) =>
                  candidate.summary.conversationId !== conversationId,
              ));
          }
        }
      }
      continue;
    }
    const queueKind =
      item.queueKind ??
      payload.queueKind ??
      (candidateKey?.startsWith("revisit:")
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
    const outerScope =
      item.scope === "active" || item.scope === "archived"
        ? item.scope
        : payload.scope === "active" || payload.scope === "archived"
          ? payload.scope
          : null;
    const candidateSummary = isRecord(candidate.summary)
      ? candidate.summary
      : null;
    const candidateConversationId =
      candidateSummary &&
      typeof candidateSummary.conversationId === "string"
        ? candidateSummary.conversationId
        : null;
    const scope =
      outerScope === "archived" ||
      candidate.scope === "archived" ||
      candidateSummary?.isArchived === true
        ? "archived"
        : outerScope === "active" ||
            candidate.scope === "active"
          ? "active"
          : null;
    if (!scope || !candidateSummary || candidateConversationId === null) {
      complete = false;
      continue;
    }
    const checkpoint = state.discovery[scope];
    if (!checkpoint) {
      complete = false;
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
  if (revisits.size > 0) {
    state.revisits = [...revisits.values()];
  } else {
    delete state.revisits;
  }
  return complete;
}

function candidateConversationIdFromKey(
  candidateKey: string | null,
): string | null {
  if (candidateKey === null) {
    return null;
  }
  const parts = candidateKey.split(":");
  return parts.length >= 3 && parts[0] === "candidate"
    ? parts.slice(2).join(":") || null
    : null;
}

function typedReadResultFromWire(
  result: Record<string, unknown>,
  kind: OpaquePageRequest["kind"],
): ReturnType<typeof typedReadResult> {
  const shapeWarnings: string[] = [];
  const rawItems = result.items;
  const rawMessages = result.messages;
  const itemsAreValid =
    Array.isArray(rawItems) && rawItems.every((item) => isRecord(item));
  const messagesAreValid =
    Array.isArray(rawMessages) &&
    rawMessages.every((message) => isRecord(message));
  if (rawItems !== undefined && !itemsAreValid) {
    shapeWarnings.push("history_items_shape_invalid");
  }
  if (rawMessages !== undefined && !messagesAreValid) {
    shapeWarnings.push("history_messages_shape_invalid");
  }
  const rawSource = result.source;
  const source =
    rawSource === undefined ? undefined : ingestContextFromWire(rawSource);
  if (rawSource !== undefined && source === undefined) {
    shapeWarnings.push("history_source_shape_invalid");
  }
  const items = itemsAreValid
    ? rawItems.filter(isRecord)
    : messagesAreValid
      ? rawMessages.filter(isRecord)
      : [];
  const messages = messagesAreValid ? rawMessages.filter(isRecord) : [];
  const detail = isRecord(result.detail) ? result.detail : undefined;
  const summary = isRecord(result.summary) ? result.summary : undefined;
  const normalized = {
    items,
    ...(detail ? { detail } : {}),
    ...(summary ? { summary } : {}),
    ...(messages.length > 0 ? { messages } : {}),
    ...(source ? { source } : {}),
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
    coverage:
      shapeWarnings.length > 0 ||
      (kind === "index" && !itemsAreValid) ||
      (kind === "messages" && !itemsAreValid && !messagesAreValid)
        ? "unrecognized"
        : isCoverage(result.coverage)
          ? result.coverage
          : "unrecognized",
    warnings: Array.isArray(result.warnings)
      ? [
          ...shapeWarnings,
          ...result.warnings.filter(
            (item): item is string => typeof item === "string",
          ),
        ]
      : shapeWarnings,
    ...(typeof result.snapshotId === "string" || result.snapshotId === null
      ? { snapshotId: result.snapshotId }
      : {}),
    ...(typeof result.nextCursor === "string" || result.nextCursor === null
      ? { nextCursor: result.nextCursor }
      : {}),
    ...(typeof result.hasMore === "boolean"
      ? { hasMore: result.hasMore }
      : {}),
    ...(typeof result.truncated === "boolean"
      ? { truncated: result.truncated }
      : {}),
    ...(isRecord(result.coverageDetails)
      ? { coverageDetails: result.coverageDetails }
      : {}),
  } as unknown as Awaited<ReturnType<HistoryReader["listConversations"]>>;
  const typed = typedReadResult(normalized);
  return {
    ...typed,
    ...(detail ? { detail: detail as unknown as ConversationDetailProjection } : {}),
    ...(summary ? { summary: summary as unknown as ConversationSummary } : {}),
    ...(messages.length > 0
      ? { messages: messages as unknown as MessageRecord[] }
      : {}),
    ...(source ? { source } : {}),
    ...(typeof result.snapshotId === "string" || result.snapshotId === null
      ? { snapshotId: result.snapshotId }
      : {}),
    ...(typeof result.nextCursor === "string" || result.nextCursor === null
      ? { nextCursor: result.nextCursor }
      : {}),
    ...(typeof result.hasMore === "boolean"
      ? { hasMore: result.hasMore }
      : {}),
    ...(typeof result.truncated === "boolean"
      ? { truncated: result.truncated }
      : {}),
    ...(isRecord(result.coverageDetails)
      ? { coverageDetails: result.coverageDetails }
      : {}),
  };
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
): HistoryMetadataPage {
  const shapeWarnings: string[] = [];
  const summary = isRecord(result.summary)
    ? (result.summary as unknown as ConversationSummary)
    : null;
  const rawItems = result.items;
  if (rawItems !== undefined && !Array.isArray(rawItems)) {
    shapeWarnings.push("retained_metadata_items_shape_invalid");
  }
  const items = Array.isArray(rawItems)
    ? rawItems.filter(isRecord)
    : [];
  if (Array.isArray(rawItems) && items.length !== rawItems.length) {
    shapeWarnings.push("retained_metadata_items_shape_invalid");
  }
  const firstSummary =
    summary ??
    (items
      .map((item) => (isRecord(item.payload) ? item.payload : item))
      .find(
        (item) => "conversationId" in item && "isArchived" in item,
      ) as unknown as ConversationSummary | undefined);
  const itemConversationIds = items
    .flatMap((item) => {
      const direct = item.conversationId;
      const payload = isRecord(item.payload)
        ? item.payload.conversationId
        : undefined;
      return [
        ...(typeof direct === "string" ? [direct] : []),
        ...(typeof payload === "string" ? [payload] : []),
      ];
    })
    .filter((item, index, all) => all.indexOf(item) === index);
  if (itemConversationIds.some((item) => item !== conversationId)) {
    throw new BoundedWorkerError("protocol_invalid", false, true);
  }
  if (firstSummary && firstSummary.conversationId !== conversationId) {
    throw new BoundedWorkerError("protocol_invalid", false, true);
  }
  const observationMessages = items.flatMap((item) => {
    const payload = isRecord(item.payload) ? item.payload : item;
    return Array.isArray(payload.messages)
      ? payload.messages.filter(isRecord)
      : [];
  });
  const rawMessages = result.messages;
  if (rawMessages !== undefined && !Array.isArray(rawMessages)) {
    shapeWarnings.push("retained_metadata_messages_shape_invalid");
  }
  const directMessages = Array.isArray(rawMessages)
    ? rawMessages.filter(isRecord)
    : [];
  const messages = [
    ...directMessages,
    ...items.flatMap((item) =>
      Array.isArray(item.messages) ? item.messages.filter(isRecord) : [],
    ),
    ...observationMessages,
  ];
  if (
    Array.isArray(rawMessages) &&
    directMessages.length !== rawMessages.length
  ) {
    shapeWarnings.push("retained_metadata_messages_shape_invalid");
  }
  const observationAttempts = items.flatMap((item) => {
    const payload = isRecord(item.payload) ? item.payload : item;
    return Array.isArray(payload.attempts)
      ? payload.attempts.filter(isRecord)
      : [];
  });
  const rawAttempts = result.attempts;
  if (rawAttempts !== undefined && !Array.isArray(rawAttempts)) {
    shapeWarnings.push("retained_metadata_attempts_shape_invalid");
  }
  const directAttempts = Array.isArray(rawAttempts)
    ? rawAttempts.filter(isRecord)
    : [];
  const attempts = dedupeWireAttempts([
    ...directAttempts,
    ...observationAttempts,
  ]);
  if (
    Array.isArray(rawAttempts) &&
    directAttempts.length !== rawAttempts.length
  ) {
    shapeWarnings.push("retained_metadata_attempts_shape_invalid");
  }
  const sources: IngestContext[] = [];
  for (const item of items) {
    if (item.context === undefined) {
      continue;
    }
    const source = ingestContextFromWire(item.context);
    if (source === undefined) {
      shapeWarnings.push("retained_metadata_context_shape_invalid");
      continue;
    }
    sources.push(source);
  }
  const directSource =
    result.source === undefined
      ? undefined
      : ingestContextFromWire(result.source);
  if (result.source !== undefined && directSource === undefined) {
    shapeWarnings.push("retained_metadata_source_shape_invalid");
  }
  const allSources = directSource ? [directSource, ...sources] : sources;
  const source = allSources[0];
  const rawCoverage = result.coverage;
  const coverageDetails: Record<string, unknown> = {
    ...(isRecord(rawCoverage) ? { pageCoverage: rawCoverage } : {}),
    ...(isRecord(result.coverageDetails) ? result.coverageDetails : {}),
  };
  const terminalPage =
    result.hasMore === false &&
    (result.nextCursor === undefined || result.nextCursor === null);
  const coverage =
    rawCoverage === "complete" ||
    rawCoverage === "partial" ||
    rawCoverage === "unknown"
      ? rawCoverage
      : isRecord(rawCoverage)
        ? metadataCoverageFromObject(rawCoverage, terminalPage)
        : undefined;
  const truncated =
    typeof result.truncated === "boolean"
      ? result.truncated
      : isRecord(rawCoverage) && typeof rawCoverage.truncated === "boolean"
        ? rawCoverage.truncated
        : undefined;
  return {
    ...(firstSummary
      ? {
          summary: {
            ...firstSummary,
            conversationId,
          },
        }
      : {}),
    ...(messages.length > 0
      ? { messages: messages as unknown as MessageRecord[] }
      : {}),
    ...(attempts.length > 0
      ? {
          attempts:
          attempts as unknown as import("../ledger/types.js").ReconstructedAttempt[],
        }
      : {}),
    ...(items.length > 0 ? { items } : {}),
    ...(source ? { source } : {}),
    ...(typeof result.schemaVersion === "string"
      ? { schemaVersion: result.schemaVersion }
      : {}),
    ...(Object.keys(coverageDetails).length > 0
      ? { coverageDetails }
      : {}),
    ...(typeof result.snapshotId === "string" || result.snapshotId === null
      ? { snapshotId: result.snapshotId }
      : {}),
    ...(typeof result.nextCursor === "string" || result.nextCursor === null
      ? { nextCursor: result.nextCursor }
      : {}),
    ...(typeof result.hasMore === "boolean"
      ? { hasMore: result.hasMore }
      : {}),
    ...(coverage ? { coverage } : {}),
    ...(truncated !== undefined ? { truncated } : {}),
    warnings: [
      ...shapeWarnings,
      ...(Array.isArray(result.warnings)
        ? result.warnings.filter(
            (item): item is string => typeof item === "string",
          )
        : []),
    ],
  };
}

function reportSnapshotFromWire(
  first: Record<string, unknown>,
  accountId: string,
  attempts: UsageReportSnapshot["attempts"],
  truncated: boolean,
  warnings: readonly string[],
): UsageReportSnapshot {
  if (
    first.snapshotVersion !== 1 ||
    first.collectorAccountId !== accountId ||
    !isRecord(first.coverage)
  ) {
    throw new BoundedWorkerError("protocol_invalid", false, true);
  }
  const coverage = parseUsageCoverage(first.coverage);
  if (truncated) {
    coverage.overall = coverage.overall === "complete" ? "partial" : coverage.overall;
    coverage.history =
      coverage.history === "complete" ? "partial" : coverage.history;
    coverage.gaps = [
      ...new Set([...coverage.gaps, "report_snapshot_truncated", ...warnings]),
    ];
  }
  const historyCoverage = isRecord(first.historyCoverage)
    ? (first.historyCoverage as unknown as UsageReportSnapshot["historyCoverage"])
    : undefined;
  return {
    snapshotVersion: 1,
    collectorAccountId: accountId,
    attempts,
    coverage,
    ...(historyCoverage ? { historyCoverage } : {}),
  };
}

function parseUsageCoverage(
  value: Record<string, unknown>,
): UsageReportSnapshot["coverage"] {
  const levels = ["history", "overall", "projects", "branches"] as const;
  if (
    levels.some(
      (key) =>
        value[key] !== "complete" &&
        value[key] !== "partial" &&
        value[key] !== "unknown",
    ) ||
    !Array.isArray(value.gaps) ||
    value.gaps.some((gap) => typeof gap !== "string")
  ) {
    throw new BoundedWorkerError("protocol_invalid", false, true);
  }
  return {
    history: value.history as UsageReportSnapshot["coverage"]["history"],
    overall: value.overall as UsageReportSnapshot["coverage"]["overall"],
    projects: value.projects as UsageReportSnapshot["coverage"]["projects"],
    branches: value.branches as UsageReportSnapshot["coverage"]["branches"],
    gaps: [...value.gaps],
  };
}

function ingestContextFromWire(value: unknown): IngestContext | undefined {
  if (!isRecord(value)) {
    return undefined;
  }
  const required = [
    "runId",
    "observedAt",
    "sourceKind",
    "sourceId",
    "schemaVersion",
  ] as const;
  if (
    required.some(
      (field) =>
        typeof value[field] !== "string" ||
        value[field].trim() === "",
    )
  ) {
    return undefined;
  }
  if (
    !isValidInstant(value.observedAt) ||
    value.provenance !== undefined && !isRecord(value.provenance)
  ) {
    return undefined;
  }
  return {
    runId: value.runId as string,
    observedAt: value.observedAt as string,
    sourceKind: value.sourceKind as string,
    sourceId: value.sourceId as string,
    schemaVersion: value.schemaVersion as string,
    ...(value.provenance === undefined
      ? {}
      : { provenance: value.provenance as Record<string, unknown> }),
  };
}

function metadataCoverageFromObject(
  value: Record<string, unknown>,
  terminalPage = false,
): "complete" | "partial" | "unknown" {
  if (
    terminalPage &&
    value.partial === false &&
    value.truncated === false
  ) {
    return "complete";
  }
  const candidates = [
    value.status,
    value.coverage,
    value.overall,
    value.state,
  ];
  for (const candidate of candidates) {
    if (
      candidate === "complete" ||
      candidate === "partial" ||
      candidate === "unknown"
    ) {
      return candidate;
    }
  }
  return value.truncated === true || value.partial === true
    ? "partial"
    : "unknown";
}

function isValidInstant(value: unknown): value is string {
  return typeof value === "string" && Number.isFinite(new Date(value).getTime());
}

function throwIfDomainBlocked(
  result: Record<string, unknown>,
  path: string,
): void {
  if (result.status !== "blocked" && result.status !== "unavailable") {
    return;
  }
  if (result.reason === "authentication") {
    throw new AuthenticationRequiredError(
      `native Chat history authentication is required for ${path}`,
      { status: 401, path },
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
      `native Chat history is rate limited for ${path}`,
      {
        ...(retryAfterMs === null
          ? {}
          : {
              retryAfter: new Date(
                Date.now() + retryAfterMs,
              ).toISOString(),
            }),
        path,
      },
    );
  }
  throw new BoundedWorkerError(
    "history_contract_unavailable",
    true,
    true,
  );
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
  reportIncomplete = false,
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
        : result.status === "complete" && !reportIncomplete
          ? "success"
          : "failure",
    retryAfterMs,
  };
}

function defaultReportRequest(result: {
  accountId: string;
  range: { start: string; end: string };
}): Record<string, unknown> {
  const start = new Date(result.range.start).getTime();
  const end = new Date(result.range.end).getTime();
  if (
    !Number.isFinite(start) ||
    !Number.isFinite(end) ||
    end <= start ||
    !Number.isSafeInteger(end - start)
  ) {
    throw new BoundedWorkerError("protocol_invalid", false, true);
  }
  return {
    account: result.accountId,
    asOf: new Date(end).toISOString(),
    modelDimension: "resolved",
    elapsedLookbackMs: end - start,
  };
}

function completionForError(
  error: unknown,
  now: number,
): CompleteTriggerRequest {
  return {
    outcome:
      error instanceof AuthenticationRequiredError
        ? "authentication"
        : "failure",
    retryAfterMs:
      error instanceof RateLimitedError
        ? retryAfterMsFromWire(error.retryAfter, now)
        : null,
  };
}

function retryAfterMsFromWire(
  value: string | null,
  now: number,
): number | null {
  if (value === null || value.trim() === "") {
    return null;
  }
  const seconds = Number(value);
  if (Number.isFinite(seconds) && seconds >= 0) {
    return Math.min(Number.MAX_SAFE_INTEGER, Math.floor(seconds * 1_000));
  }
  const instant = new Date(value).getTime();
  return Number.isFinite(instant) ? Math.max(0, instant - now) : null;
}

function scopeFromWire(
  value: Record<string, unknown>,
): LedgerScope {
  if (
    typeof value.collectorAccountId !== "string" ||
    value.collectorAccountId.trim() === "" ||
    typeof value.provider !== "string" ||
    value.provider.trim() === "" ||
    !isSurface(value.surface)
  ) {
    throw new BoundedWorkerError("history_contract_unavailable", true, true);
  }
  const nullableFields = [
    "providerUserId",
    "workspaceId",
    "quotaOwnerId",
  ] as const;
  if (
    nullableFields.some(
      (field) =>
        value[field] !== null && typeof value[field] !== "string",
    )
  ) {
    throw new BoundedWorkerError("history_contract_unavailable", true, true);
  }
  return {
    collectorAccountId: value.collectorAccountId,
    provider: value.provider,
    providerUserId: value.providerUserId as string | null,
    workspaceId: value.workspaceId as string | null,
    quotaOwnerId: value.quotaOwnerId as string | null,
    surface: value.surface,
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
    value.control === "cancelRun"
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

function boundedValue(value: unknown, maximum: number, label: string): number {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed <= 0 || parsed > maximum) {
    throw new BoundedWorkerError("bounds_exceeded", true, true);
  }
  return parsed;
}

function requiredStateVersion(value: unknown): number {
  if (!Number.isSafeInteger(value) || (value as number) < 0) {
    throw new BoundedWorkerError("protocol_invalid", false, true);
  }
  return value as number;
}

function isWorkerErrorCode(value: unknown): value is WorkerErrorCode {
  return (
    value === "protocol_invalid" ||
    value === "fence_invalid" ||
    value === "state_conflict" ||
    value === "history_contract_unavailable" ||
    value === "history_reader_failed" ||
    value === "bounds_exceeded" ||
    value === "conversation_not_found" ||
    value === "operation_unsupported" ||
    value === "cancelled"
  );
}

function isSurface(value: unknown): value is IdentityRecord["surface"] {
  return (
    value === "chat" ||
    value === "work" ||
    value === "codex" ||
    value === "deep_research" ||
    value === "agent_mode" ||
    value === "voice" ||
    value === "image_generation" ||
    value === "unknown"
  );
}

function isAuthState(value: unknown): value is IdentityRecord["authState"] {
  return (
    value === "unconfigured" ||
    value === "ready" ||
    value === "auth_required" ||
    value === "identity_mismatch" ||
    value === "browser_unavailable" ||
    value === "schema_unavailable" ||
    value === "paused"
  );
}

function isAccountBlockingError(error: unknown): boolean {
  return (
    error instanceof AuthenticationRequiredError ||
    error instanceof RateLimitedError
  );
}

function monotonicNow(): number {
  return typeof performance !== "undefined" ? performance.now() : Date.now();
}

function errorCode(error: unknown): string {
  if (error instanceof BoundedWorkerError) {
    return error.code;
  }
  if (error instanceof AuthenticationRequiredError) {
    return "auth_required";
  }
  if (error instanceof RateLimitedError) {
    return "rate_limited";
  }
  if (error instanceof Error && error.name) {
    return error.name.toLowerCase().replace(/[^a-z0-9]+/g, "_");
  }
  return "worker_error";
}

function sameValue(left: unknown, right: unknown): boolean {
  return (
    JSON.stringify(canonicalize(left)) === JSON.stringify(canonicalize(right))
  );
}

function canonicalize(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(canonicalize);
  }
  if (value === null || typeof value !== "object") {
    return value;
  }
  return Object.fromEntries(
    Object.entries(value as Record<string, unknown>)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, item]) => [key, canonicalize(item)]),
  );
}

function dedupeWireAttempts(
  attempts: readonly Record<string, unknown>[],
): Record<string, unknown>[] {
  const byId = new Map<string, Record<string, unknown>>();
  const withoutId: Record<string, unknown>[] = [];
  for (const attempt of attempts) {
    const attemptId = attempt.attemptId;
    if (typeof attemptId !== "string" || attemptId.trim() === "") {
      withoutId.push(attempt);
      continue;
    }
    if (!byId.has(attemptId)) {
      byId.set(attemptId, attempt);
    }
  }
  return [...byId.values(), ...withoutId];
}

if (process.argv.includes("--stdio-v1")) {
  runStdioWorker().catch((error: unknown) => {
    process.exitCode = error instanceof BoundedWorkerError ? 1 : 1;
  });
}
