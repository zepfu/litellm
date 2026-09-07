import { createHash } from "node:crypto";

import {
  AuthenticationRequiredError,
  CapabilityError,
  HttpStatusError,
  LegacyFallbackNotApprovedError,
  RateLimitedError,
} from "../adapters/chatgpt/adapter.js";
import type {
  AcquiredConversation,
  HistoryAccountState,
  DiscoveryCandidate,
  DiscoveryCheckpoint,
  GenerationCompletionState,
  HistoryCollectionOptions,
  HistoryCollectionRequest,
  HistoryCollectionResult,
  HistoryCoverageResult,
  HistoryRange,
  HistoryDiscoveryPageCommit,
  HistoryCheckpointStore,
  HistoryPageCommit,
  HistoryReader,
  HistoryScope,
  OlderHistoryAuditCoverage,
  OlderHistoryAuditState,
  OutstandingGenerationState,
  RevisitPageIssue,
  RevisitEntry,
  RevisitReason,
  ScopeCoverageResult,
} from "../contracts/history.js";
import {
  DEFAULT_BACKFILL_DAYS,
  DEFAULT_INDEX_PAGE_SIZE,
  DEFAULT_MAX_INDEX_PAGES,
  DEFAULT_MAX_MESSAGE_PAGES_PER_CONVERSATION,
  DEFAULT_MAX_OLDER_HISTORY_AUDIT_PAGES,
  DEFAULT_OVERLAP_MS,
  HISTORY_STATE_VERSION,
  OUTSTANDING_GENERATION_REVISIT_BASE_DELAY_MS,
  OUTSTANDING_GENERATION_REVISIT_MAX_DELAY_MS,
  OUTSTANDING_GENERATION_TIMEOUT_MS,
} from "../contracts/history.js";
import type {
  AdaptedPage,
  ConversationDetailProjection,
  ConversationSummary,
  IdentityRecord,
  MessageRecord,
  PaginationState,
} from "../contracts/records.js";
import { resolveRequestedRange } from "./range.js";

interface Candidate {
  summary: ConversationSummary;
  scopes: Set<HistoryScope>;
  missingUpdateTime: boolean;
}

interface ScopeDiscoveryResult {
  scope: HistoryScope;
  summaries: ConversationSummary[];
  coverage: ScopeCoverageResult;
  warnings: string[];
  auditState: OlderHistoryAuditState;
  auditCandidateIds: string[];
  auditScanComplete: boolean;
}

interface DetailAcquisition {
  detail: ConversationDetailProjection | null;
  messages: MessageRecord[];
  revisit: RevisitEntry | null;
  coverage: AcquiredConversation["coverage"];
  warnings: string[];
  detailPagesFetched: number;
}

interface RevisitOptions {
  continuation?: string | null;
  continuationRevision?: string | null;
  malformedPage?: RevisitPageIssue | null;
  outstandingGeneration?: OutstandingGenerationState | null;
  nextEligibleAt?: string;
  pagesFetched?: number;
  incrementAttempt?: boolean;
}

interface AuditEvidence {
  committedCandidateIds: Set<string>;
}

const TERMINAL_GENERATION_STATUSES = new Set([
  "completed",
  "done",
  "finished",
  "finished_successfully",
  "success",
]);

const NONTERMINAL_GENERATION_STATUSES = new Set([
  "generating",
  "in_progress",
  "incomplete",
  "pending",
  "processing",
  "streaming",
]);

export class HistoryCollector {
  private readonly defaultBackfillDays: number;
  private readonly overlapMs: number;
  private readonly indexPageSize: number;
  private readonly maxIndexPagesPerScope: number;
  private readonly maxMessagePagesPerConversation: number;
  private readonly legacyFallbackApproved: boolean;

  constructor(
    private readonly reader: HistoryReader,
    private readonly options: HistoryCollectionOptions,
  ) {
    this.defaultBackfillDays =
      options.defaultBackfillDays ?? DEFAULT_BACKFILL_DAYS;
    this.overlapMs = options.overlapMs ?? DEFAULT_OVERLAP_MS;
    this.indexPageSize = options.indexPageSize ?? DEFAULT_INDEX_PAGE_SIZE;
    this.maxIndexPagesPerScope =
      options.maxIndexPagesPerScope ?? DEFAULT_MAX_INDEX_PAGES;
    this.maxMessagePagesPerConversation =
      options.maxMessagePagesPerConversation ??
      DEFAULT_MAX_MESSAGE_PAGES_PER_CONVERSATION;
    this.legacyFallbackApproved =
      options.legacyFallbackApproved ??
      reader.capabilities.legacySupport === "fallback_on_404_405";
    validatePositiveInteger(this.indexPageSize, "index page size");
    validatePositiveInteger(
      this.maxIndexPagesPerScope,
      "maximum index pages per scope",
    );
    validatePositiveInteger(
      this.maxMessagePagesPerConversation,
      "maximum message pages per conversation",
    );
    if (!Number.isFinite(this.overlapMs) || this.overlapMs < 0) {
      throw new Error("overlap duration must be a non-negative finite number");
    }
  }

  async collect(
    request: HistoryCollectionRequest,
  ): Promise<HistoryCollectionResult> {
    const now = request.now ?? this.options.clock?.now() ?? new Date();
    const initialScanStartedAt = now.toISOString();
    const rangeOptions: {
      mode: HistoryCollectionRequest["mode"];
      now: Date;
      defaultBackfillDays: number;
      range?: HistoryRange;
    } = {
      mode: request.mode,
      now,
      defaultBackfillDays: this.defaultBackfillDays,
    };
    if (request.range) {
      rangeOptions.range = request.range;
    }
    let requestedRange = resolveRequestedRange(rangeOptions);
    validatePositiveInteger(
      request.indexPageSize ?? this.indexPageSize,
      "index page size",
    );
    validatePositiveInteger(
      request.maxIndexPagesPerScope ?? this.maxIndexPagesPerScope,
      "maximum index pages per scope",
    );
    validatePositiveInteger(
      request.maxMessagePagesPerConversation ??
        this.maxMessagePagesPerConversation,
      "maximum message pages per conversation",
    );
    if (request.olderHistoryAudit?.maxPages !== undefined) {
      validatePositiveInteger(
        request.olderHistoryAudit.maxPages,
        "older-history audit page budget",
      );
    }
    const frozenAcquisition = frozenImplicitAcquisition(
      this.options.store,
      request,
    );
    const scanStartedAt =
      frozenAcquisition?.scanStartedAt ?? initialScanStartedAt;
    if (frozenAcquisition) {
      requestedRange = frozenAcquisition.range;
    }
    const persistedAccountState = this.options.store.loadAccountState();
    if (isActiveCooldown(persistedAccountState, now)) {
      return blockedResult(
        this.options.accountId,
        request.mode,
        requestedRange,
        scanStartedAt,
        pausedIdentity(null, persistedAccountState),
        accountPauseWarning(persistedAccountState),
        persistedAccountState,
      );
    }
    if (
      persistedAccountState.status === "paused" &&
      persistedAccountState.reason === "cooldown"
    ) {
      this.options.store.saveAccountState(readyAccountState());
    } else if (
      persistedAccountState.status === "paused" &&
      persistedAccountState.reason !== "cooldown"
    ) {
      // A persisted non-cooldown pause requires explicit verified recovery
      // (rerun bootstrap); read-only identity inspection cannot clear it.
      return blockedResult(
        this.options.accountId,
        request.mode,
        requestedRange,
        scanStartedAt,
        pausedIdentity(null, persistedAccountState),
        accountPauseWarning(persistedAccountState),
        persistedAccountState,
      );
    }

    let identity = await this.readIdentity(now);
    if (identity.authState === "auth_required") {
      const accountState = this.pauseForError(
        new AuthenticationRequiredError(
          "authentication required during session inspection",
          { status: 401, path: "/api/auth/session" },
        ),
        now,
      );
      identity = pausedIdentity(identity, accountState);
      return blockedResult(
        this.options.accountId,
        request.mode,
        requestedRange,
        scanStartedAt,
        identity,
        accountPauseWarning(accountState),
        accountState,
      );
    }
    if (
      identity.authState === "paused" &&
      this.options.store.loadAccountState().status === "paused"
    ) {
      const accountState = this.options.store.loadAccountState();
      return blockedResult(
        this.options.accountId,
        request.mode,
        requestedRange,
        scanStartedAt,
        pausedIdentity(identity, accountState),
        accountPauseWarning(accountState),
        accountState,
      );
    }

    if (identity.authState !== "ready" || identity.surface !== "chat") {
      return blockedResult(
        this.options.accountId,
        request.mode,
        requestedRange,
        scanStartedAt,
        identity,
        "history collection requires a verified ready Chat identity",
        this.options.store.loadAccountState(),
      );
    }

    const candidates = new Map<string, Candidate>();
    const discoveries: ScopeDiscoveryResult[] = [];
    const scopeResults: ScopeCoverageResult[] = [];
    const warnings: string[] = [];
    let pagesFetched = 0;

    for (const scope of ["active", "archived"] as const) {
      if (this.options.store.loadAccountState().status === "paused") {
        break;
      }
      const discovery = await this.discoverScopeDurable(
        scope,
        request,
        requestedRange,
        now,
        scanStartedAt,
        identity,
      );
      discoveries.push(discovery);
      scopeResults.push(discovery.coverage);
      pagesFetched += discovery.coverage.pagesFetched;
      warnings.push(...discovery.warnings);
      for (const summary of discovery.summaries) {
        const existing = candidates.get(summary.conversationId);
        if (existing) {
          existing.scopes.add(scope);
          if (isMoreRecent(summary.updatedAt, existing.summary.updatedAt)) {
            existing.summary = summary;
          }
          existing.missingUpdateTime ||= summary.updatedAt === null;
        } else {
          candidates.set(summary.conversationId, {
            summary,
            scopes: new Set([scope]),
            missingUpdateTime: summary.updatedAt === null,
          });
        }
      }
      if (this.options.store.loadAccountState().status === "paused") {
        break;
      }
    }

    const scanTime = now.getTime();
    for (const revisit of this.options.store.listRevisits()) {
      if (new Date(revisit.nextEligibleAt).getTime() > scanTime) {
        continue;
      }
      const existing = candidates.get(revisit.conversationId);
      if (existing) {
        for (const scope of revisit.scopes) {
          existing.scopes.add(scope);
        }
        continue;
      }
      candidates.set(revisit.conversationId, {
        summary: revisitSummary(revisit),
        scopes: new Set(revisit.scopes),
        missingUpdateTime: true,
      });
    }

    const auditEvidence = new Map<HistoryScope, AuditEvidence>(
      discoveries.map((discovery) => [
        discovery.scope,
        { committedCandidateIds: new Set<string>() },
      ]),
    );
    const conversations: AcquiredConversation[] = [];
    let detailPagesFetched = 0;
    const revisitByConversation = new Map(
      this.options.store
        .listRevisits()
        .map((revisit) => [revisit.conversationId, revisit]),
    );
    for (const candidate of candidates.values()) {
      if (this.options.store.loadAccountState().status === "paused") {
        break;
      }
      const revisit = revisitByConversation.get(candidate.summary.conversationId);
      if (
        revisit &&
        new Date(revisit.nextEligibleAt).getTime() > scanTime
      ) {
        continue;
      }
      const acquisition = await this.acquireConversation(
        candidate,
        now,
        request,
        identity,
        scanStartedAt,
      );
      detailPagesFetched += acquisition.detailPagesFetched;
      for (const discovery of discoveries) {
        if (
          discovery.auditCandidateIds.includes(
            candidate.summary.conversationId,
          ) &&
          acquisition.detail !== null &&
          acquisition.coverage === "complete" &&
          acquisition.revisit === null
        ) {
          auditEvidence
            .get(discovery.scope)
            ?.committedCandidateIds.add(candidate.summary.conversationId);
        }
      }
      conversations.push({
        summary: candidate.summary,
        scopes: [...candidate.scopes].sort(),
        detail: acquisition.detail,
        messages: acquisition.messages,
        coverage: acquisition.coverage,
        revisit: acquisition.revisit,
        warnings: [
          ...(candidate.missingUpdateTime
            ? ["conversation_missing_update_time"]
            : []),
          ...acquisition.warnings,
        ],
      });
      if (this.options.store.loadAccountState().status === "paused") {
        break;
      }
    }

    this.finalizeOlderHistoryAudits(
      discoveries,
      auditEvidence,
      scanStartedAt,
      now,
    );
    const revisits = this.options.store.listRevisits();
    const accountState = this.options.store.loadAccountState();
    if (accountState.status === "paused") {
      identity = pausedIdentity(identity, accountState);
      warnings.push(accountPauseWarning(accountState));
    }
    const coverage = this.buildCoverage(scopeResults, conversations, warnings);
    const status =
      accountState.status !== "paused" &&
      coverage.overall === "complete" &&
      revisits.length === 0
        ? "complete"
        : "partial";

    return {
      accountId: this.options.accountId,
      mode: request.mode,
      range: requestedRange,
      scanStartedAt,
      status,
      accountState,
      identity,
      scopes: scopeResults,
      conversations,
      revisits,
      coverage,
      pagesFetched,
      detailPagesFetched,
      warnings: [...new Set(warnings)],
    };
  }

  private async readIdentity(now: Date): Promise<IdentityRecord> {
    try {
      return await this.reader.inspectSessionIdentity();
    } catch (error) {
      if (isAccountStopError(error)) {
        const accountState = this.pauseForError(error, now);
        return pausedIdentity(null, accountState, errorCode(error));
      }
      return {
        providerUserId: null,
        workspaceId: null,
        quotaOwnerId: null,
        surface: "unknown",
        authState:
          error instanceof AuthenticationRequiredError
            ? "auth_required"
            : "paused",
        identityErrors: [errorCode(error)],
      };
    }
  }

  private pauseForError(
    error: unknown,
    now: Date,
  ): HistoryAccountState {
    // Numeric Retry-After is relative to response receipt, not run start; a
    // run captured 10 minutes before the 429 must not shorten the deadline.
    const receivedAt =
      error instanceof HttpStatusError && error.retryAfter !== null
        ? this.responseReceivedAt(now)
        : now;
    const accountState: HistoryAccountState = {
      status: "paused",
      reason: isRateLimitedError(error) ? "cooldown" : "authentication",
      pausedAt: receivedAt.toISOString(),
      cooldownUntil:
        error instanceof HttpStatusError
          ? retryAfterDeadline(error.retryAfter, receivedAt)
          : null,
      lastError: errorCode(error),
    };
    this.options.store.saveAccountState(accountState);
    return accountState;
  }

  private responseReceivedAt(fallback: Date): Date {
    const wallClock = new Date();
    let sampled = wallClock;
    try {
      if (this.options.clock) {
        sampled = this.options.clock.now();
      }
    } catch {
      sampled = wallClock;
    }
    const fallbackMs = fallback.getTime();
    const sampledMs = sampled.getTime();
    const candidateMs = Number.isFinite(sampledMs)
      ? sampledMs
      : wallClock.getTime();
    if (!Number.isFinite(candidateMs)) {
      return fallback;
    }
    return new Date(
      Number.isFinite(fallbackMs)
        ? Math.max(candidateMs, fallbackMs)
        : candidateMs,
    );
  }

  private async discoverScopeDurable(
    scope: HistoryScope,
    request: HistoryCollectionRequest,
    requestedRange: HistoryRange,
    now: Date,
    scanStartedAt: string,
    identity: IdentityRecord,
  ): Promise<ScopeDiscoveryResult> {
    const previous = this.options.store.loadDiscovery(scope);
    const explicitRange = request.range !== undefined;
    const calculatedCandidateCutoff = this.candidateCutoff(
      request.mode,
      requestedRange,
      previous,
      explicitRange,
      request.overlapMs ?? this.overlapMs,
    );
    const reuseFrozen = shouldReuseFrozenAcquisition(
      previous,
      request,
      requestedRange,
      calculatedCandidateCutoff,
    );
    const acquisitionRange = reuseFrozen ? previous!.range : requestedRange;
    const candidateCutoff = reuseFrozen
      ? previous!.candidateCutoff
      : calculatedCandidateCutoff;
    const acquisitionStartAt =
      reuseFrozen && previous?.scanStartedAt
        ? previous.scanStartedAt
        : scanStartedAt;
    const candidateUpperBound =
      request.mode === "incremental" && request.range === undefined
        ? acquisitionRange.end
        : null;
    const warnings: string[] = [];
    const candidateQueue = normalizeCandidateQueue(previous?.candidateQueue);
    const candidateById = new Map<string, DiscoveryCandidate>();
    for (const candidate of candidateQueue) {
      candidateById.set(candidate.summary.conversationId, candidate);
    }
    const seenOffsets = new Set<number>();
    const priorAudit = normalizeOlderHistoryAudit(previous?.olderHistoryAudit);
    const pageBudget =
      request.maxIndexPagesPerScope ?? this.maxIndexPagesPerScope;
    const pageSize = request.indexPageSize ?? this.indexPageSize;
    const canResumeContinuation =
      reuseFrozen &&
      previous?.continuation !== null &&
      previous?.continuation !== undefined &&
      Number.isInteger(previous.continuation) &&
      previous.continuation >= 0 &&
      previous.candidateQueue !== undefined &&
      previous.headFingerprint !== undefined;
    let offset = canResumeContinuation ? previous!.continuation! : 0;
    let pagesFetched = 0;
    let status: ScopeDiscoveryResult["coverage"]["status"] = "in_progress";
    let continuation: number | null = offset;
    let paginationState: PaginationState = "unknown";
    let leadingPage: AdaptedPage<ConversationSummary> | null = null;
    let headFingerprint =
      canResumeContinuation ? previous?.headFingerprint ?? null : null;
    let restartUsed = false;
    let pendingPage: {
      offset: number;
      page: AdaptedPage<ConversationSummary>;
    } | null = null;
    let stopIndexScan = false;
    const restartAtZero = (): void => {
      restartUsed = true;
      offset = 0;
      continuation = null;
      paginationState = "unknown";
      leadingPage = null;
      headFingerprint = null;
      pendingPage = null;
      seenOffsets.clear();
    };

    const enqueueCandidate = (summary: ConversationSummary): void => {
      const existing = candidateById.get(summary.conversationId);
      if (existing) {
        if (isMoreRecent(summary.updatedAt, existing.summary.updatedAt)) {
          existing.summary = summary;
        }
        existing.missingUpdateTime ||= summary.updatedAt === null;
        return;
      }
      const candidate: DiscoveryCandidate = {
        summary,
        missingUpdateTime: summary.updatedAt === null,
      };
      candidateQueue.push(candidate);
      candidateById.set(summary.conversationId, candidate);
    };

    const enqueuePageCandidates = (
      items: ConversationSummary[],
      force = false,
    ): void => {
      for (const summary of items) {
        if (summary.updatedAt === null) {
          warnings.push("conversation_missing_update_time");
        } else if (!isValidInstant(summary.updatedAt)) {
          warnings.push("conversation_invalid_update_time");
        }
        if (
          force ||
          isCandidateSummary(
            summary,
            candidateCutoff,
            candidateUpperBound,
          )
        ) {
          enqueueCandidate(summary);
        }
      }
    };

    const saveCheckpoint = async (
      checkpointStatus: DiscoveryCheckpoint["status"],
      checkpointPaginationState: PaginationState,
      checkpointContinuation: number | null,
      watermark = previous?.lastCompleteDiscoveryStartedAt ?? null,
      auditState = priorAudit,
    ): Promise<void> => {
      const checkpoint = checkpointFor(
        this.options.accountId,
        scope,
        request.mode,
        acquisitionRange,
        candidateCutoff,
        acquisitionStartAt,
        checkpointContinuation,
        pagesFetched,
        pageBudget,
        watermark,
        checkpointStatus,
        checkpointPaginationState,
        warnings,
        now.toISOString(),
        auditState,
        candidateQueue,
        headFingerprint,
      );
      this.options.store.saveDiscovery(checkpoint);
      await this.commitDiscoveryCheckpoint(
        checkpoint,
        identity,
        acquisitionStartAt,
      );
    };

    await saveCheckpoint("in_progress", "unknown", offset);

    if (canResumeContinuation && offset > 0) {
      try {
        const headPage = await this.reader.listConversations({
          archived: scope === "archived",
          offset: 0,
          limit: pageSize,
          order: "updated",
        });
        pagesFetched += 1;
        const currentHeadFingerprint = fingerprintIndexPage(headPage);
        const resumeHeadIsValid =
          currentHeadFingerprint === previous!.headFingerprint &&
          headPage.coverage !== "unrecognized" &&
          headPage.warnings.length === 0 &&
          headPage.paginationState === "continuation" &&
          headPage.continuation === previous!.continuation;
        if (resumeHeadIsValid) {
          leadingPage = headPage;
          headFingerprint = currentHeadFingerprint;
          enqueuePageCandidates(headPage.items);
          seenOffsets.add(0);
          continuation = previous!.continuation;
          paginationState = "continuation";
          if (pagesFetched >= pageBudget) {
            warnings.push("index_page_budget_exhausted");
            status = "partial";
            paginationState = "budget_exhausted";
            await saveCheckpoint(status, paginationState, continuation);
            stopIndexScan = true;
          } else {
            offset = previous!.continuation!;
          }
        } else {
          warnings.push(
            currentHeadFingerprint === previous!.headFingerprint
              ? "saved_head_not_validated_for_resume"
              : "discovery_head_changed_restart",
          );
          restartAtZero();
          pendingPage = { offset: 0, page: headPage };
        }
      } catch (error) {
        warnings.push(`index_head_${errorCode(error)}`);
        if (isAccountStopError(error)) {
          const accountState = this.pauseForError(error, now);
          warnings.push(accountPauseWarning(accountState));
        }
        status = "partial";
        paginationState = "unknown";
        continuation = null;
        await saveCheckpoint(status, paginationState, continuation);
        stopIndexScan = true;
      }
    }

    while (!stopIndexScan && pagesFetched < pageBudget) {
      const pageOffset = pendingPage?.offset ?? offset;
      if (seenOffsets.has(pageOffset)) {
        warnings.push("repeated_index_offset");
        status = "partial";
        paginationState = "repeated_cursor";
        continuation = null;
        await saveCheckpoint(status, paginationState, continuation);
        if (!restartUsed && pageOffset > 0) {
          restartAtZero();
          status = "in_progress";
          continue;
        }
        break;
      }
      seenOffsets.add(pageOffset);

      let page: AdaptedPage<ConversationSummary>;
      if (pendingPage) {
        page = pendingPage.page;
        pendingPage = null;
      } else {
        try {
          page = await this.reader.listConversations({
            archived: scope === "archived",
            offset: pageOffset,
            limit: pageSize,
            order: "updated",
          });
        } catch (error) {
          warnings.push(`index_${errorCode(error)}`);
          if (isAccountStopError(error)) {
            const accountState = this.pauseForError(error, now);
            warnings.push(accountPauseWarning(accountState));
          }
          status = "partial";
          paginationState = "unknown";
          continuation = pageOffset;
          await saveCheckpoint(status, paginationState, continuation);
          break;
        }
      }

      pagesFetched += 1;
      paginationState = page.paginationState;
      warnings.push(...page.warnings);
      if (page.coverage === "unrecognized") {
        warnings.push("index_unrecognized_page");
      }
      if (pageOffset === 0 && leadingPage === null) {
        leadingPage = page;
        headFingerprint = fingerprintIndexPage(page);
      }
      enqueuePageCandidates(page.items);
      if (
        page.paginationState === "unknown" ||
        page.paginationState === "contradictory" ||
        page.paginationState === "repeated_cursor"
      ) {
        status = "partial";
        continuation = null;
        warnings.push(`index_${page.paginationState}`);
        await saveCheckpoint(status, paginationState, continuation);
        if (
          !restartUsed &&
          pageOffset > 0 &&
          (page.paginationState === "contradictory" ||
            page.paginationState === "repeated_cursor")
        ) {
          restartAtZero();
          status = "in_progress";
          continue;
        }
        break;
      }

      if (page.exhausted && page.paginationState === "complete") {
        if (page.coverage !== "validated_page") {
          warnings.push("index_unvalidated_terminal_page");
        }
        status = "complete";
        continuation = null;
        await saveCheckpoint(status, paginationState, continuation);
        break;
      }

      if (
        page.paginationState !== "continuation" ||
        typeof page.continuation !== "number" ||
        page.continuation <= pageOffset
      ) {
        status = "partial";
        continuation = null;
        paginationState =
          page.paginationState === "continuation"
            ? "repeated_cursor"
            : "unknown";
        warnings.push("nonadvancing_index_continuation");
        await saveCheckpoint(status, paginationState, continuation);
        if (!restartUsed && pageOffset > 0) {
          restartAtZero();
          status = "in_progress";
          continue;
        }
        break;
      }

      continuation = page.continuation;
      if (pagesFetched >= pageBudget) {
        status = "partial";
        paginationState = "budget_exhausted";
        warnings.push("index_page_budget_exhausted");
        await saveCheckpoint(status, paginationState, continuation);
        break;
      }

      offset = page.continuation;
      await saveCheckpoint("in_progress", paginationState, continuation);
    }

    if (status === "complete") {
      let reread: AdaptedPage<ConversationSummary>;
      try {
        reread = await this.reader.listConversations({
          archived: scope === "archived",
          offset: 0,
          limit: pageSize,
          order: "updated",
        });
        pagesFetched += 1;
        warnings.push(...reread.warnings);
        if (reread.coverage === "unrecognized") {
          warnings.push("index_unrecognized_page");
        }
        if (leadingPage === null || indexPagesDiffer(leadingPage, reread)) {
          status = "partial";
          paginationState = "continuation";
          continuation = null;
          warnings.push(
            leadingPage === null
              ? "leading_index_baseline_unavailable"
              : "leading_index_changed_during_scan",
          );
          headFingerprint = fingerprintIndexPage(reread);
          enqueuePageCandidates(reread.items);
          await saveCheckpoint(status, paginationState, continuation);
        }
      } catch (error) {
        if (isAccountStopError(error)) {
          const accountState = this.pauseForError(error, now);
          warnings.push(accountPauseWarning(accountState));
        }
        status = "partial";
        paginationState = "unknown";
        continuation = null;
        warnings.push(`leading_index_reread_${errorCode(error)}`);
        await saveCheckpoint(status, paginationState, continuation);
      }
    }

    if (status === "in_progress") {
      status = "partial";
      paginationState = "budget_exhausted";
      warnings.push("index_page_budget_exhausted");
      await saveCheckpoint(status, paginationState, continuation);
    }

    let auditState = priorAudit;
    let auditCandidateIds: string[] = [];
    let auditScanComplete = false;
    if (
      request.olderHistoryAudit?.enabled &&
      this.options.store.loadAccountState().status !== "paused"
    ) {
      const audit = await this.auditOlderHistory(
        scope,
        request,
        acquisitionRange,
        now,
        acquisitionStartAt,
        priorAudit,
      );
      auditState = audit.state;
      auditCandidateIds = audit.selectedConversationIds;
      auditScanComplete = audit.scanComplete;
      warnings.push(...audit.warnings);
      enqueuePageCandidates(audit.summaries, true);
    } else {
      auditState = {
        ...priorAudit,
        enabled: false,
        status: "disabled",
      };
    }

    const cleanImplicitDiscovery =
      request.mode === "incremental" &&
      !explicitRange &&
      status === "complete" &&
      warnings.length === 0;
    await saveCheckpoint(
      status,
      paginationState,
      continuation,
      cleanImplicitDiscovery
        ? acquisitionStartAt
        : previous?.lastCompleteDiscoveryStartedAt ?? null,
      auditState,
    );

    const coverage =
      pagesFetched === 0
        ? "unknown"
        : status === "complete" && warnings.length === 0
          ? "complete"
          : "partial";
    return {
      scope,
      summaries: candidateQueue.map((candidate) => candidate.summary),
      coverage: {
        scope,
        status,
        coverage,
        pagesFetched,
        candidates: candidateQueue.length,
        continuation,
        paginationState,
        candidateCutoff,
        warnings: [...new Set(warnings)],
        olderHistoryAudit: auditCoverage(auditState),
      },
      warnings,
      auditState,
      auditCandidateIds,
      auditScanComplete,
    };
  }

  private async auditOlderHistory(
    scope: HistoryScope,
    request: HistoryCollectionRequest,
    requestedRange: HistoryRange,
    now: Date,
    scanStartedAt: string,
    previous: OlderHistoryAuditState,
  ): Promise<{
    summaries: ConversationSummary[];
    warnings: string[];
    state: OlderHistoryAuditState;
    selectedConversationIds: string[];
    scanComplete: boolean;
  }> {
    const pageBudget =
      request.olderHistoryAudit?.maxPages ??
      DEFAULT_MAX_OLDER_HISTORY_AUDIT_PAGES;
    const pageSize = request.indexPageSize ?? this.indexPageSize;
    validatePositiveInteger(pageBudget, "older-history audit page budget");

    const rotate =
      previous.status === "complete" && previous.continuation === null;
    let offset = rotate ? 0 : previous.continuation ?? 0;
    let pagesFetched = 0;
    const summaries: ConversationSummary[] = [];
    const selectedConversationIds = new Set<string>();
    const warnings: string[] = [];
    const seenOffsets = new Set<number>();
    let scanComplete = false;
    let state: OlderHistoryAuditState = {
      ...previous,
      enabled: true,
      status: "in_progress",
      continuation: offset,
      lastStartedAt: scanStartedAt,
      lastPageAt: previous.lastPageAt,
    };

    while (pagesFetched < pageBudget) {
      if (this.options.store.loadAccountState().status === "paused") {
        state = { ...state, status: "partial", continuation: offset };
        break;
      }
      if (seenOffsets.has(offset)) {
        warnings.push("older_history_audit_repeated_offset");
        state = {
          ...state,
          status: "partial",
          continuation: offset,
        };
        break;
      }
      seenOffsets.add(offset);

      let page: AdaptedPage<ConversationSummary>;
      try {
        page = await this.reader.listConversations({
          archived: scope === "archived",
          offset,
          limit: pageSize,
          order: "updated",
        });
      } catch (error) {
        warnings.push(`older_history_audit_${errorCode(error)}`);
        if (isAccountStopError(error)) {
          const accountState = this.pauseForError(error, now);
          warnings.push(accountPauseWarning(accountState));
        }
        state = {
          ...state,
          status: "partial",
          continuation: offset,
        };
        break;
      }

      pagesFetched += 1;
      state = {
        ...state,
        pagesFetched: state.pagesFetched + 1,
        lastPageAt: now.toISOString(),
      };
      warnings.push(...page.warnings.map((warning) => `older_audit_${warning}`));
      for (const summary of page.items) {
        if (isOlderHistoryAuditCandidate(summary, requestedRange.start)) {
          summaries.push(summary);
          selectedConversationIds.add(summary.conversationId);
        }
      }

      if (
        page.paginationState === "unknown" ||
        page.paginationState === "contradictory" ||
        page.paginationState === "repeated_cursor"
      ) {
        warnings.push(`older_history_audit_${page.paginationState}`);
        state = {
          ...state,
          status: "partial",
          continuation:
            typeof page.continuation === "number" ? page.continuation : offset,
        };
        break;
      }
      if (page.exhausted && page.paginationState === "complete") {
        scanComplete =
          page.coverage === "validated_page" && page.warnings.length === 0;
        state = {
          ...state,
          status: "partial",
          continuation: null,
        };
        break;
      }

      if (
        page.paginationState !== "continuation" ||
        typeof page.continuation !== "number" ||
        page.continuation <= offset
      ) {
        warnings.push("older_history_audit_nonadvancing_continuation");
        state = {
          ...state,
          status: "partial",
          continuation:
            typeof page.continuation === "number" ? page.continuation : offset,
        };
        break;
      }

      offset = page.continuation;
      state = { ...state, continuation: offset };
      if (pagesFetched >= pageBudget) {
        warnings.push("older_history_audit_page_budget_exhausted");
        state = { ...state, status: "partial" };
        break;
      }
    }

    if (state.status === "in_progress") {
      warnings.push("older_history_audit_page_budget_exhausted");
      state = { ...state, status: "partial" };
    }
    return {
      summaries,
      warnings,
      state,
      selectedConversationIds: [...selectedConversationIds],
      scanComplete,
    };
  }

  private candidateCutoff(
    mode: HistoryCollectionRequest["mode"],
    requestedRange: HistoryRange,
    previous: DiscoveryCheckpoint | null,
    explicitRange: boolean,
    overlapMs: number,
  ): string {
    if (explicitRange || mode !== "incremental") {
      return requestedRange.start;
    }
    const previousWatermark = previous?.lastCompleteDiscoveryStartedAt;
    if (!previousWatermark) {
      return requestedRange.start;
    }
    const watermark = new Date(previousWatermark);
    if (!Number.isFinite(watermark.getTime())) {
      return requestedRange.start;
    }
    return new Date(watermark.getTime() - overlapMs).toISOString();
  }

  private async acquireConversation(
    candidate: Candidate,
    now: Date,
    request: HistoryCollectionRequest,
    identity: IdentityRecord,
    scanStartedAt: string,
  ): Promise<DetailAcquisition> {
    const existingRevisit =
      this.options.store
        .listRevisits()
        .find((entry) => entry.conversationId === candidate.summary.conversationId) ??
      null;
    let detail: ConversationDetailProjection;
    try {
      detail = await this.reader.fetchConversation(
        candidate.summary.conversationId,
        {
          allowLegacyFallback:
            request.legacyFallbackApproved ??
            this.options.legacyFallbackApproved ??
            this.legacyFallbackApproved,
        },
      );
    } catch (error) {
      const accountState = isAccountStopError(error)
        ? this.pauseForError(error, now)
        : null;
      const pageWarnings = [
        `detail_${errorCode(error)}`,
        ...(accountState ? [accountPauseWarning(accountState)] : []),
      ];
      const revisit = this.saveRevisit(
        candidate,
        now,
        classifyRevisitReason(error),
        0,
        errorCode(error),
        existingRevisit,
      );
      await this.commitPage({
        accountId: this.options.accountId,
        mode: request.mode,
        scanStartedAt,
        identity: accountState
          ? pausedIdentity(identity, accountState, errorCode(error))
          : identity,
        summary: candidate.summary,
        scopes: [...candidate.scopes].sort(),
        detail: null,
        messages: [],
        coverage: "partial",
        warnings: pageWarnings,
        pageKind: "detail",
        pageNumber: existingRevisit?.detailPagesFetched ?? 0,
        nextContinuation: revisit.continuation,
        revisit,
      });
      return {
        detail: null,
        messages: [],
        revisit,
        coverage: "partial",
        warnings: pageWarnings,
        detailPagesFetched: 0,
      };
    }

    const messages = dedupeMessages(detail.messages);
    detail = { ...detail, messages };
    const detailRevision = conversationRevision(detail);
    const detailIssue = detailRevisitIssue(detail);
    let detailRevisit = existingRevisit;
    if (detailIssue) {
      detailRevisit = this.saveRevisit(
        candidate,
        now,
        detailIssue.reason,
        0,
        null,
        existingRevisit,
        {
          continuation:
            existingRevisit?.continuation ?? normalizedContinuation(detail.continuation),
          continuationRevision:
            existingRevisit?.continuation !== null &&
            existingRevisit?.continuation !== undefined
              ? existingRevisit.continuationRevision
              : detailRevision,
          malformedPage: mergePageIssues(
            existingRevisit?.malformedPage ?? null,
            detailIssue,
          ),
        },
      );
    }
    if (detail.detailRoute === "legacy") {
      const revisit = this.finalizeRevisit(
        candidate,
        now,
        messages,
        detailRevisit,
        detailIssue?.reason ?? null,
        0,
        {
          malformedPage:
            detailIssue ?? existingRevisit?.malformedPage ?? null,
          validatedExhaustion: isValidatedDetailExhaustion(detail),
        },
      );
      await this.commitPage({
        accountId: this.options.accountId,
        mode: request.mode,
        scanStartedAt,
        identity,
        summary: candidate.summary,
        scopes: [...candidate.scopes].sort(),
        detail: { ...detail, messages, continuation: null },
        messages,
        coverage:
          revisit || detail.coverage !== "validated_page"
            ? "partial"
            : "complete",
        warnings: [
          ...detail.warnings,
          ...revisitWarnings(revisit),
        ],
        pageKind: "detail",
        pageNumber: detailRevisit?.detailPagesFetched ?? 0,
        nextContinuation: null,
        revisit,
      });
      return {
        detail,
        messages,
        revisit,
        coverage:
          revisit || detail.coverage !== "validated_page"
            ? "partial"
            : "complete",
        warnings: detail.warnings,
        detailPagesFetched: 0,
      };
    }

    if (
      detailRevisit ||
      detail.coverage !== "validated_page" ||
      detail.paginationState !== "complete"
    ) {
      await this.commitPage({
        accountId: this.options.accountId,
        mode: request.mode,
        scanStartedAt,
        identity,
        summary: candidate.summary,
        scopes: [...candidate.scopes].sort(),
        detail: { ...detail, messages },
        messages,
        coverage: "partial",
        warnings: detail.warnings,
        pageKind: "detail",
        pageNumber: detailRevisit?.detailPagesFetched ?? 0,
        nextContinuation:
          detailRevisit?.continuation ??
          (detail.paginationState === "continuation"
            ? detail.continuation
            : null),
        revisit: detailRevisit,
      });
    }

    const pageResult = await this.fetchModernMessages(
      candidate,
      detail,
      messages,
      now,
      detailRevisit,
      request.maxMessagePagesPerConversation ??
        this.maxMessagePagesPerConversation,
      request.mode,
      identity,
      scanStartedAt,
      detailRevision,
      detailIssue,
      existingRevisit,
    );
    return {
      detail: { ...detail, messages: pageResult.messages },
      messages: pageResult.messages,
      revisit: pageResult.revisit,
      coverage:
        pageResult.revisit || detail.coverage !== "validated_page"
          ? "partial"
          : "complete",
      warnings: [...detail.warnings, ...pageResult.warnings],
      detailPagesFetched: pageResult.detailPagesFetched,
    };
  }

  private async fetchModernMessages(
    candidate: Candidate,
    detail: ConversationDetailProjection,
    initialMessages: MessageRecord[],
    now: Date,
    existingRevisit: RevisitEntry | null,
    pageBudget: number,
    mode: HistoryCollectionRequest["mode"],
    identity: IdentityRecord,
    scanStartedAt: string,
    detailRevision: string | null,
    detailIssue: RevisitPageIssue | null,
    persistedRevisit: RevisitEntry | null,
  ): Promise<{
    messages: MessageRecord[];
    revisit: RevisitEntry | null;
    warnings: string[];
    detailPagesFetched: number;
  }> {
    const messages = [...initialMessages];
    const warnings: string[] = [];
    const seenCursors = new Set<string>();
    const savedContinuation = normalizedContinuation(
      existingRevisit?.continuation,
    );
    const continuationExpired =
      savedContinuation !== null &&
      isContinuationExpired(
        existingRevisit?.continuationRevision ?? null,
        detailRevision,
      );
    let before = continuationExpired
      ? null
      : savedContinuation ??
        (detail.paginationState === "continuation"
          ? normalizedContinuation(detail.continuation)
          : null);
    let detailPagesFetched = 0;
    let currentRevisit = existingRevisit;
    let continuationRestarted = continuationExpired;
    let pageNumber = existingRevisit?.detailPagesFetched ?? 0;
    let malformedPage = mergePageIssues(
      persistedRevisit?.malformedPage ?? null,
      detailIssue,
    );
    let restartValidationActive = continuationExpired;
    const startedFromNewest =
      continuationExpired ||
      (persistedRevisit?.malformedPage !== null &&
        persistedRevisit?.malformedPage !== undefined &&
        persistedRevisit.continuation === null &&
        detailIssue === null);

    if (continuationExpired) {
      warnings.push("messages_saved_continuation_expired_restarting");
    }

    while (detailPagesFetched < pageBudget) {
      let page;
      try {
        page = await this.reader.fetchMessages(
          candidate.summary.conversationId,
          {
            before,
            conversationSurface: detail.surface,
          },
        );
      } catch (error) {
        const reason = classifyRevisitReason(error);
        const accountState = isAccountStopError(error)
          ? this.pauseForError(error, now)
          : null;
        warnings.push(
          `messages_${errorCode(error)}`,
          ...(accountState ? [accountPauseWarning(accountState)] : []),
        );
        const revisit = this.saveRevisit(
          candidate,
          now,
          reason,
          0,
          errorCode(error),
          currentRevisit,
          {
            continuation: before,
            continuationRevision:
              before === null ? null : detailRevision,
            malformedPage,
            outstandingGeneration: generationStateFor(
              messages,
              currentRevisit,
              now,
            ),
            incrementAttempt: false,
          },
        );
        await this.commitPage({
          accountId: this.options.accountId,
          mode,
          scanStartedAt,
          identity: accountState
            ? pausedIdentity(identity, accountState, errorCode(error))
            : identity,
          summary: candidate.summary,
          scopes: [...candidate.scopes].sort(),
          detail: {
            ...detail,
            messages: dedupeMessages(messages),
            continuation: before,
            paginationState:
              before === null ? "unknown" : "continuation",
            coverage: "partial",
            warnings: [...detail.warnings, ...warnings],
          },
          messages: dedupeMessages(messages),
          coverage: "partial",
          warnings: uniqueWarnings([...detail.warnings, ...warnings]),
          pageKind: "messages",
          pageNumber,
          nextContinuation: before,
          revisit,
        });
        return {
          messages: dedupeMessages(messages),
          revisit,
          warnings: uniqueWarnings([
            ...warnings,
            ...revisitWarnings(revisit),
          ]),
          detailPagesFetched,
        };
      }
      detailPagesFetched += 1;
      pageNumber += 1;
      messages.push(...page.items);
      warnings.push(...page.warnings);
      const mergedMessages = dedupeMessages(messages);
      const pageIssue = messagePageRevisitIssue(page);
      malformedPage = mergePageIssues(malformedPage, pageIssue);

      if (page.paginationState === "complete" && page.exhausted) {
        const validatedPage = isValidatedPageExhaustion(page);
        const clearMalformedPage =
          malformedPage !== null &&
          (startedFromNewest || restartValidationActive) &&
          detailIssue === null &&
          pageIssue === null &&
          isValidatedDetailExhaustion(detail) &&
          validatedPage;
        const effectiveMalformedPage = clearMalformedPage
          ? null
          : malformedPage;
        const revisit = this.finalizeRevisit(
          candidate,
          now,
          mergedMessages,
          currentRevisit,
          detailIssue?.reason ?? null,
          1,
          {
            malformedPage: effectiveMalformedPage,
            validatedExhaustion:
              isValidatedDetailExhaustion(detail) && validatedPage,
          },
        );
        const pageWarnings = uniqueWarnings([
          ...detail.warnings,
          ...warnings,
          ...revisitWarnings(revisit),
        ]);
        await this.commitPage({
          accountId: this.options.accountId,
          mode,
          scanStartedAt,
          identity,
          summary: candidate.summary,
          scopes: [...candidate.scopes].sort(),
          detail: {
            ...detail,
            messages: mergedMessages,
            continuation: null,
            paginationState: page.paginationState,
            coverage:
              detail.coverage === "unrecognized"
                ? "unrecognized"
                : page.coverage,
            warnings: pageWarnings,
          },
          messages: mergedMessages,
          coverage: revisit ? "partial" : detail.coverage === "validated_page" && page.coverage === "validated_page" ? "complete" : "partial",
          warnings: pageWarnings,
          pageKind: "messages",
          pageNumber,
          nextContinuation: null,
          revisit,
        });
        return {
          messages: mergedMessages,
          revisit,
          warnings: pageWarnings,
          detailPagesFetched,
        };
      }

      const invalidPagination =
        page.paginationState === "repeated_cursor" ||
        page.paginationState === "contradictory" ||
        page.paginationState !== "continuation" ||
        typeof page.continuation !== "string" ||
        !page.continuation.trim();
      if (
        invalidPagination &&
        before !== null &&
        !continuationRestarted
      ) {
        continuationRestarted = true;
        restartValidationActive = true;
        warnings.push("messages_bad_continuation_restarting");
        const restartIssue =
          pageIssue ??
          ({
            reason: "bad_continuation",
            warnings: ["messages_bad_continuation_restarting"],
          } satisfies RevisitPageIssue);
        malformedPage = mergePageIssues(malformedPage, restartIssue);
        currentRevisit = this.saveRevisit(
          candidate,
          now,
          "bad_continuation",
          0,
          null,
          currentRevisit,
          {
            continuation: null,
            continuationRevision: null,
            malformedPage,
            outstandingGeneration: generationStateFor(
              mergedMessages,
              currentRevisit,
              now,
            ),
            pagesFetched: 1,
            incrementAttempt: false,
          },
        );
        await this.commitPage({
          accountId: this.options.accountId,
          mode,
          scanStartedAt,
          identity,
          summary: candidate.summary,
          scopes: [...candidate.scopes].sort(),
          detail: {
            ...detail,
            messages: mergedMessages,
            continuation: null,
            paginationState: page.paginationState,
            coverage: "partial",
            warnings: [...detail.warnings, ...warnings],
          },
          messages: mergedMessages,
          coverage: "partial",
          warnings: uniqueWarnings([...detail.warnings, ...warnings]),
          pageKind: "messages",
          pageNumber,
          nextContinuation: null,
          revisit: currentRevisit,
        });
        before = null;
        seenCursors.clear();
        continue;
      }

      if (
        invalidPagination
      ) {
        const issue =
          pageIssue ??
          ({
            reason:
              page.paginationState === "repeated_cursor"
                ? "repeated_cursor"
                : page.paginationState === "contradictory"
                  ? "contradictory_pagination"
                  : "unknown_pagination",
            warnings: [`messages_${page.paginationState}`],
          } satisfies RevisitPageIssue);
        malformedPage = mergePageIssues(malformedPage, issue);
        warnings.push(...issue.warnings);
        const revisit = this.saveRevisit(
          candidate,
          now,
          issue.reason,
          0,
          null,
          currentRevisit,
          {
            continuation: null,
            continuationRevision: null,
            malformedPage,
            outstandingGeneration: generationStateFor(
              mergedMessages,
              currentRevisit,
              now,
            ),
            pagesFetched: 1,
            incrementAttempt: false,
          },
        );
        await this.commitPage({
          accountId: this.options.accountId,
          mode,
          scanStartedAt,
          identity,
          summary: candidate.summary,
          scopes: [...candidate.scopes].sort(),
          detail: {
            ...detail,
            messages: mergedMessages,
            continuation: null,
            paginationState: page.paginationState,
            coverage: "partial",
            warnings: [...detail.warnings, ...warnings],
          },
          messages: mergedMessages,
          coverage: "partial",
          warnings: uniqueWarnings([...detail.warnings, ...warnings]),
          pageKind: "messages",
          pageNumber,
          nextContinuation: null,
          revisit,
        });
        return {
          messages: mergedMessages,
          revisit,
          warnings: uniqueWarnings([
            ...warnings,
            ...revisitWarnings(revisit),
          ]),
          detailPagesFetched,
        };
      }

      const cursor = (
        typeof page.continuation === "string" ? page.continuation : ""
      ).trim();
      if (seenCursors.has(cursor) || cursor === before) {
        if (before !== null && !continuationRestarted) {
          continuationRestarted = true;
          restartValidationActive = true;
          warnings.push("messages_bad_continuation_restarting");
          const restartIssue =
            pageIssue ??
            ({
              reason: "bad_continuation",
              warnings: ["messages_bad_continuation_restarting"],
            } satisfies RevisitPageIssue);
          malformedPage = mergePageIssues(malformedPage, restartIssue);
          currentRevisit = this.saveRevisit(
            candidate,
            now,
            "bad_continuation",
            0,
            null,
            currentRevisit,
            {
              continuation: null,
              continuationRevision: null,
              malformedPage,
              outstandingGeneration: generationStateFor(
                mergedMessages,
                currentRevisit,
                now,
              ),
              pagesFetched: 1,
              incrementAttempt: false,
            },
          );
          await this.commitPage({
            accountId: this.options.accountId,
            mode,
            scanStartedAt,
            identity,
            summary: candidate.summary,
            scopes: [...candidate.scopes].sort(),
            detail: {
              ...detail,
              messages: mergedMessages,
              continuation: null,
              paginationState: "repeated_cursor",
              coverage: "partial",
              warnings: [...detail.warnings, ...warnings],
            },
            messages: mergedMessages,
            coverage: "partial",
            warnings: uniqueWarnings([...detail.warnings, ...warnings]),
            pageKind: "messages",
            pageNumber,
            nextContinuation: null,
            revisit: currentRevisit,
          });
          before = null;
          seenCursors.clear();
          continue;
        }
        warnings.push("messages_repeated_cursor");
        const issue = {
          reason: "repeated_cursor",
          warnings: ["messages_repeated_cursor"],
        } satisfies RevisitPageIssue;
        malformedPage = mergePageIssues(malformedPage, issue);
        const revisit = this.saveRevisit(
          candidate,
          now,
          "repeated_cursor",
          0,
          null,
          currentRevisit,
          {
            continuation: null,
            continuationRevision: null,
            malformedPage,
            outstandingGeneration: generationStateFor(
              mergedMessages,
              currentRevisit,
              now,
            ),
            pagesFetched: 1,
            incrementAttempt: false,
          },
        );
        await this.commitPage({
          accountId: this.options.accountId,
          mode,
          scanStartedAt,
          identity,
          summary: candidate.summary,
          scopes: [...candidate.scopes].sort(),
          detail: {
            ...detail,
            messages: mergedMessages,
            continuation: null,
            paginationState: "repeated_cursor",
            coverage: "partial",
            warnings: [...detail.warnings, ...warnings],
          },
          messages: mergedMessages,
          coverage: "partial",
          warnings: uniqueWarnings([...detail.warnings, ...warnings]),
          pageKind: "messages",
          pageNumber,
          nextContinuation: null,
          revisit,
        });
        return {
          messages: mergedMessages,
          revisit,
          warnings: uniqueWarnings([
            ...warnings,
            ...revisitWarnings(revisit),
          ]),
          detailPagesFetched,
        };
      }
      seenCursors.add(cursor);
      before = cursor;
      const progressReason =
        pageIssue?.reason ?? malformedPage?.reason ?? "page_budget";
      currentRevisit = this.saveRevisit(
        candidate,
        now,
        progressReason,
        0,
        null,
        currentRevisit,
        {
          continuation: cursor,
          continuationRevision: detailRevision,
          malformedPage,
          outstandingGeneration: generationStateFor(
            mergedMessages,
            currentRevisit,
            now,
          ),
          pagesFetched: 1,
          incrementAttempt: false,
        },
      );
      if (detailPagesFetched >= pageBudget) {
        warnings.push("message_page_budget_exhausted");
        currentRevisit = this.saveRevisit(
          candidate,
          now,
          "page_budget",
          0,
          null,
          currentRevisit,
          {
            continuation: cursor,
            continuationRevision: detailRevision,
            malformedPage,
            outstandingGeneration: generationStateFor(
              mergedMessages,
              currentRevisit,
              now,
            ),
            incrementAttempt: false,
          },
        );
      }
      await this.commitPage({
        accountId: this.options.accountId,
        mode,
        scanStartedAt,
        identity,
        summary: candidate.summary,
        scopes: [...candidate.scopes].sort(),
        detail: {
          ...detail,
          messages: mergedMessages,
          continuation: cursor,
          paginationState: "continuation",
          coverage: "partial",
          warnings: [...detail.warnings, ...warnings],
        },
        messages: mergedMessages,
        coverage: "partial",
        warnings: uniqueWarnings([...detail.warnings, ...warnings]),
        pageKind: "messages",
        pageNumber,
        nextContinuation: cursor,
        revisit: currentRevisit,
      });
      if (detailPagesFetched >= pageBudget) {
        return {
          messages: mergedMessages,
          revisit: currentRevisit,
          warnings: uniqueWarnings([
            ...warnings,
            ...revisitWarnings(currentRevisit),
          ]),
          detailPagesFetched,
        };
      }
    }

    warnings.push("message_page_budget_exhausted");
    const revisit = this.saveRevisit(
      candidate,
      now,
      "page_budget",
      0,
      null,
      currentRevisit,
      {
        continuation: before,
        continuationRevision: before === null ? null : detailRevision,
        malformedPage,
        outstandingGeneration: generationStateFor(
          messages,
          currentRevisit,
          now,
        ),
        incrementAttempt: false,
      },
    );
    return {
      messages: dedupeMessages(messages),
      revisit,
      warnings: uniqueWarnings([...warnings, ...revisitWarnings(revisit)]),
      detailPagesFetched,
    };
  }

  private finalizeOlderHistoryAudits(
    discoveries: ScopeDiscoveryResult[],
    evidence: Map<HistoryScope, AuditEvidence>,
    scanStartedAt: string,
    now: Date,
  ): void {
    for (const discovery of discoveries) {
      if (!discovery.auditState.enabled) {
        continue;
      }
      const committedCandidateIds =
        evidence.get(discovery.scope)?.committedCandidateIds ??
        new Set<string>();
      const complete =
        discovery.auditScanComplete &&
        discovery.auditCandidateIds.every((conversationId) =>
          committedCandidateIds.has(conversationId),
        );
      const auditState: OlderHistoryAuditState = {
        ...discovery.auditState,
        status: complete ? "complete" : "partial",
        conversationsAudited:
          discovery.auditState.conversationsAudited +
          committedCandidateIds.size,
        lastCompletedAt: complete
          ? scanStartedAt
          : discovery.auditState.lastCompletedAt,
      };
      const checkpoint = this.options.store.loadDiscovery(discovery.scope);
      if (checkpoint) {
        this.options.store.saveDiscovery({
          ...checkpoint,
          olderHistoryAudit: auditState,
          updatedAt: now.toISOString(),
        });
      }
      discovery.coverage.olderHistoryAudit = auditState;
    }
  }

  private saveRevisit(
    candidate: Candidate,
    now: Date,
    reason: RevisitReason,
    detailPagesFetched: number,
    lastError: string | null,
    existing: RevisitEntry | null,
    options: RevisitOptions = {},
  ): RevisitEntry {
    const nowIso = now.toISOString();
    const pageDelta = options.pagesFetched ?? detailPagesFetched;
    const incrementAttempt = options.incrementAttempt ?? true;
    const outstandingGeneration =
      options.outstandingGeneration === undefined
        ? existing?.outstandingGeneration ?? null
        : options.outstandingGeneration;
    const attempts = Math.max(
      1,
      (existing?.attempts ?? 0) + (incrementAttempt ? 1 : 0),
    );
    const continuation =
      options.continuation === undefined
        ? existing?.continuation ?? null
        : normalizedContinuation(options.continuation);
    const continuationRevision =
      continuation === null
        ? null
        : options.continuationRevision === undefined
          ? existing?.continuationRevision ?? null
          : normalizedContinuationRevision(options.continuationRevision);
    const entry: RevisitEntry = {
      stateVersion: HISTORY_STATE_VERSION,
      accountId: this.options.accountId,
      conversationId: candidate.summary.conversationId,
      scopes: [...candidate.scopes].sort(),
      status: "pending",
      reason,
      firstSeenAt: existing?.firstSeenAt ?? nowIso,
      lastSeenAt: nowIso,
      attempts,
      nextEligibleAt:
        options.nextEligibleAt ??
        (outstandingGeneration
          ? nextOutstandingGenerationAt(now, attempts, outstandingGeneration)
          : nowIso),
      lastError,
      detailPagesFetched: (existing?.detailPagesFetched ?? 0) + pageDelta,
      continuation,
      continuationRevision,
      malformedPage:
        options.malformedPage === undefined
          ? existing?.malformedPage ?? null
          : options.malformedPage,
      outstandingGeneration,
    };
    this.options.store.upsertRevisit(entry);
    return entry;
  }

  private finalizeRevisit(
    candidate: Candidate,
    now: Date,
    messages: MessageRecord[],
    existing: RevisitEntry | null,
    detailReason: RevisitReason | null,
    detailPagesFetched = 0,
    options: {
      malformedPage?: RevisitPageIssue | null;
      validatedExhaustion?: boolean;
    } = {},
  ): RevisitEntry | null {
    const generationState = generationCompletionState(messages);
    const malformedPage =
      options.malformedPage === undefined
        ? existing?.malformedPage ?? null
        : options.malformedPage;
    const reason = generationState
      ? "nonterminal_generation"
      : malformedPage?.reason ?? detailReason;
    if (reason) {
      const outstandingGeneration = generationState
        ? buildOutstandingGenerationState(
            existing?.outstandingGeneration ?? null,
            generationState,
            now,
          )
        : null;
      return this.saveRevisit(
        candidate,
        now,
        reason,
        0,
        null,
        existing,
        {
          continuation: null,
          continuationRevision: null,
          malformedPage,
          outstandingGeneration,
          pagesFetched: detailPagesFetched,
          incrementAttempt: generationState !== null && existing !== null,
        },
      );
    }
    if (options.validatedExhaustion !== true) {
      if (!existing) {
        return null;
      }
      return this.saveRevisit(
        candidate,
        now,
        existing.reason,
        0,
        existing.lastError,
        existing,
        {
          continuation: null,
          continuationRevision: null,
          malformedPage,
          outstandingGeneration: null,
          pagesFetched: detailPagesFetched,
          incrementAttempt: false,
        },
      );
    }
    this.options.store.completeRevisit(
      this.options.accountId,
      candidate.summary.conversationId,
    );
    return null;
  }

  private async commitDiscoveryCheckpoint(
    checkpoint: DiscoveryCheckpoint,
    identity: IdentityRecord,
    scanStartedAt: string,
  ): Promise<void> {
    if (this.options.onDiscoveryPageCommit) {
      const page: HistoryDiscoveryPageCommit = {
        checkpoint,
        identity,
        scanStartedAt,
      };
      await this.options.onDiscoveryPageCommit(page);
    }
  }

  private async commitPage(page: HistoryPageCommit): Promise<void> {
    if (this.options.onPageCommit) {
      await this.options.onPageCommit(page);
      return;
    }
    this.options.store.acknowledgeCandidates(
      page.summary.conversationId,
      page.scopes,
    );
  }

  private buildCoverage(
    scopes: ScopeCoverageResult[],
    conversations: AcquiredConversation[],
    warnings: string[],
  ): HistoryCoverageResult {
    const active =
      scopes.find((scope) => scope.scope === "active") ??
      unknownScopeCoverage("active");
    const archived =
      scopes.find((scope) => scope.scope === "archived") ??
      unknownScopeCoverage("archived");
    const projectObserved = conversations.some(
      (conversation) => conversation.summary.projectId !== null,
    );
    const versionMetadataObserved = conversations.some(
      (conversation) => conversation.summary.hasVersions === true,
    );
    const activeBranchOnly =
      conversations.length > 0 &&
      conversations.every(
        (conversation) => conversation.summary.hasVersions === false,
      );
    const projects = projectObserved
      ? "validated_for_discovered_projects"
      : "unknown";
    const branches = versionMetadataObserved
      ? "version_metadata_observed"
      : activeBranchOnly
        ? "active_branch_only"
        : "unknown";
    this.reader.capabilities.projectCoverage = projects === "unknown" ? "unknown" : "partial";
    this.reader.capabilities.branchVisibility = branches;
    const gaps: string[] = [...warnings];
    if (projects === "unknown") {
      gaps.push("project_coverage_unknown");
    } else {
      gaps.push("project_visibility_unproven");
    }
    if (branches === "unknown") {
      gaps.push("branch_visibility_unknown");
    } else {
      gaps.push("branch_visibility_unproven");
    }
    if (conversations.some((conversation) => conversation.coverage !== "complete")) {
      gaps.push("incomplete_conversation_details");
    }
    const auditEnabled =
      active.olderHistoryAudit.enabled || archived.olderHistoryAudit.enabled;
    const auditStatus = !auditEnabled
      ? "disabled"
      : active.olderHistoryAudit.status === "complete" &&
          archived.olderHistoryAudit.status === "complete"
        ? "complete"
        : active.olderHistoryAudit.status === "in_progress" ||
            archived.olderHistoryAudit.status === "in_progress"
          ? "in_progress"
          : "partial";
    if (auditEnabled && auditStatus !== "complete") {
      gaps.push("older_history_audit_incomplete");
    }
    const overall =
      active.coverage === "complete" &&
      archived.coverage === "complete" &&
      gaps.length === 0
        ? "complete"
        : active.coverage === "unknown" && archived.coverage === "unknown"
          ? "unknown"
          : "partial";
    return {
      active,
      archived,
      projects,
      branches,
      olderHistoryAudit: {
        enabled: auditEnabled,
        status: auditStatus,
        active: active.olderHistoryAudit,
        archived: archived.olderHistoryAudit,
      },
      overall,
      gaps: [...new Set(gaps)],
    };
  }
}

function checkpointFor(
  accountId: string,
  scope: HistoryScope,
  mode: HistoryCollectionRequest["mode"],
  range: HistoryRange,
  candidateCutoff: string,
  scanStartedAt: string,
  continuation: number | null,
  pagesFetched: number,
  pageBudget: number,
  lastCompleteDiscoveryStartedAt: string | null,
  status: DiscoveryCheckpoint["status"],
  paginationState: PaginationState,
  warnings: string[],
  updatedAt: string,
  olderHistoryAudit: OlderHistoryAuditState,
  candidateQueue: DiscoveryCandidate[] = [],
  headFingerprint: string | null = null,
): DiscoveryCheckpoint {
  return {
    stateVersion: HISTORY_STATE_VERSION,
    accountId,
    scope,
    status,
    mode,
    range,
    candidateCutoff,
    scanStartedAt,
    continuation,
    pagesFetched,
    pageBudget,
    lastCompleteDiscoveryStartedAt,
    lastPageAt: updatedAt,
    paginationState,
    warnings: [...new Set(warnings)],
    updatedAt,
    candidateQueue: candidateQueue.map((candidate) => ({
      summary: { ...candidate.summary },
      missingUpdateTime: candidate.missingUpdateTime,
    })),
    headFingerprint,
    olderHistoryAudit,
  };
}

function normalizeOlderHistoryAudit(
  state: OlderHistoryAuditState | undefined,
): OlderHistoryAuditState {
  if (!state) {
    return {
      enabled: false,
      status: "disabled",
      continuation: null,
      pagesFetched: 0,
      conversationsAudited: 0,
      lastStartedAt: null,
      lastPageAt: null,
      lastCompletedAt: null,
    };
  }
  return {
    enabled: state.enabled === true,
    status: state.enabled === true ? state.status : "disabled",
    continuation:
      Number.isInteger(state.continuation) && (state.continuation ?? 0) >= 0
        ? state.continuation
        : null,
    pagesFetched:
      Number.isInteger(state.pagesFetched) && state.pagesFetched >= 0
        ? state.pagesFetched
        : 0,
    conversationsAudited:
      Number.isInteger(state.conversationsAudited) &&
      state.conversationsAudited >= 0
        ? state.conversationsAudited
        : 0,
    lastStartedAt: state.lastStartedAt ?? null,
    lastPageAt: state.lastPageAt ?? null,
    lastCompletedAt: state.lastCompletedAt ?? null,
  };
}

function auditCoverage(
  state: OlderHistoryAuditState,
): OlderHistoryAuditCoverage {
  return { ...normalizeOlderHistoryAudit(state) };
}

function frozenImplicitAcquisition(
  store: HistoryCheckpointStore,
  request: HistoryCollectionRequest,
): { range: HistoryRange; scanStartedAt: string } | null {
  if (request.mode !== "incremental" || request.range !== undefined) {
    return null;
  }
  const unfinished = (["active", "archived"] as const)
    .map((scope) => store.loadDiscovery(scope))
    .filter(
      (checkpoint): checkpoint is DiscoveryCheckpoint =>
        checkpoint !== null &&
        checkpoint.mode === "incremental" &&
        (checkpoint.status !== "complete" || checkpoint.continuation !== null) &&
        checkpoint.scanStartedAt !== null &&
        isValidInstant(checkpoint.scanStartedAt) &&
        validHistoryRange(checkpoint.range),
    );
  const first = unfinished[0];
  if (!first || first.scanStartedAt === null) {
    return null;
  }
  if (
    unfinished.some(
      (checkpoint) =>
        checkpoint.scanStartedAt !== first.scanStartedAt ||
        checkpoint.range.start !== first.range.start ||
        checkpoint.range.end !== first.range.end,
    )
  ) {
    return null;
  }
  return {
    range: { ...first.range },
    scanStartedAt: first.scanStartedAt,
  };
}

function shouldReuseFrozenAcquisition(
  previous: DiscoveryCheckpoint | null,
  request: HistoryCollectionRequest,
  requestedRange: HistoryRange,
  candidateCutoff: string,
): boolean {
  if (
    !previous ||
    previous.mode !== request.mode ||
    (previous.status === "complete" && previous.continuation === null) ||
    previous.scanStartedAt === null ||
    !isValidInstant(previous.scanStartedAt) ||
    !validHistoryRange(previous.range)
  ) {
    return false;
  }
  return (
    previous.range.start === requestedRange.start &&
    previous.range.end === requestedRange.end &&
    previous.candidateCutoff === candidateCutoff
  );
}

function normalizeCandidateQueue(
  queue: DiscoveryCandidate[] | undefined,
): DiscoveryCandidate[] {
  const normalized = new Map<string, DiscoveryCandidate>();
  for (const candidate of queue ?? []) {
    const summary = candidate?.summary;
    const conversationId = summary?.conversationId;
    if (typeof conversationId !== "string" || conversationId.trim() === "") {
      continue;
    }
    const existing = normalized.get(conversationId);
    const next: DiscoveryCandidate = {
      summary: { ...summary },
      missingUpdateTime:
        candidate.missingUpdateTime === true || summary.updatedAt === null,
    };
    if (!existing) {
      normalized.set(conversationId, next);
      continue;
    }
    if (isMoreRecent(next.summary.updatedAt, existing.summary.updatedAt)) {
      existing.summary = next.summary;
    }
    existing.missingUpdateTime ||= next.missingUpdateTime;
  }
  return [...normalized.values()];
}

function fingerprintIndexPage(page: AdaptedPage<ConversationSummary>): string {
  return createHash("sha256")
    .update(JSON.stringify(indexPageFingerprint(page)))
    .digest("hex");
}

function validHistoryRange(range: HistoryRange): boolean {
  const start = new Date(range.start).getTime();
  const end = new Date(range.end).getTime();
  return Number.isFinite(start) && Number.isFinite(end) && start < end;
}

function isOlderHistoryAuditCandidate(
  summary: ConversationSummary,
  cutoff: string,
): boolean {
  if (summary.updatedAt === null) {
    return true;
  }
  const updatedAt = new Date(summary.updatedAt).getTime();
  const cutoffTime = new Date(cutoff).getTime();
  return !Number.isFinite(updatedAt) ||
    !Number.isFinite(cutoffTime) ||
    updatedAt < cutoffTime;
}

function isCandidateSummary(
  summary: ConversationSummary,
  candidateCutoff: string,
  rangeEnd: string | null,
): boolean {
  if (summary.updatedAt === null) {
    return true;
  }
  const updatedAt = new Date(summary.updatedAt);
  const start = new Date(candidateCutoff);
  if (!Number.isFinite(updatedAt.getTime()) || !Number.isFinite(start.getTime())) {
    return true;
  }
  if (rangeEnd === null) {
    return updatedAt.getTime() >= start.getTime();
  }
  const end = new Date(rangeEnd);
  if (!Number.isFinite(end.getTime())) {
    return true;
  }
  return (
    updatedAt.getTime() >= start.getTime() &&
    updatedAt.getTime() < end.getTime()
  );
}

function isValidInstant(value: string): boolean {
  return Number.isFinite(new Date(value).getTime());
}

function indexPagesDiffer(
  left: AdaptedPage<ConversationSummary>,
  right: AdaptedPage<ConversationSummary>,
): boolean {
  return JSON.stringify(indexPageFingerprint(left)) !== JSON.stringify(
    indexPageFingerprint(right),
  );
}

function indexPageFingerprint(
  page: AdaptedPage<ConversationSummary>,
): {
  items: Array<{
    conversationId: string;
    updatedAt: string | null;
    workspaceId: string | null;
    projectId: string | null;
    hasVersions: boolean | null;
    currentNode: string | null;
    isArchived: boolean;
    surface: ConversationSummary["surface"];
  }>;
  continuation: string | number | null;
  exhausted: boolean;
  paginationState: PaginationState;
} {
  return {
    items: page.items.map((item) => ({
      conversationId: item.conversationId,
      updatedAt: item.updatedAt,
      workspaceId: item.workspaceId,
      projectId: item.projectId,
      hasVersions: item.hasVersions,
      currentNode: item.currentNode,
      isArchived: item.isArchived,
      surface: item.surface,
    })),
    continuation: page.continuation,
    exhausted: page.exhausted,
    paginationState: page.paginationState,
  };
}

function isMoreRecent(left: string | null, right: string | null): boolean {
  if (left === null) {
    return false;
  }
  if (right === null) {
    return true;
  }
  return new Date(left).getTime() > new Date(right).getTime();
}

function conversationRevision(
  detail: ConversationDetailProjection,
): string | null {
  const updatedAt = detail.updatedAt?.trim() ?? "";
  const currentNode = detail.currentNode?.trim() ?? "";
  if (!updatedAt && !currentNode) {
    return null;
  }
  return `${updatedAt}|${currentNode}`;
}

function normalizedContinuation(
  continuation: string | null | undefined,
): string | null {
  return typeof continuation === "string" && continuation.trim()
    ? continuation.trim()
    : null;
}

function normalizedContinuationRevision(
  revision: string | null | undefined,
): string | null {
  return typeof revision === "string" && revision.trim()
    ? revision.trim()
    : null;
}

function isContinuationExpired(
  savedRevision: string | null,
  currentRevision: string | null,
): boolean {
  return (
    savedRevision !== null &&
    currentRevision !== null &&
    savedRevision !== currentRevision
  );
}

function isValidatedDetailExhaustion(
  detail: ConversationDetailProjection,
): boolean {
  return (
    detail.coverage === "validated_page" &&
    detail.paginationState === "complete" &&
    detail.continuation === null &&
    detail.warnings.length === 0
  );
}

function isValidatedPageExhaustion<T>(
  page: AdaptedPage<T>,
): boolean {
  return (
    page.coverage === "validated_page" &&
    page.paginationState === "complete" &&
    page.exhausted &&
    page.warnings.length === 0
  );
}

function detailRevisitIssue(
  detail: ConversationDetailProjection,
): RevisitPageIssue | null {
  const reason = detailRevisitReason(detail);
  return reason
    ? { reason, warnings: [...detail.warnings] }
    : null;
}

function detailRevisitReason(
  detail: ConversationDetailProjection,
): RevisitReason | null {
  if (detail.coverage === "unrecognized") {
    return "unrecognized_detail";
  }
  if (detail.paginationState === "contradictory") {
    return "contradictory_pagination";
  }
  if (detail.paginationState === "unknown") {
    return "unknown_pagination";
  }
  if (detail.paginationState === "continuation") {
    return detail.continuation ? "incomplete_detail" : "unknown_pagination";
  }
  if (detail.coverage === "partial" || detail.warnings.length > 0) {
    return "partial_detail";
  }
  return null;
}

function messagePageRevisitIssue(
  page: AdaptedPage<MessageRecord>,
): RevisitPageIssue | null {
  const reason =
    page.paginationState === "repeated_cursor"
      ? "repeated_cursor"
      : page.paginationState === "contradictory"
        ? "contradictory_pagination"
        : page.paginationState === "unknown"
          ? "unknown_pagination"
          : page.coverage === "unrecognized"
            ? "unrecognized_detail"
            : page.coverage === "partial" || page.warnings.length > 0
              ? "partial_detail"
              : null;
  return reason
    ? { reason, warnings: [...page.warnings] }
    : null;
}

function generationCompletionState(
  messages: MessageRecord[],
): GenerationCompletionState | null {
  let unknown = false;
  for (const message of messages) {
    if (message.role !== "assistant" && message.role !== "tool") {
      continue;
    }
    const status = (message.status ?? "").trim().toLowerCase();
    if (NONTERMINAL_GENERATION_STATUSES.has(status)) {
      return "nonterminal";
    }
    if (status && TERMINAL_GENERATION_STATUSES.has(status)) {
      continue;
    }
    if (["cancelled", "error", "failed", "interrupted", "rejected"].includes(status)) {
      continue;
    }
    const hasGenerationIdentity =
      message.generationId !== null || message.requestId !== null;
    if (!hasGenerationIdentity) {
      continue;
    }
    unknown = true;
  }
  return unknown ? "unknown" : null;
}

function generationStateFor(
  messages: MessageRecord[],
  existing: RevisitEntry | null,
  now: Date,
): OutstandingGenerationState | null {
  const state = generationCompletionState(messages);
  return state
    ? buildOutstandingGenerationState(
        existing?.outstandingGeneration ?? null,
        state,
        now,
      )
    : null;
}

function buildOutstandingGenerationState(
  existing: OutstandingGenerationState | null,
  state: GenerationCompletionState,
  now: Date,
): OutstandingGenerationState {
  const since = existing?.since ?? now.toISOString();
  const sinceMs = new Date(since).getTime();
  const timedOut =
    existing?.timedOut === true ||
    (Number.isFinite(sinceMs) &&
      now.getTime() - sinceMs >= OUTSTANDING_GENERATION_TIMEOUT_MS);
  return {
    state: existing?.state === "unknown" || timedOut ? "unknown" : state,
    since,
    timedOut,
  };
}

function nextOutstandingGenerationAt(
  now: Date,
  attempts: number,
  state: OutstandingGenerationState,
): string {
  const delay = state.timedOut
    ? OUTSTANDING_GENERATION_REVISIT_MAX_DELAY_MS
    : Math.min(
        OUTSTANDING_GENERATION_REVISIT_BASE_DELAY_MS *
          2 ** Math.max(0, attempts - 1),
        OUTSTANDING_GENERATION_REVISIT_MAX_DELAY_MS,
      );
  return new Date(now.getTime() + delay).toISOString();
}

function revisitWarnings(revisit: RevisitEntry | null): string[] {
  const state = revisit?.outstandingGeneration;
  if (!state) {
    return [];
  }
  return [
    "nonterminal_generation_pending",
    ...(state.state === "unknown"
      ? ["generation_completion_unknown"]
      : []),
    ...(state.timedOut ? ["nonterminal_generation_timeout"] : []),
  ];
}

function mergePageIssues(
  left: RevisitPageIssue | null,
  right: RevisitPageIssue | null,
): RevisitPageIssue | null {
  if (!left) {
    return right;
  }
  if (!right) {
    return left;
  }
  return {
    reason: left.reason,
    warnings: [...new Set([...left.warnings, ...right.warnings])],
  };
}

function dedupeMessages(messages: MessageRecord[]): MessageRecord[] {
  const byId = new Map<string, MessageRecord>();
  for (const message of messages) {
    const previous = byId.get(message.messageId);
    if (!previous) {
      byId.set(message.messageId, message);
      continue;
    }
    // Message pages can omit graph links that were present in the detail mapping.
    byId.set(message.messageId, {
      ...message,
      nodeId: message.nodeId === message.messageId
        ? previous.nodeId ?? message.nodeId
        : message.nodeId ?? previous.nodeId,
      parentId: message.parentId ?? previous.parentId,
      children: [...new Set([...previous.children, ...message.children])].sort(),
      role: message.role ?? previous.role,
      channel: message.channel ?? previous.channel,
      createdAt: message.createdAt ?? previous.createdAt,
      status: message.status ?? previous.status,
      endTurn: message.endTurn ?? previous.endTurn,
      requestedModelRaw: message.requestedModelRaw ?? previous.requestedModelRaw,
      requestedModeRaw: message.requestedModeRaw ?? previous.requestedModeRaw,
      requestedReasoningEffortRaw:
        message.requestedReasoningEffortRaw ?? previous.requestedReasoningEffortRaw,
      recordedFinalModelRaw:
        message.recordedFinalModelRaw ?? previous.recordedFinalModelRaw,
      generationId: message.generationId ?? previous.generationId,
      requestId: message.requestId ?? previous.requestId,
      origin: message.origin ?? previous.origin,
      metadata: { ...previous.metadata, ...message.metadata },
    });
  }
  return [...byId.values()];
}

function revisitSummary(entry: RevisitEntry): ConversationSummary {
  return {
    conversationId: entry.conversationId,
    createdAt: null,
    updatedAt: null,
    isArchived: entry.scopes.length === 1 && entry.scopes[0] === "archived",
    workspaceId: null,
    projectId: null,
    surface: "unknown",
    origin: null,
    hasVersions: null,
    currentNode: null,
    coverage: "partial",
  };
}

function classifyRevisitReason(error: unknown): RevisitReason {
  if (error instanceof LegacyFallbackNotApprovedError) {
    return "detail_unavailable";
  }
  if (error instanceof RateLimitedError) {
    return "detail_unavailable";
  }
  if (error instanceof AuthenticationRequiredError) {
    return "detail_unavailable";
  }
  return "incomplete_detail";
}

function errorCode(error: unknown): string {
  if (error instanceof RateLimitedError) {
    return "rate_limited";
  }
  if (error instanceof AuthenticationRequiredError) {
    return "auth_required";
  }
  if (error instanceof LegacyFallbackNotApprovedError) {
    return "legacy_fallback_not_approved";
  }
  if (error instanceof CapabilityError) {
    return `capability_${error.capability}`;
  }
  if (error instanceof HttpStatusError) {
    return `http_${error.status}`;
  }
  if (error instanceof Error && error.name) {
    return error.name.toLowerCase().replace(/[^a-z0-9]+/g, "_");
  }
  return "adapter_error";
}

function isAccountStopError(error: unknown): boolean {
  return (
    error instanceof AuthenticationRequiredError ||
    (error instanceof HttpStatusError &&
      [401, 403, 429].includes(error.status))
  );
}

function isRateLimitedError(error: unknown): boolean {
  return (
    error instanceof RateLimitedError ||
    (error instanceof HttpStatusError && error.status === 429)
  );
}

function retryAfterDeadline(value: string | null, now: Date): string | null {
  if (value === null) {
    return null;
  }
  const trimmed = value.trim();
  if (!trimmed) {
    return null;
  }
  if (/^\d+(?:\.\d+)?$/.test(trimmed)) {
    const seconds = Number(trimmed);
    const deadline = now.getTime() + seconds * 1000;
    const date = new Date(deadline);
    return Number.isFinite(seconds) && Number.isFinite(date.getTime())
      ? date.toISOString()
      : null;
  }
  const date = new Date(trimmed);
  return Number.isFinite(date.getTime()) ? date.toISOString() : null;
}

function readyAccountState(): HistoryAccountState {
  return {
    status: "ready",
    reason: null,
    pausedAt: null,
    cooldownUntil: null,
    lastError: null,
  };
}

function isActiveCooldown(
  state: HistoryAccountState,
  now: Date,
): boolean {
  if (state.status !== "paused" || state.reason !== "cooldown") {
    return false;
  }
  if (state.cooldownUntil === null) {
    return true;
  }
  return new Date(state.cooldownUntil).getTime() > now.getTime();
}

function accountPauseWarning(state: HistoryAccountState): string {
  return state.reason === "cooldown"
    ? "account_paused_cooldown"
    : "account_paused_authentication";
}

function pausedIdentity(
  identity: IdentityRecord | null,
  state: HistoryAccountState,
  errorCodeValue?: string,
): IdentityRecord {
  return {
    providerUserId: identity?.providerUserId ?? null,
    workspaceId: identity?.workspaceId ?? null,
    quotaOwnerId: identity?.quotaOwnerId ?? null,
    surface: identity?.surface ?? "unknown",
    authState: "paused",
    identityErrors: uniqueWarnings([
      ...(identity?.identityErrors ?? []),
      accountPauseWarning(state),
      ...(state.lastError ? [state.lastError] : []),
      ...(errorCodeValue ? [errorCodeValue] : []),
    ]),
  };
}

function unknownScopeCoverage(scope: HistoryScope): ScopeCoverageResult {
  return {
    scope,
    status: "not_started",
    coverage: "unknown",
    pagesFetched: 0,
    candidates: 0,
    continuation: null,
    paginationState: "unknown",
    candidateCutoff: "",
    warnings: ["scope_not_scanned"],
    olderHistoryAudit: normalizeOlderHistoryAudit(undefined),
  };
}

function blockedResult(
  accountId: string,
  mode: HistoryCollectionRequest["mode"],
  range: HistoryRange,
  scanStartedAt: string,
  identity: IdentityRecord,
  warning: string,
  accountState: HistoryAccountState,
): HistoryCollectionResult {
  const active = unknownScopeCoverage("active");
  const archived = unknownScopeCoverage("archived");
  return {
    accountId,
    mode,
    range,
    scanStartedAt,
    status: "blocked",
    accountState,
    identity,
    scopes: [active, archived],
    conversations: [],
    revisits: [],
    coverage: {
      active,
      archived,
      projects: "unknown",
      branches: "unknown",
      olderHistoryAudit: {
        enabled: false,
        status: "disabled",
        active: active.olderHistoryAudit,
        archived: archived.olderHistoryAudit,
      },
      overall: "unknown",
      gaps: [warning],
    },
    pagesFetched: 0,
    detailPagesFetched: 0,
    warnings: [warning],
  };
}

function validatePositiveInteger(value: number, label: string): void {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${label} must be a positive integer`);
  }
}

function uniqueWarnings(warnings: string[]): string[] {
  return [...new Set(warnings)];
}
