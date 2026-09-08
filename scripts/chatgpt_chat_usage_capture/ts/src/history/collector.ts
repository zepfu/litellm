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
  DiscoveryCandidate,
  DiscoveryCheckpoint,
  HistoryCollectionOptions,
  HistoryCollectionRequest,
  HistoryCollectionResult,
  HistoryCoverageResult,
  HistoryRange,
  HistoryDiscoveryPageCommit,
  HistoryCheckpointStore,
  HistoryAccountState,
  HistoryPageCommit,
  HistoryReader,
  HistoryScope,
  OlderHistoryAuditCoverage,
  OlderHistoryAuditState,
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
}

interface DetailAcquisition {
  detail: ConversationDetailProjection | null;
  messages: MessageRecord[];
  revisit: RevisitEntry | null;
  coverage: AcquiredConversation["coverage"];
  warnings: string[];
  detailPagesFetched: number;
  blocked: boolean;
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
  private readonly committedConversationIds = new Set<string>();

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

    const identity = await this.readIdentity();

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
    const scopeResults: ScopeCoverageResult[] = [];
    const warnings: string[] = [];
    let pagesFetched = 0;

    for (const scope of ["active", "archived"] as const) {
      const discovery = await this.discoverScopeDurable(
        scope,
        request,
        requestedRange,
        now,
        scanStartedAt,
        identity,
      );
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

    const conversations: AcquiredConversation[] = [];
    let detailPagesFetched = 0;
    let blocked = false;
    for (const candidate of candidates.values()) {
      const acquisition = await this.acquireConversation(
        candidate,
        now,
        request,
        identity,
        scanStartedAt,
        requestedRange,
      );
      detailPagesFetched += acquisition.detailPagesFetched;
      blocked ||= acquisition.blocked;
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
      if (blocked) {
        break;
      }
    }

    const revisits = this.options.store.listRevisits();
    const coverage = this.buildCoverage(scopeResults, conversations, warnings);
    const status =
      !blocked && coverage.overall === "complete" && revisits.length === 0
        ? "complete"
        : blocked
          ? "blocked"
          : "partial";

    return {
      accountId: this.options.accountId,
      mode: request.mode,
      range: requestedRange,
      scanStartedAt,
      status,
      accountState: this.options.store.loadAccountState(),
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

  private async readIdentity(): Promise<IdentityRecord> {
    try {
      return await this.reader.inspectSessionIdentity();
    } catch (error) {
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
            request.mode === "incremental" && !explicitRange
              ? acquisitionRange.end
              : null,
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
          status = "partial";
          paginationState = "unknown";
          continuation = null;
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
    if (request.olderHistoryAudit?.enabled) {
      const audit = await this.auditOlderHistory(
        scope,
        request,
        acquisitionRange,
        now,
        acquisitionStartAt,
        priorAudit,
      );
      auditState = audit.state;
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
    const warnings: string[] = [];
    const seenOffsets = new Set<number>();
    let state: OlderHistoryAuditState = {
      ...previous,
      enabled: true,
      status: "in_progress",
      continuation: offset,
      lastStartedAt: scanStartedAt,
      lastPageAt: previous.lastPageAt,
    };

    while (pagesFetched < pageBudget) {
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
        conversationsAudited:
          state.conversationsAudited + page.items.length,
        lastPageAt: now.toISOString(),
      };
      warnings.push(...page.warnings.map((warning) => `older_audit_${warning}`));
      summaries.push(
        ...page.items.filter((summary) =>
          isOlderHistoryAuditCandidate(summary, requestedRange.start),
        ),
      );

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
        const complete =
          page.coverage === "validated_page" && page.warnings.length === 0;
        state = {
          ...state,
          status: complete ? "complete" : "partial",
          continuation: null,
          lastCompletedAt: complete ? scanStartedAt : state.lastCompletedAt,
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
    return { summaries, warnings, state };
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
    requestedRange: HistoryRange,
  ): Promise<DetailAcquisition> {
    const warnings: string[] = [];
    const implicitUpperBound =
      request.mode === "incremental" && request.range === undefined;
    let retainedMessages: MessageRecord[] = [];
    if (this.options.loadConversationMetadata) {
      try {
        const retained = await this.options.loadConversationMetadata(
          candidate.summary.conversationId,
        );
        if (retained) {
          retainedMessages = dedupeMessages(retained.messages ?? []);
          if (isMoreRecent(retained.summary.updatedAt, candidate.summary.updatedAt)) {
            candidate.summary = retained.summary;
          }
        }
      } catch (error) {
        warnings.push(`retained_metadata_${errorCode(error)}`);
      }
    }
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
      const revisit = this.saveRevisit(
        candidate,
        now,
        classifyRevisitReason(error),
        0,
        errorCode(error),
        existingRevisit,
      );
      const blockedByAccount = isAccountBlockingError(error);
      if (blockedByAccount) {
        this.pauseAccount(error, now);
      }
      await this.commitPage({
        accountId: this.options.accountId,
        mode: request.mode,
        scanStartedAt,
        identity,
        summary: candidate.summary,
        scopes: [...candidate.scopes].sort(),
        detail: null,
        messages: messagesBeforeExclusiveEnd(
          retainedMessages,
          implicitUpperBound ? requestedRange.end : null,
        ),
        coverage: "partial",
        warnings: [...warnings, `detail_${errorCode(error)}`],
        pageKind: "detail",
        pageNumber: existingRevisit?.detailPagesFetched ?? 0,
        nextContinuation: revisit.continuation,
        revisit,
      });
      if (!blockedByAccount) {
        this.pauseAccount(error, now);
      }
      return {
        detail: null,
        messages: messagesBeforeExclusiveEnd(
          retainedMessages,
          implicitUpperBound ? requestedRange.end : null,
        ),
        revisit,
        coverage: "partial",
        warnings: [...warnings, `detail_${errorCode(error)}`],
        detailPagesFetched: 0,
        blocked: isAccountBlockingError(error),
      };
    }

    const messages = messagesBeforeExclusiveEnd(
      dedupeMessages([...retainedMessages, ...detail.messages]),
      implicitUpperBound ? requestedRange.end : null,
    );
    detail = { ...detail, messages };
    const continuationRevision = conversationRevision(detail);
    const detailReason = detailRevisitReason(detail);
    let detailRevisit = existingRevisit;
    if (detailReason) {
      detailRevisit = this.saveRevisit(
        candidate,
        now,
        detailReason,
        0,
        null,
        existingRevisit,
        {
          continuation: detail.continuation,
          continuationRevision,
        },
      );
    }
    if (
      detailRevisit !== null &&
      detailRevisit.continuation !== null &&
      detailRevisit.continuationRevision !== null &&
      detailRevisit.continuationRevision !== continuationRevision
    ) {
      warnings.push("conversation_revision_changed_restart");
      detailRevisit = this.saveRevisit(
        candidate,
        now,
        "bad_continuation",
        0,
        null,
        detailRevisit,
        {
          continuation: null,
          continuationRevision,
          incrementAttempt: false,
        },
      );
    }
    if (detail.detailRoute === "legacy") {
      const revisit = this.finalizeRevisit(
        candidate,
        now,
        messages,
        detailRevisit,
        detailReason,
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
          ...warnings,
          ...detail.warnings,
          ...(revisit?.reason === "nonterminal_generation"
            ? ["nonterminal_generation_pending"]
            : []),
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
        blocked: false,
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
        warnings: [...warnings, ...detail.warnings],
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
      implicitUpperBound ? requestedRange.end : null,
    );
    return {
      detail: { ...detail, messages: pageResult.messages },
      messages: pageResult.messages,
      revisit: pageResult.revisit,
      coverage:
        pageResult.revisit || detail.coverage !== "validated_page"
          ? "partial"
          : "complete",
      warnings: [...warnings, ...detail.warnings, ...pageResult.warnings],
      detailPagesFetched: pageResult.detailPagesFetched,
      blocked: false,
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
    rangeEnd: string | null,
  ): Promise<{
    messages: MessageRecord[];
    revisit: RevisitEntry | null;
    warnings: string[];
    detailPagesFetched: number;
  }> {
    const messages = [...initialMessages];
    const warnings: string[] = [];
    const seenCursors = new Set<string>();
    let before =
      existingRevisit?.continuation ??
      (detail.paginationState === "continuation" ? detail.continuation : null);
    let detailPagesFetched = 0;
    let currentRevisit = existingRevisit;
    let continuationRestarted = false;
    let pageNumber = existingRevisit?.detailPagesFetched ?? 0;
    let retainedPageReason: RevisitReason | null = null;
    const continuationRevision = conversationRevision(detail);

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
        warnings.push(`messages_${errorCode(error)}`);
        const revisit = this.saveRevisit(
          candidate,
          now,
          reason,
          0,
          errorCode(error),
          currentRevisit,
          {
            continuation: before,
            continuationRevision,
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
          warnings,
          detailPagesFetched,
        };
      }
      detailPagesFetched += 1;
      pageNumber += 1;
      messages.push(...messagesBeforeExclusiveEnd(page.items, rangeEnd));
      warnings.push(...page.warnings);
      const mergedMessages = dedupeMessages(messages);
      const pageRevisitReason = messagePageRevisitReason(page);
      const malformedPage =
        page.paginationState === "contradictory"
          ? {
              reason: "contradictory_pagination" as const,
              warnings: [...page.warnings],
            }
          : page.coverage === "unrecognized"
            ? {
                reason: "unrecognized_detail" as const,
                warnings: [...page.warnings],
              }
            : page.coverage === "partial" || page.warnings.length > 0
              ? {
                  reason: "partial_detail" as const,
                  warnings: [...page.warnings],
                }
              : null;
      if (malformedPage) {
        this.markMalformedPage(
          currentRevisit,
          malformedPage.reason,
          malformedPage.warnings,
        );
        if (!currentRevisit) {
          currentRevisit = this.saveRevisit(
            candidate,
            now,
            malformedPage.reason,
            0,
            null,
            null,
            {
              continuation: null,
              continuationRevision,
              incrementAttempt: false,
              malformedPage,
            },
          );
        }
      }

      if (page.paginationState === "complete" && page.exhausted) {
        const revisit = this.finalizeRevisit(
          candidate,
          now,
          mergedMessages,
          currentRevisit,
          detailRevisitReason(detail) ??
            retainedPageReason ??
            pageRevisitReason,
          1,
        );
        const pageWarnings = uniqueWarnings([...detail.warnings, ...warnings]);
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
          warnings,
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
        warnings.push("messages_bad_continuation_restarting");
        currentRevisit = this.saveRevisit(
          candidate,
          now,
          "bad_continuation",
          0,
          null,
          currentRevisit,
          {
            continuation: null,
            pagesFetched: 1,
            incrementAttempt: false,
            continuationRevision,
            ...(malformedPage ? { malformedPage } : {}),
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
        const reason =
          page.paginationState === "repeated_cursor"
            ? "repeated_cursor"
            : page.paginationState === "contradictory"
              ? "contradictory_pagination"
              : "unknown_pagination";
        warnings.push(`messages_${page.paginationState}`);
        const revisit = this.saveRevisit(
          candidate,
          now,
          reason,
          0,
          null,
          currentRevisit,
          {
            continuation: null,
            pagesFetched: 1,
            incrementAttempt: false,
            continuationRevision,
            ...(malformedPage ? { malformedPage } : {}),
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
          warnings,
          detailPagesFetched,
        };
      }

      const cursor = (
        typeof page.continuation === "string" ? page.continuation : ""
      ).trim();
      if (seenCursors.has(cursor) || cursor === before) {
        if (before !== null && !continuationRestarted) {
          continuationRestarted = true;
          warnings.push("messages_bad_continuation_restarting");
          currentRevisit = this.saveRevisit(
            candidate,
            now,
            "bad_continuation",
            0,
            null,
            currentRevisit,
            {
              continuation: null,
              pagesFetched: 1,
              incrementAttempt: false,
              continuationRevision,
              ...(malformedPage ? { malformedPage } : {}),
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
        const revisit = this.saveRevisit(
          candidate,
          now,
          "repeated_cursor",
          0,
          null,
          currentRevisit,
          {
            continuation: null,
            pagesFetched: 1,
            incrementAttempt: false,
            continuationRevision,
            ...(malformedPage ? { malformedPage } : {}),
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
          warnings,
          detailPagesFetched,
        };
      }
      seenCursors.add(cursor);
      before = cursor;
      retainedPageReason ??= pageRevisitReason;
      currentRevisit = this.saveRevisit(
        candidate,
        now,
        pageRevisitReason ?? "page_budget",
        0,
        null,
        currentRevisit,
        {
          continuation: cursor,
          pagesFetched: 1,
          incrementAttempt: false,
          continuationRevision,
          ...(malformedPage ? { malformedPage } : {}),
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
            incrementAttempt: false,
            continuationRevision,
            ...(malformedPage ? { malformedPage } : {}),
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
          warnings,
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
        incrementAttempt: false,
        continuationRevision,
      },
    );
    const boundedMessages = messagesBeforeExclusiveEnd(
      dedupeMessages(messages),
      rangeEnd,
    );
    return {
      messages: boundedMessages,
      revisit,
      warnings,
      detailPagesFetched,
    };
  }

  private saveRevisit(
    candidate: Candidate,
    now: Date,
    reason: RevisitReason,
    detailPagesFetched: number,
    lastError: string | null,
    existing: RevisitEntry | null,
    options: {
      continuation?: string | null;
      pagesFetched?: number;
      incrementAttempt?: boolean;
      continuationRevision?: string | null;
      malformedPage?: RevisitPageIssue | null;
    } = {},
  ): RevisitEntry {
    const nowIso = now.toISOString();
    const pageDelta = options.pagesFetched ?? detailPagesFetched;
    const incrementAttempt = options.incrementAttempt ?? true;
    const entry: RevisitEntry = {
      stateVersion: HISTORY_STATE_VERSION,
      accountId: this.options.accountId,
      conversationId: candidate.summary.conversationId,
      scopes: [...candidate.scopes].sort(),
      status: "pending",
      reason,
      firstSeenAt: existing?.firstSeenAt ?? nowIso,
      lastSeenAt: nowIso,
      attempts: Math.max(
        1,
        (existing?.attempts ?? 0) + (incrementAttempt ? 1 : 0),
      ),
      nextEligibleAt: nowIso,
      lastError,
      detailPagesFetched: (existing?.detailPagesFetched ?? 0) + pageDelta,
      continuationRevision:
        options.continuationRevision ?? existing?.continuationRevision ?? null,
      malformedPage: options.malformedPage ?? existing?.malformedPage ?? null,
      outstandingGeneration:
        reason === "nonterminal_generation"
          ? existing?.outstandingGeneration ?? {
              state: "unknown",
              since: nowIso,
              timedOut: false,
            }
          : existing?.outstandingGeneration ?? null,
      continuation:
        options.continuation === undefined
          ? existing?.continuation ?? null
          : options.continuation,
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
  ): RevisitEntry | null {
    const reason = hasOutstandingGeneration(messages)
      ? "nonterminal_generation"
      : detailReason;
    if (reason) {
      return this.saveRevisit(
        candidate,
        now,
        reason,
        0,
        null,
        existing,
        {
          continuation: null,
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

  private pauseAccount(error: unknown, now: Date): void {
    const nowIso = now.toISOString();
    if (error instanceof AuthenticationRequiredError) {
      this.setAccountState({
        status: "paused",
        reason: "authentication",
        pausedAt: nowIso,
        cooldownUntil: null,
        lastError: "auth_required",
      });
      return;
    }
    if (error instanceof RateLimitedError) {
      this.setAccountState({
        status: "paused",
        reason: "cooldown",
        pausedAt: nowIso,
        cooldownUntil: parseRetryAfter(error.retryAfter, now).toISOString(),
        lastError: "rate_limited",
      });
    }
  }

  private setAccountState(state: HistoryAccountState): void {
    this.options.store.saveAccountState(state);
    if (this.options.onAccountStateCommit) {
      void this.options.onAccountStateCommit(state);
    }
  }

  private markMalformedPage(
    currentRevisit: RevisitEntry | null,
    reason: Extract<RevisitReason, "partial_detail" | "unrecognized_detail" | "contradictory_pagination">,
    warnings: readonly string[],
  ): void {
    if (!currentRevisit) {
      return;
    }
    const next: RevisitEntry = {
      ...currentRevisit,
      malformedPage: { reason, warnings: [...warnings] },
    };
    this.options.store.upsertRevisit(next);
    currentRevisit.malformedPage = next.malformedPage;
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

  private async commitPage(
    page: Omit<HistoryPageCommit, "accountState">,
  ): Promise<void> {
    const committedPage: HistoryPageCommit = {
      ...page,
      accountState: this.options.store.loadAccountState(),
    };
    if (this.options.onPageCommit) {
      await this.options.onPageCommit(committedPage);
      this.committedConversationIds.add(committedPage.summary.conversationId);
      return;
    }
    this.options.store.acknowledgeCandidates(
      committedPage.summary.conversationId,
      committedPage.scopes,
    );
    this.committedConversationIds.add(committedPage.summary.conversationId);
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
    if (
      conversations.some(
        (conversation) =>
          !this.committedConversationIds.has(conversation.summary.conversationId),
      )
    ) {
      gaps.push("uncommitted_conversation_details");
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

function messagesBeforeExclusiveEnd(
  messages: MessageRecord[],
  rangeEnd: string | null,
): MessageRecord[] {
  if (rangeEnd === null) {
    return messages;
  }
  const end = new Date(rangeEnd);
  if (!Number.isFinite(end.getTime())) {
    return messages;
  }
  return messages.filter((message) => {
    if (message.createdAt === null) {
      return true;
    }
    const createdAt = new Date(message.createdAt);
    return (
      !Number.isFinite(createdAt.getTime()) ||
      createdAt.getTime() < end.getTime()
    );
  });
}

function conversationRevision(detail: ConversationDetailProjection): string {
  return createHash("sha256")
    .update(
      JSON.stringify({
        conversationId: detail.conversationId,
        currentNode: detail.currentNode,
        updatedAt: detail.updatedAt,
        continuation: detail.continuation,
      }),
    )
    .digest("hex");
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

function messagePageRevisitReason(
  page: AdaptedPage<MessageRecord>,
): RevisitReason | null {
  if (page.coverage === "unrecognized") {
    return "unrecognized_detail";
  }
  if (page.coverage === "partial" || page.warnings.length > 0) {
    return "partial_detail";
  }
  return null;
}

function hasOutstandingGeneration(messages: MessageRecord[]): boolean {
  return messages.some((message) => {
    if (message.role !== "assistant" && message.role !== "tool") {
      return false;
    }
    const status = (message.status ?? "").trim().toLowerCase();
    if (NONTERMINAL_GENERATION_STATUSES.has(status)) {
      return true;
    }
    if (status && TERMINAL_GENERATION_STATUSES.has(status)) {
      return false;
    }
    if (["cancelled", "error", "failed", "interrupted", "rejected"].includes(status)) {
      return false;
    }
    const hasGenerationIdentity =
      message.generationId !== null || message.requestId !== null;
    if (!hasGenerationIdentity) {
      return false;
    }
    // An unknown provider status is never made terminal by an end-turn flag.
    // Preserve it for a later reread when generation identity is present.
    return true;
  });
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

function isAccountBlockingError(error: unknown): boolean {
  return (
    error instanceof AuthenticationRequiredError ||
    error instanceof RateLimitedError
  );
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
  accountState: HistoryAccountState = {
    status: "ready",
    reason: null,
    pausedAt: null,
    cooldownUntil: null,
    lastError: null,
  },
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

function parseRetryAfter(value: string | null, now: Date): Date {
  if (!value) {
    return now;
  }
  const seconds = Number(value);
  if (Number.isFinite(seconds) && seconds >= 0) {
    return new Date(now.getTime() + seconds * 1000);
  }
  const instant = new Date(value);
  return Number.isFinite(instant.getTime()) && instant.getTime() >= now.getTime()
    ? instant
    : now;
}
