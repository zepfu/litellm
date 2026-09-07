import {
  AuthenticationRequiredError,
  CapabilityError,
  HttpStatusError,
  LegacyFallbackNotApprovedError,
  RateLimitedError,
} from "../adapters/chatgpt/adapter.js";
import type {
  AcquiredConversation,
  DiscoveryCheckpoint,
  HistoryCollectionOptions,
  HistoryCollectionRequest,
  HistoryCollectionResult,
  HistoryCoverageResult,
  HistoryRange,
  HistoryReader,
  HistoryScope,
  RevisitEntry,
  RevisitReason,
  ScopeCoverageResult,
} from "../contracts/history.js";
import {
  DEFAULT_BACKFILL_DAYS,
  DEFAULT_INDEX_PAGE_SIZE,
  DEFAULT_MAX_INDEX_PAGES,
  DEFAULT_MAX_MESSAGE_PAGES_PER_CONVERSATION,
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
}

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
    const scanStartedAt = now.toISOString();
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
    const requestedRange = resolveRequestedRange(rangeOptions);
    const identity = await this.readIdentity();

    if (identity.authState !== "ready" || identity.surface !== "chat") {
      return blockedResult(
        this.options.accountId,
        request.mode,
        requestedRange,
        scanStartedAt,
        identity,
        "history collection requires a verified ready Chat identity",
      );
    }

    const candidates = new Map<string, Candidate>();
    const scopeResults: ScopeCoverageResult[] = [];
    const warnings: string[] = [];
    let pagesFetched = 0;

    for (const scope of ["active", "archived"] as const) {
      const discovery = await this.discoverScope(
        scope,
        request,
        requestedRange,
        now,
        scanStartedAt,
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
    for (const candidate of candidates.values()) {
      const acquisition = await this.acquireConversation(
        candidate,
        now,
        request,
      );
      detailPagesFetched += acquisition.detailPagesFetched;
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
    }

    const revisits = this.options.store.listRevisits();
    const coverage = this.buildCoverage(scopeResults, conversations, warnings);
    const status =
      coverage.overall === "complete" && revisits.length === 0
        ? "complete"
        : "partial";

    return {
      accountId: this.options.accountId,
      mode: request.mode,
      range: requestedRange,
      scanStartedAt,
      status,
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

  private async discoverScope(
    scope: HistoryScope,
    request: HistoryCollectionRequest,
    requestedRange: HistoryRange,
    now: Date,
    scanStartedAt: string,
  ): Promise<ScopeDiscoveryResult> {
    const previous = this.options.store.loadDiscovery(scope);
    const explicitRange = request.range !== undefined;
    const candidateCutoff = this.candidateCutoff(
      request.mode,
      requestedRange,
      previous,
      explicitRange,
      request.overlapMs ?? this.overlapMs,
    );
    const warnings: string[] = [];
    const summaries: ConversationSummary[] = [];
    const summaryIds = new Set<string>();
    const seenOffsets = new Set<number>();
    const pageBudget =
      request.maxIndexPagesPerScope ?? this.maxIndexPagesPerScope;
    const shouldResume =
      request.mode === "incremental" &&
      !explicitRange &&
      previous?.continuation !== null &&
      previous?.continuation !== undefined &&
      previous.status !== "complete";
    let offset = shouldResume ? previous!.continuation! : 0;
    let pagesFetched = 0;
    let status: ScopeDiscoveryResult["coverage"]["status"] = "in_progress";
    let continuation: number | null = offset;
    let paginationState: PaginationState = "unknown";
    let completeDiscoveryStartedAt = previous?.lastCompleteDiscoveryStartedAt ?? null;
    let leadingPage: AdaptedPage<ConversationSummary> | null = null;

    this.options.store.saveDiscovery(
      checkpointFor(
        this.options.accountId,
        scope,
        request.mode,
        requestedRange,
        candidateCutoff,
        scanStartedAt,
        offset,
        pagesFetched,
        pageBudget,
        previous?.lastCompleteDiscoveryStartedAt ?? null,
        "in_progress",
        "unknown",
        warnings,
        now.toISOString(),
      ),
    );

    while (pagesFetched < pageBudget) {
      if (seenOffsets.has(offset)) {
        warnings.push("repeated_index_offset");
        status = "partial";
        paginationState = "repeated_cursor";
        continuation = offset;
        break;
      }
      seenOffsets.add(offset);

      let page;
      try {
        page = await this.reader.listConversations({
          archived: scope === "archived",
          offset,
          limit: request.indexPageSize ?? this.indexPageSize,
          order: "updated",
        });
      } catch (error) {
        warnings.push(`index_${errorCode(error)}`);
        status = "partial";
        paginationState = "unknown";
        continuation = offset;
        break;
      }

      pagesFetched += 1;
      paginationState = page.paginationState;
      warnings.push(...page.warnings);
      if (offset === 0 && leadingPage === null) {
        leadingPage = page;
      }
      for (const summary of page.items) {
        if (summary.updatedAt === null) {
          warnings.push("conversation_missing_update_time");
        }
        if (
          isCandidateSummary(summary, candidateCutoff, requestedRange.end) &&
          !summaryIds.has(summary.conversationId)
        ) {
          summaries.push(summary);
          summaryIds.add(summary.conversationId);
        }
      }
      if (
        page.paginationState === "unknown" ||
        page.paginationState === "contradictory" ||
        page.paginationState === "repeated_cursor"
      ) {
        status = "partial";
        continuation =
          typeof page.continuation === "number" ? page.continuation : offset;
        warnings.push(`index_${page.paginationState}`);
        break;
      }
      if (page.exhausted && page.paginationState === "complete") {
        status = "complete";
        continuation = null;
        completeDiscoveryStartedAt = scanStartedAt;
        this.options.store.saveDiscovery(
          checkpointFor(
            this.options.accountId,
            scope,
            request.mode,
            requestedRange,
            candidateCutoff,
            scanStartedAt,
            null,
            pagesFetched,
            pageBudget,
            completeDiscoveryStartedAt,
            status,
            paginationState,
            warnings,
            now.toISOString(),
          ),
        );
        break;
      }
      if (
        page.paginationState !== "continuation" ||
        typeof page.continuation !== "number" ||
        page.continuation <= offset
      ) {
        status = "partial";
        continuation =
          typeof page.continuation === "number" ? page.continuation : offset;
        paginationState =
          page.paginationState === "continuation"
            ? "repeated_cursor"
            : "unknown";
        warnings.push("nonadvancing_index_continuation");
        break;
      }

      continuation = page.continuation;
      if (pagesFetched >= pageBudget) {
        status = "partial";
        paginationState = "budget_exhausted";
        warnings.push("index_page_budget_exhausted");
        break;
      }

      this.options.store.saveDiscovery(
        checkpointFor(
          this.options.accountId,
          scope,
          request.mode,
          requestedRange,
          candidateCutoff,
          scanStartedAt,
          continuation,
          pagesFetched,
          pageBudget,
          previous?.lastCompleteDiscoveryStartedAt ?? null,
          "in_progress",
          paginationState,
          warnings,
          now.toISOString(),
        ),
      );
      offset = page.continuation;
    }

    if (status === "complete") {
      let reread: AdaptedPage<ConversationSummary>;
      try {
        reread = await this.reader.listConversations({
          archived: scope === "archived",
          offset: 0,
          limit: request.indexPageSize ?? this.indexPageSize,
          order: "updated",
        });
        pagesFetched += 1;
        warnings.push(...reread.warnings);
        if (
          leadingPage === null ||
          indexPagesDiffer(leadingPage, reread)
        ) {
          status = "partial";
          completeDiscoveryStartedAt =
            previous?.lastCompleteDiscoveryStartedAt ?? null;
          paginationState = "continuation";
          continuation = 0;
          warnings.push(
            leadingPage === null
              ? "leading_index_baseline_unavailable"
              : "leading_index_changed_during_scan",
          );
          for (const summary of reread.items) {
            if (
              summary.updatedAt === null ||
              isCandidateSummary(summary, candidateCutoff, requestedRange.end)
            ) {
              if (summary.updatedAt === null) {
                warnings.push("conversation_missing_update_time");
              }
              if (!summaryIds.has(summary.conversationId)) {
                summaries.push(summary);
                summaryIds.add(summary.conversationId);
              }
            }
          }
        }
      } catch (error) {
        status = "partial";
        completeDiscoveryStartedAt =
          previous?.lastCompleteDiscoveryStartedAt ?? null;
        paginationState = "unknown";
        continuation = 0;
        warnings.push(`leading_index_reread_${errorCode(error)}`);
      }
    }

    if (status === "in_progress") {
      status = "partial";
      paginationState = "budget_exhausted";
      warnings.push("index_page_budget_exhausted");
    }
    if (status !== "complete") {
      this.options.store.saveDiscovery(
        checkpointFor(
          this.options.accountId,
          scope,
          request.mode,
          requestedRange,
          candidateCutoff,
          scanStartedAt,
          continuation,
          pagesFetched,
          pageBudget,
          previous?.lastCompleteDiscoveryStartedAt ?? null,
          status,
          paginationState,
          warnings,
          now.toISOString(),
        ),
      );
    }

    const coverage =
      pagesFetched === 0
        ? "unknown"
        : status === "complete" && warnings.length === 0
          ? "complete"
          : "partial";
    return {
      scope,
      summaries,
      coverage: {
        scope,
        status,
        coverage,
        pagesFetched,
        candidates: summaries.length,
        continuation,
        paginationState,
        candidateCutoff,
        warnings: [...new Set(warnings)],
      },
      warnings,
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
    const overlapStart = new Date(
      new Date(previousWatermark).getTime() - overlapMs,
    );
    const requestedStart = new Date(requestedRange.start);
    const end = new Date(requestedRange.end);
    const selected = new Date(Math.max(overlapStart.getTime(), requestedStart.getTime()));
    return new Date(Math.min(selected.getTime(), end.getTime())).toISOString();
  }

  private async acquireConversation(
    candidate: Candidate,
    now: Date,
    request: HistoryCollectionRequest,
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
      const revisit = this.saveRevisit(
        candidate,
        now,
        classifyRevisitReason(error),
        0,
        errorCode(error),
        existingRevisit,
      );
      return {
        detail: null,
        messages: [],
        revisit,
        coverage: "partial",
        warnings: [`detail_${errorCode(error)}`],
        detailPagesFetched: 0,
      };
    }

    const messages = dedupeMessages(detail.messages);
    if (detail.detailRoute === "legacy") {
      this.options.store.completeRevisit(
        this.options.accountId,
        candidate.summary.conversationId,
      );
      return {
        detail,
        messages,
        revisit: null,
        coverage: detail.coverage === "validated_page" ? "complete" : "partial",
        warnings: detail.warnings,
        detailPagesFetched: 0,
      };
    }

    const pageResult = await this.fetchModernMessages(
      candidate,
      detail,
      messages,
      now,
      existingRevisit,
      request.maxMessagePagesPerConversation ??
        this.maxMessagePagesPerConversation,
    );
    return {
      detail,
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
  ): Promise<{
    messages: MessageRecord[];
    revisit: RevisitEntry | null;
    warnings: string[];
    detailPagesFetched: number;
  }> {
    const messages = [...initialMessages];
    const warnings: string[] = [];
    const seenCursors = new Set<string>();
    let before: string | null = null;
    let detailPagesFetched = 0;
    let reason: RevisitReason | null = null;

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
        reason = classifyRevisitReason(error);
        warnings.push(`messages_${errorCode(error)}`);
        break;
      }
      detailPagesFetched += 1;
      messages.push(...page.items);
      warnings.push(...page.warnings);
      if (page.paginationState === "complete" && page.exhausted) {
        this.options.store.completeRevisit(
          this.options.accountId,
          candidate.summary.conversationId,
        );
        return {
          messages: dedupeMessages(messages),
          revisit: null,
          warnings,
          detailPagesFetched,
        };
      }
      if (
        page.paginationState === "repeated_cursor" ||
        page.paginationState === "contradictory"
      ) {
        reason =
          page.paginationState === "repeated_cursor"
            ? "repeated_cursor"
            : "contradictory_pagination";
        warnings.push(`messages_${page.paginationState}`);
        break;
      }
      if (
        page.paginationState !== "continuation" ||
        typeof page.continuation !== "string" ||
        !page.continuation.trim()
      ) {
        reason = "unknown_pagination";
        warnings.push("messages_unknown_pagination");
        break;
      }
      const cursor = page.continuation.trim();
      if (seenCursors.has(cursor) || cursor === before) {
        reason = "repeated_cursor";
        warnings.push("messages_repeated_cursor");
        break;
      }
      seenCursors.add(cursor);
      before = cursor;
    }

    if (!reason) {
      reason = "page_budget";
      warnings.push("message_page_budget_exhausted");
    }
    const revisit = this.saveRevisit(
      candidate,
      now,
      reason,
      detailPagesFetched,
      null,
      existingRevisit,
    );
    return {
      messages: dedupeMessages(messages),
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
  ): RevisitEntry {
    const nowIso = now.toISOString();
    const entry: RevisitEntry = {
      stateVersion: HISTORY_STATE_VERSION,
      accountId: this.options.accountId,
      conversationId: candidate.summary.conversationId,
      scopes: [...candidate.scopes].sort(),
      status: "pending",
      reason,
      firstSeenAt: existing?.firstSeenAt ?? nowIso,
      lastSeenAt: nowIso,
      attempts: (existing?.attempts ?? 0) + 1,
      nextEligibleAt: nowIso,
      lastError,
      detailPagesFetched,
    };
    this.options.store.upsertRevisit(entry);
    return entry;
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
  };
}

function isCandidateSummary(
  summary: ConversationSummary,
  candidateCutoff: string,
  rangeEnd: string,
): boolean {
  if (summary.updatedAt === null) {
    return true;
  }
  const updatedAt = new Date(summary.updatedAt);
  const start = new Date(candidateCutoff);
  const end = new Date(rangeEnd);
  if (
    !Number.isFinite(updatedAt.getTime()) ||
    !Number.isFinite(start.getTime()) ||
    !Number.isFinite(end.getTime())
  ) {
    return true;
  }
  return updatedAt.getTime() >= start.getTime() && updatedAt.getTime() < end.getTime();
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
  };
}

function blockedResult(
  accountId: string,
  mode: HistoryCollectionRequest["mode"],
  range: HistoryRange,
  scanStartedAt: string,
  identity: IdentityRecord,
  warning: string,
): HistoryCollectionResult {
  const active = unknownScopeCoverage("active");
  const archived = unknownScopeCoverage("archived");
  return {
    accountId,
    mode,
    range,
    scanStartedAt,
    status: "blocked",
    identity,
    scopes: [active, archived],
    conversations: [],
    revisits: [],
    coverage: {
      active,
      archived,
      projects: "unknown",
      branches: "unknown",
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
