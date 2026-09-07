import type {
  AttemptRecord,
  MessageRecord,
} from "../contracts/records.js";
import { scopeKey, stableId } from "../ledger/identity.js";
import type {
  LedgerScope,
  MappingEvidence,
  ModelMappingVersion,
  ReconstructedAttempt,
} from "../ledger/types.js";
import { mapModelEvidence } from "./model-mapping.js";

const NON_ANSWER_CHANNELS = new Set(["analysis", "reasoning", "commentary", "tool"]);
const NONTERMINAL_STATUSES = new Set([
  "in_progress",
  "streaming",
  "generating",
  "pending",
  "incomplete",
]);
const FAILED_STATUSES = new Set(["error", "failed", "cancelled", "interrupted"]);
const REJECTED_STATUSES = new Set(["rejected", "moderation_blocked"]);
const TERMINAL_STATUSES = new Set([
  "completed",
  "finished_successfully",
  "success",
  "done",
]);

interface GraphNode {
  key: string;
  record: MessageRecord;
  branchRoot: string;
}

interface NodeGroup {
  identityBasis: string;
  identityKey: string;
  nodes: GraphNode[];
  branchRoot: string | null;
}

export function reconstructAttempts(
  messages: ReadonlyArray<MessageRecord>,
  options: {
    scope: LedgerScope;
    conversationId: string;
    mapping: ModelMappingVersion;
  },
): ReconstructedAttempt[] {
  const normalized = deduplicateMessages(messages);
  const graph = buildGraph(normalized);
  const attempts: ReconstructedAttempt[] = [];
  const claimed = new Set<string>();

  for (const user of sortedMessages(normalized).filter((item) => item.role === "user")) {
    const descendants = descendantsForUser(user, graph);
    const groups = groupsForUser(user, descendants, graph);
    if (groups.length === 0) {
      groups.push({
        identityBasis: "provisional",
        identityKey: `prompt:${user.messageId}`,
        nodes: [],
        branchRoot: null,
      });
    }
    for (const group of groups) {
      const attempt = buildAttempt(user, group, options);
      if (!claimed.has(attempt.attemptId)) {
        claimed.add(attempt.attemptId);
        attempts.push(attempt);
      }
    }
  }

  const orphanNodes = normalized
    .filter(
      (item) =>
        (item.role === "assistant" || item.role === "tool") &&
        !claimedMessageIds(attempts).has(item.messageId),
    )
    .map((item) => graph.byKey.get(item.nodeId ?? item.messageId))
    .filter((item): item is GraphNode => item !== undefined);
  for (const group of groupsForOrphans(orphanNodes, graph)) {
    const attempt = buildAttempt(null, group, options);
    if (!claimed.has(attempt.attemptId)) {
      claimed.add(attempt.attemptId);
      attempts.push(attempt);
    }
  }

  return attempts.sort(compareAttempts);
}

function buildAttempt(
  user: MessageRecord | null,
  group: NodeGroup,
  options: {
    scope: LedgerScope;
    conversationId: string;
    mapping: ModelMappingVersion;
  },
): ReconstructedAttempt {
  const nodes = sortedNodes(group.nodes);
  const final = finalAnswer(nodes);
  const generationIds = uniqueStrings(nodes.map((node) => node.record.generationId));
  const requestIds = uniqueStrings(nodes.map((node) => node.record.requestId));
  const promptKey = user
    ? `${user.messageId}:${group.identityKey}`
    : `orphan:${group.identityKey}`;
  const aliases: Array<[string, string]> = [
    ["prompt", promptKey],
    ...(group.branchRoot ? [["branch", `${options.conversationId}:${group.branchRoot}`] as [string, string]] : []),
    ...generationIds.map((id) => ["generation", `${options.conversationId}:${id}`] as [string, string]),
    ...requestIds.map((id) => ["request", `${options.conversationId}:${id}`] as [string, string]),
  ];
  const identityBasis = group.identityBasis;
  const identityKey = `${options.conversationId}|${identityBasis}|${group.identityKey}`;
  const attemptId = stableId(scopeKey(options.scope), identityKey);
  const requestedModelRaw = user?.requestedModelRaw ?? null;
  const requestedModeRaw = user?.requestedModeRaw ?? null;
  const requestedReasoningEffortRaw = user?.requestedReasoningEffortRaw ?? null;
  const recordedFinalModelRaw = final?.record.recordedFinalModelRaw ?? null;
  const resolvedModelRaw = firstMetadataString(
    final?.record.metadata,
    [
      ...(user ? [user.metadata] : []),
      ...nodes.map((node) => node.record.metadata),
    ],
    ["resolved_model", "resolved_model_slug"],
  );
  const requestedFamily = mapModelEvidence(
    {
      slug: requestedModelRaw,
      mode: requestedModeRaw,
      reasoningEffort: requestedReasoningEffortRaw,
    },
    options.mapping,
    options.scope.collectorAccountId,
  );
  const recordedFinalFamily = mapModelEvidence(
    {
      slug: recordedFinalModelRaw,
      mode: null,
      reasoningEffort: null,
    },
    options.mapping,
    options.scope.collectorAccountId,
  );
  const resolvedFamily = mapModelEvidence(
    {
      slug: resolvedModelRaw,
      mode: null,
      reasoningEffort: null,
    },
    options.mapping,
    options.scope.collectorAccountId,
  );
  const generationStarted = nodes.some(
    (node) => node.record.role === "assistant" || node.record.role === "tool",
  );
  const outcome = outcomeFor(nodes, final, generationStarted);
  const timing = timingFor(user, nodes, final);
  const surface = surfaceFor([...(user ? [user] : []), ...nodes.map((node) => node.record)]);
  const origin = originFor([...(user ? [user] : []), ...nodes.map((node) => node.record)]);
  const warnings: string[] = [];
  if (identityBasis === "unresolved") {
    warnings.push("unresolved_linkage");
  }
  if (group.identityBasis === "provisional") {
    warnings.push("provisional_identity");
  }
  if (!final && generationStarted) {
    warnings.push("terminal_answer_not_observed");
  }
  if (surface !== "chat") {
    warnings.push(`surface:${surface}`);
  }
  if (origin !== null && ["shared", "imported", "copied"].includes(origin)) {
    warnings.push(`origin:${origin}`);
  }

  const record: AttemptRecord = {
    attemptId,
    conversationId: options.conversationId,
    identityBasis,
    timeBasis: timing.timeBasis,
    attemptTime: timing.attemptTime,
    earliestPossibleAt: timing.earliestPossibleAt,
    latestPossibleAt: timing.latestPossibleAt,
    requestedModelRaw,
    requestedModeRaw,
    requestedReasoningEffortRaw,
    recordedFinalModelRaw,
    resolvedModelRaw,
    requestedFamily,
    recordedFinalFamily,
    resolvedFamily,
    mappingVersion: options.mapping.version,
    outcome,
    completedAnswer: final !== null,
    generationStarted,
    surface,
    origin,
    aliases,
    evidenceMessageIds: [
      ...(user ? [user.messageId] : []),
      ...nodes.map((node) => node.record.messageId),
    ].sort(),
    revision: 1,
    warnings,
  };
  return { ...record, scope: options.scope };
}

function buildGraph(messages: ReadonlyArray<MessageRecord>): {
  byKey: Map<string, GraphNode>;
  children: Map<string, Set<string>>;
} {
  const byKey = new Map<string, GraphNode>();
  for (const record of messages) {
    const key = record.nodeId ?? record.messageId;
    byKey.set(key, { key, record, branchRoot: key });
    byKey.set(record.messageId, { key, record, branchRoot: key });
  }
  const children = new Map<string, Set<string>>();
  for (const node of new Set(byKey.values())) {
    const links = new Set<string>();
    for (const child of node.record.children) {
      const childNode = byKey.get(child);
      if (childNode) {
        links.add(childNode.key);
      }
    }
    for (const candidate of new Set(byKey.values())) {
      if (
        candidate.record.parentId !== null &&
        (candidate.record.parentId === node.key ||
          candidate.record.parentId === node.record.messageId)
      ) {
        links.add(candidate.key);
      }
    }
    children.set(node.key, links);
  }
  return { byKey, children };
}

function descendantsForUser(
  user: MessageRecord,
  graph: { byKey: Map<string, GraphNode>; children: Map<string, Set<string>> },
): GraphNode[] {
  const userNode = graph.byKey.get(user.nodeId ?? user.messageId);
  if (!userNode) {
    return [];
  }
  const roots = graph.children.get(userNode.key) ?? new Set<string>();
  if (roots.size === 0) {
    for (const candidate of new Set(graph.byKey.values())) {
      if (
        candidate.record.role !== "user" &&
        candidate.record.parentId !== null &&
        (candidate.record.parentId === userNode.key ||
          candidate.record.parentId === user.messageId)
      ) {
        roots.add(candidate.key);
      }
    }
  }
  const visited = new Set<string>();
  const output: GraphNode[] = [];
  const queue = [...roots].sort();
  while (queue.length > 0) {
    const key = queue.shift()!;
    if (visited.has(key)) {
      continue;
    }
    visited.add(key);
    const node = graph.byKey.get(key);
    if (!node || node.record.role === "user") {
      continue;
    }
    output.push(node);
    for (const child of graph.children.get(node.key) ?? []) {
      if (!visited.has(child)) {
        queue.push(child);
      }
    }
    queue.sort();
  }
  const branchRoots = new Map<string, string>();
  for (const node of output) {
    branchRoots.set(node.key, roots.has(node.key) ? node.key : branchRootFor(node, output, roots));
  }
  return output.map((node) => ({
    ...node,
    branchRoot: branchRoots.get(node.key) ?? node.key,
  }));
}

function groupsForUser(
  user: MessageRecord,
  nodes: GraphNode[],
  graph: { byKey: Map<string, GraphNode>; children: Map<string, Set<string>> },
): NodeGroup[] {
  if (nodes.length === 0) {
    return [];
  }
  const groups: NodeGroup[] = [];
  const assigned = new Set<string>();
  const generationIds = uniqueStrings(nodes.map((node) => node.record.generationId));
  for (const generationId of generationIds) {
    const seeds = nodes.filter((node) => node.record.generationId === generationId);
    const groupNodes = expandIdentityGroup(seeds, nodes, graph, (node) => {
      const value = node.record.generationId;
      return value === generationId || (value === null && connectedToGeneration(node, generationId, nodes, graph));
    });
    if (groupNodes.length > 0) {
      groupNodes.forEach((node) => assigned.add(node.key));
      groups.push({
        identityBasis: "generation",
        identityKey: generationId,
        nodes: groupNodes,
        branchRoot: uniqueStrings(groupNodes.map((node) => node.branchRoot))[0] ?? null,
      });
    }
  }

  const requestBranches = new Map<string, GraphNode[]>();
  for (const node of nodes) {
    if (assigned.has(node.key) || node.record.requestId === null) {
      continue;
    }
    const key = `${node.branchRoot}|${node.record.requestId}`;
    const existing = requestBranches.get(key) ?? [];
    existing.push(node);
    requestBranches.set(key, existing);
  }
  for (const [key, seeds] of [...requestBranches.entries()].sort(([left], [right]) => left.localeCompare(right))) {
    const parts = key.split("|", 2);
    const branchRoot = parts[0] ?? "";
    const requestId = parts[1] ?? "";
    const groupNodes = expandIdentityGroup(seeds, nodes.filter((node) => !assigned.has(node.key)), graph, (node) =>
      node.record.requestId === requestId ||
      (node.record.requestId === null && node.branchRoot === branchRoot),
    );
    groupNodes.forEach((node) => assigned.add(node.key));
    groups.push({
      identityBasis: "request",
      identityKey: `${requestId}:${branchRoot}`,
      nodes: groupNodes,
      branchRoot,
    });
  }

  const remaining = nodes.filter((node) => !assigned.has(node.key));
  const byBranch = new Map<string, GraphNode[]>();
  for (const node of remaining) {
    const existing = byBranch.get(node.branchRoot) ?? [];
    existing.push(node);
    byBranch.set(node.branchRoot, existing);
  }
  for (const [branchRoot, branchNodes] of [...byBranch.entries()].sort(([left], [right]) => left.localeCompare(right))) {
    groups.push({
      identityBasis: branchNodes.some((node) => node.record.generationId || node.record.requestId)
        ? "provisional"
        : "provisional",
      identityKey: `${user.messageId}:${branchRoot}:${branchNodes.map((node) => node.record.messageId).sort().join(",")}`,
      nodes: branchNodes,
      branchRoot,
    });
  }
  return groups.sort((left, right) => left.identityKey.localeCompare(right.identityKey));
}

function groupsForOrphans(
  nodes: GraphNode[],
  graph: { byKey: Map<string, GraphNode>; children: Map<string, Set<string>> },
): NodeGroup[] {
  const byIdentity = new Map<string, GraphNode[]>();
  for (const node of nodes) {
    const key = node.record.generationId
      ? `generation:${node.record.generationId}`
      : node.record.requestId
        ? `request:${node.record.requestId}:${node.branchRoot}`
        : `unresolved:${node.branchRoot}`;
    const existing = byIdentity.get(key) ?? [];
    existing.push(node);
    byIdentity.set(key, existing);
  }
  return [...byIdentity.entries()]
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([key, seeds]) => ({
      identityBasis: key.startsWith("generation:")
        ? "generation"
        : key.startsWith("request:")
          ? "request"
          : "unresolved",
      identityKey: key,
      nodes: expandIdentityGroup(seeds, nodes, graph, () => true),
      branchRoot: seeds[0]?.branchRoot ?? null,
    }));
}

function expandIdentityGroup(
  seeds: GraphNode[],
  candidates: GraphNode[],
  graph: { byKey: Map<string, GraphNode>; children: Map<string, Set<string>> },
  accepts: (node: GraphNode) => boolean,
): GraphNode[] {
  const candidateKeys = new Set(candidates.map((node) => node.key));
  const output = new Map<string, GraphNode>();
  const queue = [...seeds].sort((left, right) => left.key.localeCompare(right.key));
  while (queue.length > 0) {
    const node = queue.shift()!;
    if (output.has(node.key) || !candidateKeys.has(node.key) || !accepts(node)) {
      continue;
    }
    output.set(node.key, node);
    const neighbors = new Set<string>([
      ...(graph.children.get(node.key) ?? []),
      ...candidates
        .filter(
          (candidate) =>
            candidate.record.parentId === node.key ||
            candidate.record.parentId === node.record.messageId,
        )
        .map((candidate) => candidate.key),
    ]);
    for (const neighbor of neighbors) {
      const candidate = graph.byKey.get(neighbor);
      if (candidate && !output.has(candidate.key)) {
        queue.push(candidate);
      }
    }
    queue.sort((left, right) => left.key.localeCompare(right.key));
  }
  return sortedNodes([...output.values()]);
}

function connectedToGeneration(
  node: GraphNode,
  generationId: string,
  nodes: GraphNode[],
  graph: { byKey: Map<string, GraphNode>; children: Map<string, Set<string>> },
): boolean {
  const neighboringIds = new Set<string>();
  for (const candidate of nodes) {
    if (
      candidate.record.generationId &&
      (candidate.record.parentId === node.key ||
        candidate.record.parentId === node.record.messageId ||
        (graph.children.get(node.key) ?? new Set()).has(candidate.key) ||
        (graph.children.get(candidate.key) ?? new Set()).has(node.key))
    ) {
      neighboringIds.add(candidate.record.generationId);
    }
  }
  return neighboringIds.size === 1 && neighboringIds.has(generationId);
}

function finalAnswer(nodes: GraphNode[]): GraphNode | null {
  const candidates = nodes.filter((node) => {
    const record = node.record;
    return (
      record.role === "assistant" &&
      !NON_ANSWER_CHANNELS.has((record.channel ?? "").toLowerCase()) &&
      record.endTurn === true &&
      isTerminalSuccess(record.status)
    );
  });
  return candidates.sort(compareNodes).at(-1) ?? null;
}

function isTerminalSuccess(status: string | null): boolean {
  const normalized = (status ?? "").trim().toLowerCase();
  if (NONTERMINAL_STATUSES.has(normalized) || FAILED_STATUSES.has(normalized) || REJECTED_STATUSES.has(normalized)) {
    return false;
  }
  return normalized === "" || TERMINAL_STATUSES.has(normalized);
}

function outcomeFor(
  nodes: GraphNode[],
  final: GraphNode | null,
  generationStarted: boolean,
): string {
  const statuses = new Set(
    nodes
      .map((node) => (node.record.status ?? "").trim().toLowerCase())
      .filter(Boolean),
  );
  if (final) {
    return "completed";
  }
  if (!generationStarted) {
    return statuses.has("rejected") || statuses.has("moderation_blocked")
      ? "rejected_before_start"
      : "unresolved";
  }
  if ([...statuses].some((status) => REJECTED_STATUSES.has(status))) {
    return "rejected_before_start";
  }
  if ([...statuses].some((status) => status === "cancelled" || status === "interrupted")) {
    return "cancelled_after_start";
  }
  if ([...statuses].some((status) => status === "error" || status === "failed")) {
    return "failed_after_start";
  }
  return "completion_unknown";
}

function timingFor(
  user: MessageRecord | null,
  nodes: GraphNode[],
  final: GraphNode | null,
): {
  attemptTime: string | null;
  timeBasis: string;
  earliestPossibleAt: string | null;
  latestPossibleAt: string | null;
} {
  const times = nodes
    .map((node) => node.record.createdAt)
    .filter((value): value is string => value !== null && validTime(value))
    .sort();
  if (user?.createdAt && validTime(user.createdAt)) {
    return {
      attemptTime: user.createdAt,
      timeBasis: "user_message",
      earliestPossibleAt: user.createdAt,
      latestPossibleAt: final?.record.createdAt ?? times.at(-1) ?? user.createdAt,
    };
  }
  if (times.length > 0) {
    return {
      attemptTime: null,
      timeBasis: "bounded_interval",
      earliestPossibleAt: times[0] ?? null,
      latestPossibleAt: times.at(-1) ?? null,
    };
  }
  return {
    attemptTime: null,
    timeBasis: "unknown",
    earliestPossibleAt: null,
    latestPossibleAt: null,
  };
}

function surfaceFor(records: MessageRecord[]): MessageRecord["surface"] {
  const concrete = uniqueStrings(records.map((record) => record.surface).filter((value) => value !== "unknown"));
  if (concrete.length === 0) {
    return "unknown";
  }
  if (concrete.length === 1) {
    return (concrete[0] ?? "unknown") as MessageRecord["surface"];
  }
  return "unknown";
}

function originFor(records: MessageRecord[]): string | null {
  const values = new Set(records.map((record) => record.origin).filter((value): value is string => value !== null));
  if (values.has("imported")) {
    return "imported";
  }
  if (values.has("copied")) {
    return "copied";
  }
  if (values.has("shared") || values.has("true")) {
    return "shared";
  }
  return [...values][0] ?? null;
}

function firstMetadataString(
  finalMetadata: Record<string, unknown> | undefined,
  metadata: Array<Record<string, unknown>>,
  keys: string[],
): string | null {
  const values = [
    ...(finalMetadata ? [finalMetadata] : []),
    ...metadata,
  ];
  for (const item of values) {
    for (const key of keys) {
      const value = item[key];
      if (typeof value === "string" && value.trim()) {
        return value.trim();
      }
    }
  }
  return null;
}

function branchRootFor(node: GraphNode, nodes: GraphNode[], roots: Set<string>): string {
  let current = node;
  const visited = new Set<string>();
  while (!visited.has(current.key)) {
    visited.add(current.key);
    if (roots.has(current.key)) {
      return current.key;
    }
    const parent = nodes.find(
      (candidate) =>
        candidate.key === current.record.parentId ||
        candidate.record.messageId === current.record.parentId,
    );
    if (!parent) {
      return current.key;
    }
    current = parent;
  }
  return node.key;
}

function deduplicateMessages(messages: ReadonlyArray<MessageRecord>): MessageRecord[] {
  const byId = new Map<string, MessageRecord>();
  for (const message of messages) {
    byId.set(message.messageId, message);
  }
  return [...byId.values()];
}

function claimedMessageIds(attempts: ReadonlyArray<AttemptRecord>): Set<string> {
  return new Set(attempts.flatMap((attempt) => attempt.evidenceMessageIds));
}

function uniqueStrings(values: ReadonlyArray<string | null>): string[] {
  return [...new Set(values.filter((value): value is string => value !== null && value !== ""))].sort();
}

function sortedMessages(messages: ReadonlyArray<MessageRecord>): MessageRecord[] {
  return [...messages].sort(compareMessages);
}

function sortedNodes(nodes: ReadonlyArray<GraphNode>): GraphNode[] {
  return [...nodes].sort(compareNodes);
}

function compareAttempts(left: AttemptRecord, right: AttemptRecord): number {
  return (
    (left.attemptTime ?? left.earliestPossibleAt ?? "").localeCompare(
      right.attemptTime ?? right.earliestPossibleAt ?? "",
    ) || left.attemptId.localeCompare(right.attemptId)
  );
}

function compareMessages(left: MessageRecord, right: MessageRecord): number {
  return (
    (left.createdAt ?? "").localeCompare(right.createdAt ?? "") ||
    left.messageId.localeCompare(right.messageId)
  );
}

function compareNodes(left: GraphNode, right: GraphNode): number {
  return compareMessages(left.record, right.record);
}

function validTime(value: string): boolean {
  return Number.isFinite(Date.parse(value));
}

export type { MappingEvidence };
