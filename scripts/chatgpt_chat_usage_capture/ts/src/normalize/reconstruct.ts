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
    const promptGroup = promptEvidenceGroupFor(user, groups);
    for (const group of groups) {
      const attempt = buildAttempt(
        user,
        group,
        options,
        shouldUsePromptEvidence(user, group, groups, promptGroup),
      );
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
    const attempt = buildAttempt(null, group, options, false);
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
  usePromptEvidence: boolean,
): ReconstructedAttempt {
  const nodes = sortedNodes(group.nodes);
  const final = finalAnswer(nodes);
  const generationIds = uniqueStrings(nodes.map((node) => node.record.generationId));
  const requestIds = uniqueStrings(nodes.map((node) => node.record.requestId));
  const promptKey = user
    ? `${options.conversationId}:${user.messageId}:${group.branchRoot ?? group.identityKey}`
    : null;
  const aliases: Array<[string, string]> = [
    ...(promptKey ? [["prompt", promptKey] as [string, string]] : []),
    ...(group.branchRoot ? [["branch", `${options.conversationId}:${group.branchRoot}`] as [string, string]] : []),
    ...generationIds.map((id) => ["generation", `${options.conversationId}:${id}`] as [string, string]),
    ...requestIds.map((id) => ["request", `${options.conversationId}:${id}`] as [string, string]),
    ...nodes.map((node) => [
      "message",
      `${options.conversationId}:${node.record.messageId}`,
    ] as [string, string]),
  ];
  const identityBasis = group.identityBasis;
  const identityKey = `${options.conversationId}|${identityBasis}|${group.identityKey}`;
  const attemptId = stableId(scopeKey(options.scope), identityKey);
  const requestedModelRaw = usePromptEvidence ? user?.requestedModelRaw ?? null : null;
  const requestedModeRaw = usePromptEvidence ? user?.requestedModeRaw ?? null : null;
  const requestedReasoningEffortRaw = usePromptEvidence
    ? user?.requestedReasoningEffortRaw ?? null
    : null;
  const recordedFinalModelRaw = final?.record.recordedFinalModelRaw ?? null;
  const resolvedModelRaw = firstMetadataString(
    final?.record.metadata,
    [
      ...(usePromptEvidence && user ? [user.metadata] : []),
      ...nodes.map((node) => node.record.metadata),
    ],
    ["resolved_model", "resolved_model_slug"],
  );
  const timing = timingFor(user, nodes, final, usePromptEvidence);
  const requestedFamily = mapModelEvidence(
    {
      slug: requestedModelRaw,
      mode: requestedModeRaw,
      reasoningEffort: requestedReasoningEffortRaw,
    },
    options.mapping,
    options.scope.collectorAccountId,
    { at: timing.attemptTime },
  );
  const recordedFinalFamily = mapModelEvidence(
    {
      slug: recordedFinalModelRaw,
      mode: null,
      reasoningEffort: null,
    },
    options.mapping,
    options.scope.collectorAccountId,
    { at: timing.attemptTime },
  );
  const resolvedFamily = mapModelEvidence(
    {
      slug: resolvedModelRaw,
      mode: null,
      reasoningEffort: null,
    },
    options.mapping,
    options.scope.collectorAccountId,
    { at: timing.attemptTime },
  );
  const generationStarted = generationStartedFor(nodes);
  const outcome = outcomeFor(nodes, final, generationStarted);
  const surface = surfaceFor([...(user ? [user] : []), ...nodes.map((node) => node.record)]);
  const origin = originFor([...(user ? [user] : []), ...nodes.map((node) => node.record)]);
  const warnings: string[] = [];
  if (identityBasis === "unresolved") {
    warnings.push("unresolved_linkage");
  }
  if (group.identityBasis === "provisional") {
    warnings.push("provisional_identity");
  }
  if (user && !usePromptEvidence && hasGenerationSpecificIdentity(group)) {
    warnings.push("prompt_evidence_not_linked");
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
  const generationLinks = generationLinksFor(nodes, graph);
  const generationIds = uniqueStrings(nodes.map((node) => node.record.generationId));
  for (const generationId of generationIds) {
    const groupNodes = nodes.filter((node) => {
      if (assigned.has(node.key)) {
        return false;
      }
      if (node.record.generationId === generationId) {
        return true;
      }
      const linkedGenerations = generationLinks.get(node.key);
      return (
        node.record.generationId === null &&
        linkedGenerations?.size === 1 &&
        linkedGenerations.has(generationId)
      );
    });
    if (groupNodes.length > 0) {
      groupNodes.forEach((node) => assigned.add(node.key));
      groups.push({
        identityBasis: "generation",
        identityKey: generationId,
        nodes: groupNodes,
        branchRoot: singleString(groupNodes.map((node) => node.branchRoot)),
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
      identityBasis: "provisional",
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
  const groups: NodeGroup[] = [];
  const assigned = new Set<string>();
  const generationLinks = generationLinksFor(nodes, graph);
  for (const generationId of uniqueStrings(nodes.map((node) => node.record.generationId))) {
    const groupNodes = nodes.filter((node) => {
      if (assigned.has(node.key)) {
        return false;
      }
      if (node.record.generationId === generationId) {
        return true;
      }
      const linkedGenerations = generationLinks.get(node.key);
      return (
        node.record.generationId === null &&
        linkedGenerations?.size === 1 &&
        linkedGenerations.has(generationId)
      );
    });
    if (groupNodes.length > 0) {
      groupNodes.forEach((node) => assigned.add(node.key));
      groups.push({
        identityBasis: "generation",
        identityKey: generationId,
        nodes: groupNodes,
        branchRoot: singleString(groupNodes.map((node) => node.branchRoot)),
      });
    }
  }

  const requestGroups = new Map<string, GraphNode[]>();
  for (const node of nodes) {
    if (assigned.has(node.key) || node.record.requestId === null) {
      continue;
    }
    const key = `${node.branchRoot}|${node.record.requestId}`;
    const existing = requestGroups.get(key) ?? [];
    existing.push(node);
    requestGroups.set(key, existing);
  }
  for (const [key, seeds] of [...requestGroups.entries()].sort(([left], [right]) =>
    left.localeCompare(right),
  )) {
    const [branchRoot = "", requestId = ""] = key.split("|", 2);
    const groupNodes = expandIdentityGroup(
      seeds,
      nodes.filter((node) => !assigned.has(node.key)),
      graph,
      (node) =>
        node.record.requestId === requestId ||
        (node.record.requestId === null && node.branchRoot === branchRoot),
    );
    if (groupNodes.length === 0) {
      continue;
    }
    groupNodes.forEach((node) => assigned.add(node.key));
    groups.push({
      identityBasis: "request",
      identityKey: `${requestId}:${branchRoot}`,
      nodes: groupNodes,
      branchRoot,
    });
  }

  const remaining = new Map<string, GraphNode[]>();
  for (const node of nodes) {
    if (assigned.has(node.key)) {
      continue;
    }
    const existing = remaining.get(node.branchRoot) ?? [];
    existing.push(node);
    remaining.set(node.branchRoot, existing);
  }
  for (const [branchRoot, branchNodes] of [...remaining.entries()].sort(([left], [right]) =>
    left.localeCompare(right),
  )) {
    groups.push({
      identityBasis: "unresolved",
      identityKey: `unresolved:${branchRoot}:${branchNodes
        .map((node) => node.record.messageId)
        .sort()
        .join(",")}`,
      nodes: branchNodes,
      branchRoot,
    });
  }
  return groups.sort((left, right) => left.identityKey.localeCompare(right.identityKey));
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
    for (const candidate of graphNeighbors(node, candidates, graph)) {
      if (!output.has(candidate.key)) {
        queue.push(candidate);
      }
    }
    queue.sort((left, right) => left.key.localeCompare(right.key));
  }
  return sortedNodes([...output.values()]);
}

function generationLinksFor(
  nodes: GraphNode[],
  graph: { byKey: Map<string, GraphNode>; children: Map<string, Set<string>> },
): Map<string, Set<string>> {
  const links = new Map<string, Set<string>>();
  for (const node of nodes) {
    if (node.record.generationId !== null) {
      continue;
    }
    const generations = new Set<string>();
    const visited = new Set<string>();
    const queue = [node];
    while (queue.length > 0) {
      const current = queue.shift()!;
      if (visited.has(current.key)) {
        continue;
      }
      visited.add(current.key);
      if (current.record.generationId !== null) {
        generations.add(current.record.generationId);
        continue;
      }
      for (const candidate of graphNeighbors(current, nodes, graph)) {
        if (!visited.has(candidate.key)) {
          queue.push(candidate);
        }
      }
    }
    links.set(node.key, generations);
  }
  return links;
}

function graphNeighbors(
  node: GraphNode,
  candidates: GraphNode[],
  graph: { byKey: Map<string, GraphNode>; children: Map<string, Set<string>> },
): GraphNode[] {
  const candidateKeys = new Set(candidates.map((candidate) => candidate.key));
  const neighborKeys = new Set<string>();
  for (const child of graph.children.get(node.key) ?? []) {
    if (candidateKeys.has(child)) {
      neighborKeys.add(child);
    }
  }
  for (const candidate of candidates) {
    if (
      candidate.record.parentId === node.key ||
      candidate.record.parentId === node.record.messageId ||
      node.record.parentId === candidate.key ||
      node.record.parentId === candidate.record.messageId
    ) {
      neighborKeys.add(candidate.key);
    }
  }
  return [...neighborKeys]
    .map((key) => graph.byKey.get(key))
    .filter((candidate): candidate is GraphNode => candidate !== undefined)
    .sort((left, right) => left.key.localeCompare(right.key));
}

function promptEvidenceGroupFor(
  user: MessageRecord,
  groups: NodeGroup[],
): NodeGroup | null {
  const explicitGroups = groups.filter((group) =>
    group.nodes.some(
      (node) =>
        (user.generationId !== null && node.record.generationId === user.generationId) ||
        (user.requestId !== null && node.record.requestId === user.requestId),
    ),
  );
  if (user.generationId !== null || user.requestId !== null) {
    return explicitGroups[0] ?? null;
  }
  const generationGroups = groups.filter(hasGenerationSpecificIdentity);
  if (generationGroups.length === 0) {
    return groups[0] ?? null;
  }
  return [...generationGroups].sort(compareGroupEvidence)[0] ?? null;
}

function shouldUsePromptEvidence(
  user: MessageRecord,
  group: NodeGroup,
  groups: NodeGroup[],
  promptGroup: NodeGroup | null,
): boolean {
  if (user.generationId !== null || user.requestId !== null) {
    return group.nodes.some(
      (node) =>
        (user.generationId !== null && node.record.generationId === user.generationId) ||
        (user.requestId !== null && node.record.requestId === user.requestId),
    );
  }
  if (!hasGenerationSpecificIdentity(group)) {
    return true;
  }
  if (groups.filter(hasGenerationSpecificIdentity).length <= 1) {
    return true;
  }
  return group === promptGroup;
}

function hasGenerationSpecificIdentity(group: NodeGroup): boolean {
  return (
    group.identityBasis === "generation" ||
    group.identityBasis === "request" ||
    group.nodes.some(
      (node) => node.record.generationId !== null || node.record.requestId !== null,
    )
  );
}

function compareGroupEvidence(left: NodeGroup, right: NodeGroup): number {
  return (
    firstNodeTime(left).localeCompare(firstNodeTime(right)) ||
    left.identityKey.localeCompare(right.identityKey)
  );
}

function firstNodeTime(group: NodeGroup): string {
  return (
    sortedNodes(group.nodes)
      .map((node) => node.record.createdAt)
      .find((value): value is string => value !== null && validTime(value)) ?? ""
  );
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
    return "rejected_after_start";
  }
  if ([...statuses].some((status) => status === "cancelled" || status === "interrupted")) {
    return "cancelled_after_start";
  }
  if ([...statuses].some((status) => status === "error" || status === "failed")) {
    return "failed_after_start";
  }
  return "completion_unknown";
}

function generationStartedFor(nodes: GraphNode[]): boolean {
  return nodes.some((node) => {
    const role = (node.record.role ?? "").trim().toLowerCase();
    if (role === "tool") {
      return true;
    }
    if (role !== "assistant") {
      return false;
    }
    return !REJECTED_STATUSES.has((node.record.status ?? "").trim().toLowerCase());
  });
}

function timingFor(
  user: MessageRecord | null,
  nodes: GraphNode[],
  final: GraphNode | null,
  usePromptEvidence: boolean,
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
  if (usePromptEvidence && user?.createdAt && validTime(user.createdAt)) {
    return {
      attemptTime: user.createdAt,
      timeBasis: "user_message",
      earliestPossibleAt: user.createdAt,
      latestPossibleAt: final?.record.createdAt ?? times.at(-1) ?? user.createdAt,
    };
  }
  const preFinalTimes = nodes
    .filter((node) => node !== final)
    .map((node) => node.record.createdAt)
    .filter((value): value is string => value !== null && validTime(value))
    .sort();
  if (final !== null && preFinalTimes.length > 0 && times.length > 0) {
    return {
      attemptTime: null,
      timeBasis: "bounded_interval",
      earliestPossibleAt: preFinalTimes[0] ?? null,
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

function singleString(values: ReadonlyArray<string | null>): string | null {
  const unique = uniqueStrings(values);
  return unique.length === 1 ? unique[0] ?? null : null;
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
