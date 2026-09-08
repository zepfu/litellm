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

interface AttemptGraph {
  byKey: Map<string, GraphNode>;
  children: Map<string, Set<string>>;
  parents: Map<string, Set<string>>;
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
  const claimedEvidenceMessageIds = new Set<string>();

  for (const user of sortedMessages(normalized).filter((item) => item.role === "user")) {
    const descendants = descendantsForUser(user, graph, claimedEvidenceMessageIds);
    const groups = groupsForUser(user, descendants, graph);
    if (groups.length === 0) {
      groups.push({
        identityBasis: "provisional",
        identityKey: `prompt:${user.messageId}`,
        nodes: [],
        branchRoot: null,
      });
    }
    const promptGroup = promptEvidenceGroupFor(user, groups, graph);
    for (const group of groups) {
      const attempt = buildAttempt(
        user,
        group,
        options,
        shouldUsePromptEvidence(group, promptGroup),
      );
      for (const node of group.nodes) {
        claimedEvidenceMessageIds.add(node.record.messageId);
      }
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

function buildGraph(messages: ReadonlyArray<MessageRecord>): AttemptGraph {
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
  const parents = new Map<string, Set<string>>();
  for (const [parentKey, childKeys] of children) {
    for (const childKey of childKeys) {
      const linkedParents = parents.get(childKey) ?? new Set<string>();
      linkedParents.add(parentKey);
      parents.set(childKey, linkedParents);
    }
  }
  return { byKey, children, parents };
}

function descendantsForUser(
  user: MessageRecord,
  graph: AttemptGraph,
  claimedEvidenceMessageIds: ReadonlySet<string>,
): GraphNode[] {
  const userNode = graph.byKey.get(user.nodeId ?? user.messageId);
  if (!userNode) {
    return [];
  }
  const candidates = graphNodes(graph).filter(
    (node) =>
      node.key !== userNode.key &&
      !claimedEvidenceMessageIds.has(node.record.messageId),
  );
  const roots = new Set(
    childNeighbors(userNode, candidates, graph)
      .filter((node) => node.record.role !== "user")
      .map((node) => node.key),
  );
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
    for (const neighbor of childNeighbors(node, candidates, graph)) {
      if (neighbor.record.role !== "user" && !visited.has(neighbor.key)) {
        queue.push(neighbor.key);
      }
    }
    queue.sort();
  }
  const branchRoots = new Map<string, string>();
  for (const node of output) {
    branchRoots.set(
      node.key,
      roots.has(node.key) ? node.key : branchRootFor(node, output, roots, graph),
    );
  }
  return output.map((node) => ({
    ...node,
    branchRoot: branchRoots.get(node.key) ?? node.key,
  }));
}

function groupsForUser(
  user: MessageRecord,
  nodes: GraphNode[],
  graph: AttemptGraph,
): NodeGroup[] {
  if (nodes.length === 0) {
    return [];
  }
  const groups: NodeGroup[] = [];
  const assigned = new Set<string>();
  const generationLinks = generationLinksFor(nodes, graph);
  // Generation IDs own identity; a reused request ID cannot absorb an anchor.
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

  const requestCandidates = nodes.filter(
    (node) => !assigned.has(node.key) && (generationLinks.get(node.key)?.size ?? 0) <= 1,
  );
  const requestBranches = requestGroupsFor(requestCandidates);
  for (const seeds of requestBranches) {
    const { branchRoot, record: { requestId } } = seeds[0]!;
    const groupNodes = expandIdentityGroup(
      seeds,
      requestCandidates.filter((node) => !assigned.has(node.key)),
      graph,
      (node) =>
        node.branchRoot === branchRoot &&
        (node.record.requestId === requestId || node.record.requestId === null),
    );
    groupNodes.forEach((node) => assigned.add(node.key));
    groups.push({
      identityBasis: "request",
      identityKey: JSON.stringify([requestId, branchRoot]),
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
      identityBasis: branchNodes.some((node) => (generationLinks.get(node.key)?.size ?? 0) > 1)
        ? "unresolved"
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
  graph: AttemptGraph,
): NodeGroup[] {
  const componentRoots = connectedComponentRoots(nodes, graph);
  const normalizedNodes = nodes.map((node) => ({
    ...node,
    branchRoot: componentRoots.get(node.key) ?? node.branchRoot,
  }));
  const groups: NodeGroup[] = [];
  const assigned = new Set<string>();
  const generationLinks = generationLinksFor(normalizedNodes, graph);
  for (const generationId of uniqueStrings(normalizedNodes.map((node) => node.record.generationId))) {
    const groupNodes = normalizedNodes.filter((node) => {
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

  const requestCandidates = normalizedNodes.filter(
    (node) => !assigned.has(node.key) && (generationLinks.get(node.key)?.size ?? 0) <= 1,
  );
  const requestGroups = requestGroupsFor(requestCandidates);
  for (const seeds of requestGroups) {
    const { branchRoot, record: { requestId } } = seeds[0]!;
    const groupNodes = expandIdentityGroup(
      seeds,
      requestCandidates.filter((node) => !assigned.has(node.key)),
      graph,
      (node) =>
        node.branchRoot === branchRoot &&
        (node.record.requestId === requestId || node.record.requestId === null),
    );
    if (groupNodes.length === 0) {
      continue;
    }
    groupNodes.forEach((node) => assigned.add(node.key));
    groups.push({
      identityBasis: "request",
      identityKey: JSON.stringify([requestId, branchRoot]),
      nodes: groupNodes,
      branchRoot,
    });
  }

  const remaining = new Map<string, GraphNode[]>();
  for (const node of normalizedNodes) {
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

function requestGroupsFor(nodes: GraphNode[]): GraphNode[][] {
  const groups = new Map<string, GraphNode[]>();
  for (const node of nodes) {
    if (node.record.requestId === null) {
      continue;
    }
    const key = JSON.stringify([node.branchRoot, node.record.requestId]);
    const group = groups.get(key) ?? [];
    group.push(node);
    groups.set(key, group);
  }
  return [...groups.entries()]
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([, group]) => group);
}

function expandIdentityGroup(
  seeds: GraphNode[],
  candidates: GraphNode[],
  graph: AttemptGraph,
  accepts: (node: GraphNode) => boolean,
): GraphNode[] {
  const byKey = new Map(candidates.map((node) => [node.key, node]));
  const output = new Map<string, GraphNode>();
  const queue = [...seeds].sort((left, right) => left.key.localeCompare(right.key));
  while (queue.length > 0) {
    const node = byKey.get(queue.shift()!.key);
    if (!node || output.has(node.key) || !accepts(node)) {
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
  graph: AttemptGraph,
): Map<string, Set<string>> {
  const links = new Map<string, Set<string>>();
  for (const node of nodes) {
    if (node.record.generationId !== null) {
      continue;
    }
    const generations = new Set<string>();
    const candidates = nodes.filter((candidate) => candidate.branchRoot === node.branchRoot);
    // A terminal/original response cannot borrow a later regeneration's ID.
    // Forward linkage requires positive fragment evidence; backward linkage
    // stops at terminal responses and conflicting fallback request identities.
    for (const direction of ["parents", "children"] as const) {
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
        if (direction === "children" && !isGenerationFragment(current)) {
          continue;
        }
        for (const candidate of linkedNodes(current, candidates, graph, [direction])) {
          if (
            visited.has(candidate.key) ||
            (direction === "parents" && isTerminalNode(candidate)) ||
            (node.record.requestId !== null &&
              candidate.record.requestId !== null &&
              candidate.record.requestId !== node.record.requestId)
          ) {
            continue;
          }
          queue.push(candidate);
        }
      }
    }
    links.set(node.key, generations);
  }
  return links;
}

function isTerminalNode(node: GraphNode): boolean {
  const status = (node.record.status ?? "").trim().toLowerCase();
  return (
    node.record.endTurn === true ||
    FAILED_STATUSES.has(status) ||
    REJECTED_STATUSES.has(status)
  );
}

function isGenerationFragment(node: GraphNode): boolean {
  if (isTerminalNode(node)) {
    return false;
  }
  const record = node.record;
  return (
    record.role === "tool" ||
    (record.role === "assistant" &&
      (record.endTurn === false ||
        NON_ANSWER_CHANNELS.has((record.channel ?? "").toLowerCase()) ||
        NONTERMINAL_STATUSES.has((record.status ?? "").trim().toLowerCase())))
  );
}

function graphNeighbors(
  node: GraphNode,
  candidates: GraphNode[],
  graph: AttemptGraph,
): GraphNode[] {
  return linkedNodes(node, candidates, graph, ["children", "parents"]);
}

function childNeighbors(
  node: GraphNode,
  candidates: GraphNode[],
  graph: AttemptGraph,
): GraphNode[] {
  return linkedNodes(node, candidates, graph, ["children"]);
}

function linkedNodes(
  node: GraphNode,
  candidates: GraphNode[],
  graph: AttemptGraph,
  directions: ReadonlyArray<"children" | "parents">,
): GraphNode[] {
  const byKey = new Map(candidates.map((candidate) => [candidate.key, candidate]));
  const keys = new Set(directions.flatMap((direction) => [...(graph[direction].get(node.key) ?? [])]));
  return [...keys]
    .map((key) => byKey.get(key))
    .filter((candidate): candidate is GraphNode => candidate !== undefined)
    .sort((left, right) => left.key.localeCompare(right.key));
}

function connectedComponentRoots(
  nodes: GraphNode[],
  graph: AttemptGraph,
): Map<string, string> {
  const remaining = new Set(nodes.map((node) => node.key));
  const roots = new Map<string, string>();
  for (const seed of [...remaining].sort()) {
    if (!remaining.has(seed)) {
      continue;
    }
    const component: string[] = [];
    const queue = [seed];
    while (queue.length > 0) {
      const key = queue.shift()!;
      if (!remaining.has(key)) {
        continue;
      }
      remaining.delete(key);
      component.push(key);
      const node = graph.byKey.get(key);
      if (!node) {
        continue;
      }
      for (const neighbor of graphNeighbors(node, nodes, graph)) {
        if (remaining.has(neighbor.key)) {
          queue.push(neighbor.key);
        }
      }
    }
    const root = component.sort()[0]!;
    for (const key of component) {
      roots.set(key, root);
    }
  }
  return roots;
}

function promptEvidenceGroupFor(
  user: MessageRecord,
  groups: NodeGroup[],
  graph: AttemptGraph,
): NodeGroup | null {
  if (user.generationId !== null) {
    const generationGroups = groups.filter((group) =>
      group.nodes.some((node) => node.record.generationId === user.generationId),
    );
    return generationGroups.length === 1 ? generationGroups[0] ?? null : null;
  }
  if (user.requestId !== null) {
    const requestGroups = groups.filter((group) =>
      group.nodes.some((node) => node.record.requestId === user.requestId),
    );
    return requestGroups.length === 1 ? requestGroups[0] ?? null : null;
  }
  if (groups.length === 1) {
    return groups[0] ?? null;
  }

  // Identity strength does not identify the original response. Prefer graph
  // precedence, then strictly ordered sibling evidence; never break ties by ID.
  const nodes = groups.flatMap((group) => group.nodes);
  const predecessors = new Map(groups.map((group) => [group, new Set<NodeGroup>()]));
  for (const group of groups) {
    const reachable = descendantsOfGroup(group, nodes, graph);
    for (const other of groups) {
      if (other !== group && other.nodes.some((node) => reachable.has(node.key))) {
        predecessors.get(other)!.add(group);
      }
    }
  }
  const roots = groups.filter((group) => predecessors.get(group)!.size === 0);
  if (roots.length === 1) {
    return roots[0] ?? null;
  }
  const observed = roots.map((group) => ({
    group,
    time: firstNodeTime(group, graph),
  }));
  if (observed.some(({ time }) => time === null)) {
    return null;
  }
  observed.sort((left, right) => left.time! - right.time!);
  const first = observed[0];
  const second = observed[1];
  if (!first || !second || first.time! >= second.time!) {
    return null;
  }
  if (first.group.identityBasis === "generation") {
    const requestIds = new Set(first.group.nodes.map((node) => node.record.requestId));
    requestIds.delete(null);
    if (groups.some((group) =>
      group !== first.group &&
      group.identityBasis === "generation" &&
      group.nodes.some((node) => requestIds.has(node.record.requestId)),
    )) {
      return null;
    }
  }
  return first.group;
}

function descendantsOfGroup(
  group: NodeGroup,
  nodes: GraphNode[],
  graph: AttemptGraph,
): Set<string> {
  const visited = new Set<string>();
  const queue = [...group.nodes];
  while (queue.length > 0) {
    const node = queue.shift()!;
    if (visited.has(node.key)) {
      continue;
    }
    visited.add(node.key);
    queue.push(...childNeighbors(node, nodes, graph));
  }
  return visited;
}

function shouldUsePromptEvidence(
  group: NodeGroup,
  promptGroup: NodeGroup | null,
): boolean {
  return group === promptGroup &&
    (group.nodes.length === 0 ||
      ((group.identityBasis === "generation" || group.identityBasis === "request") &&
        hasGenerationSpecificIdentity(group)));
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

function firstNodeTime(group: NodeGroup, graph: AttemptGraph): number | null {
  const keys = new Set(group.nodes.map((node) => node.key));
  const entries = group.nodes.filter((node) =>
    ![...(graph.parents.get(node.key) ?? [])].some((parent) => keys.has(parent)),
  );
  const times = entries.map((node) => node.record.createdAt);
  if (times.length === 0 || times.some((time) => time === null || !validTime(time))) {
    return null;
  }
  return Math.min(...times.map((time) => Date.parse(time!)));
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
      timeBasis: "response_observed",
      earliestPossibleAt: null,
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
  const values = new Set<string>();
  for (const record of records) {
    if (record.origin !== null) {
      values.add(record.origin);
    }
    if (record.metadata.imported === true) {
      values.add("imported");
    }
    if (record.metadata.from_copy === true) {
      values.add("copied");
    }
    if (record.metadata.from_shared === true) {
      values.add("shared");
    }
  }
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

function branchRootFor(
  node: GraphNode,
  nodes: GraphNode[],
  roots: Set<string>,
  graph: AttemptGraph,
): string {
  const queue = [node];
  const visited = new Set<string>();
  while (queue.length > 0) {
    const current = queue.shift()!;
    if (visited.has(current.key)) {
      continue;
    }
    visited.add(current.key);
    if (roots.has(current.key)) {
      return current.key;
    }
    for (const neighbor of graphNeighbors(current, nodes, graph)) {
      if (!visited.has(neighbor.key)) {
        queue.push(neighbor);
      }
    }
  }
  return node.key;
}

function graphNodes(graph: AttemptGraph): GraphNode[] {
  return [...new Map(
    [...graph.byKey.values()].map((node) => [node.key, node] as const),
  ).values()];
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
