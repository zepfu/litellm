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
  branchRoot: string | null;
  ownerKey: string | null;
  userLinked: boolean;
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

interface PromptContext {
  user: MessageRecord;
  branchRoots: string[];
}

interface GenerationCandidates {
  ids: Set<string>;
  ambiguous: boolean;
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
  const views = sortedMessages(normalized)
    .filter((item) => item.role === "user")
    .map((user) => ({ user, nodes: descendantsForUser(user, graph) }));
  const nodes = normalizeOwnership(views, graph);
  const groups = groupsForNodes(nodes, graph);
  const contexts = new Map<NodeGroup, PromptContext[]>();
  const promptOwners = new Map<NodeGroup, MessageRecord[]>();

  for (const view of views) {
    const viewNodes = new Map(view.nodes.map((node) => [node.key, node]));
    const visibleGroups = new Map<NodeGroup, NodeGroup>();
    for (const group of groups) {
      const visibleNodes = group.nodes
        .map((node) => viewNodes.get(node.key))
        .filter((node): node is GraphNode => node !== undefined);
      if (visibleNodes.length === 0) {
        continue;
      }
      const context = contexts.get(group) ?? [];
      context.push({
        user: view.user,
        branchRoots: uniqueStrings(visibleNodes.map((node) => node.branchRoot)),
      });
      contexts.set(group, context);
      visibleGroups.set({ ...group, nodes: visibleNodes }, group);
    }
    if (visibleGroups.size === 0) {
      const group: NodeGroup = {
        identityBasis: "provisional",
        identityKey: JSON.stringify(["prompt", view.user.messageId]),
        nodes: [],
        branchRoot: null,
      };
      groups.push(group);
      contexts.set(group, [{ user: view.user, branchRoots: [] }]);
      visibleGroups.set(group, group);
    }
    const selected = promptEvidenceGroupFor(view.user, [...visibleGroups.keys()], graph);
    const group = selected ? visibleGroups.get(selected) : undefined;
    if (group) {
      const owners = promptOwners.get(group) ?? [];
      owners.push(view.user);
      promptOwners.set(group, owners);
    }
  }
  return groups.map((group) => {
    const groupContexts = contexts.get(group) ?? [];
    const owners = promptOwners.get(group) ?? [];
    const prompt = groupContexts.length === 1 && owners.length === 1
      ? owners[0] ?? null
      : null;
    return buildAttempt(groupContexts, group, options, prompt);
  }).sort(compareAttempts);
}

function buildAttempt(
  contexts: PromptContext[],
  group: NodeGroup,
  options: {
    scope: LedgerScope;
    conversationId: string;
    mapping: ModelMappingVersion;
  },
  user: MessageRecord | null,
): ReconstructedAttempt {
  const usePromptEvidence = user !== null;
  const nodes = sortedNodes(group.nodes);
  const final = finalAnswer(nodes);
  const generationIds = uniqueStrings(nodes.map((node) => node.record.generationId));
  const requestIds = uniqueStrings(nodes.map((node) => node.record.requestId));
  const promptKeys = contexts.flatMap((context) =>
    (context.branchRoots.length > 0 ? context.branchRoots : [group.identityKey])
      .map((branch) => stableId("prompt", options.conversationId, context.user.messageId, branch)),
  );
  const branchRoots = uniqueStrings(nodes.map((node) => node.branchRoot));
  const aliases: Array<[string, string]> = [
    ...promptKeys.map((key) => ["prompt", key] as [string, string]),
    ...branchRoots.map((branch) => ["branch", `${options.conversationId}:${branch}`] as [string, string]),
    ...generationIds.map((id) => ["generation", `${options.conversationId}:${id}`] as [string, string]),
    ...requestIds.map((id) => ["request", `${options.conversationId}:${id}`] as [string, string]),
    ...nodes.map((node) => [
      "message",
      `${options.conversationId}:${node.record.messageId}`,
    ] as [string, string]),
  ];
  const identityBasis = group.identityBasis;
  const identityKey = JSON.stringify([options.conversationId, identityBasis, group.identityKey]);
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
  const contextualRecords = [
    ...contexts.map((context) => context.user),
    ...nodes.map((node) => node.record),
  ];
  const surface = surfaceFor(contextualRecords);
  const origin = originFor(contextualRecords);
  const warnings: string[] = [];
  if (identityBasis === "unresolved") {
    warnings.push("unresolved_linkage");
  }
  if (group.identityBasis === "provisional") {
    warnings.push("provisional_identity");
  }
  if (contexts.length > 0 && !usePromptEvidence && hasGenerationSpecificIdentity(group)) {
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
      ...contexts.map((context) => context.user.messageId),
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
    const node = { key, record, branchRoot: key, ownerKey: null, userLinked: false };
    byKey.set(key, node);
    byKey.set(record.messageId, node);
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
): GraphNode[] {
  const userNode = graph.byKey.get(user.nodeId ?? user.messageId);
  if (!userNode) {
    return [];
  }
  const candidates = graphNodes(graph).filter((node) => node.key !== userNode.key);
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
  const branchRoots = new Map<string, string | null>();
  for (const node of output) {
    branchRoots.set(
      node.key,
      branchRootFor(node, output, roots, graph),
    );
  }
  return output.map((node) => ({
    ...node,
    branchRoot: branchRoots.get(node.key) ?? null,
  }));
}

function normalizeOwnership(
  views: ReadonlyArray<{ user: MessageRecord; nodes: GraphNode[] }>,
  graph: AttemptGraph,
): GraphNode[] {
  const evidence = new Map<string, GraphNode>();
  const owners = new Map<string, Set<string>>();
  const branches = new Map<string, Set<string | null>>();
  const ambiguous = new Set<string>();
  for (const { user, nodes } of views) {
    for (const node of nodes) {
      evidence.set(node.key, { ...node, userLinked: true });
      const nodeOwners = owners.get(node.key) ?? new Set<string>();
      nodeOwners.add(JSON.stringify(["user", user.messageId, node.branchRoot]));
      owners.set(node.key, nodeOwners);
      const nodeBranches = branches.get(node.key) ?? new Set<string | null>();
      nodeBranches.add(node.branchRoot);
      branches.set(node.key, nodeBranches);
      if (node.branchRoot === null) {
        ambiguous.add(node.key);
      }
    }
  }
  const orphans = graphNodes(graph).filter((node) =>
    !evidence.has(node.key) && ["assistant", "tool"].includes(node.record.role ?? ""),
  );
  const componentRoots = connectedComponentRoots(orphans, graph);
  for (const node of orphans) {
    const root = componentRoots.get(node.key)!;
    evidence.set(node.key, node);
    owners.set(node.key, new Set([JSON.stringify(["orphan", root])]));
    branches.set(node.key, new Set([root]));
  }
  return [...evidence.values()].map((node) => ({
    ...node,
    branchRoot: branches.get(node.key)?.has(null)
      ? null
      : singleString([...(branches.get(node.key) ?? [])]),
    ownerKey: !ambiguous.has(node.key) && owners.get(node.key)?.size === 1
      ? [...owners.get(node.key)!][0]!
      : null,
  }));
}

function groupsForNodes(
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
      const linkedGenerations = generationLinks.get(node.key)!;
      return (
        node.record.generationId === null &&
        !linkedGenerations.ambiguous &&
        linkedGenerations.ids.size === 1 &&
        linkedGenerations.ids.has(generationId)
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

  const unresolved = new Set(nodes.filter((node) => {
    const linked = generationLinks.get(node.key);
    return node.record.generationId === null &&
      (node.ownerKey === null || linked?.ambiguous || (linked?.ids.size ?? 0) > 1);
  }).map((node) => node.key));
  const requestCandidates = nodes.filter((node) =>
    !assigned.has(node.key) && !unresolved.has(node.key),
  );
  const componentRoots = connectedComponentRoots(requestCandidates, graph);
  const requestGroups = new Map<string, NodeGroup>();
  const requestOwners = new Map<string, string>();
  for (const node of requestCandidates) {
    if (node.record.requestId === null) {
      continue;
    }
    const key = JSON.stringify([node.ownerKey, componentRoots.get(node.key), node.record.requestId]);
    const group = requestGroups.get(key) ?? {
      identityBasis: "request",
      identityKey: key,
      nodes: [],
      branchRoot: node.branchRoot,
    };
    group.nodes.push(node);
    requestGroups.set(key, group);
    requestOwners.set(node.key, key);
  }
  // Candidate owners are immutable: request iteration cannot consume a bridge.
  for (const node of requestCandidates) {
    if (node.record.requestId !== null) {
      continue;
    }
    const owners = nearestOwnersFor(node, requestCandidates, requestOwners, graph);
    if (owners.size === 1) {
      requestGroups.get([...owners][0]!)!.nodes.push(node);
    } else if (owners.size > 1) {
      unresolved.add(node.key);
    }
  }
  for (const group of requestGroups.values()) {
    group.nodes.forEach((node) => assigned.add(node.key));
    groups.push(group);
  }

  const remaining = nodes.filter((node) => !assigned.has(node.key));
  const residualRoots = connectedComponentRoots(remaining, graph);
  const residualGroups = new Map<string, NodeGroup>();
  for (const node of remaining) {
    const root = residualRoots.get(node.key)!;
    const key = JSON.stringify([node.ownerKey, root]);
    const group = residualGroups.get(key) ?? {
      identityBasis: "provisional",
      identityKey: key,
      nodes: [],
      branchRoot: node.branchRoot,
    };
    if (!node.userLinked || unresolved.has(node.key)) {
      group.identityBasis = "unresolved";
    }
    group.nodes.push(node);
    residualGroups.set(key, group);
  }
  for (const group of residualGroups.values()) {
    group.identityKey = JSON.stringify([
      group.identityKey,
      group.nodes.map((node) => node.record.messageId).sort(),
    ]);
    group.branchRoot = singleString(group.nodes.map((node) => node.branchRoot));
    groups.push(group);
  }
  return groups.sort((left, right) => left.identityKey.localeCompare(right.identityKey));
}

function nearestOwnersFor(
  node: GraphNode,
  candidates: GraphNode[],
  owners: ReadonlyMap<string, string>,
  graph: AttemptGraph,
): Set<string> {
  const result = new Set<string>();
  for (const direction of ["parents", "children"] as const) {
    const visited = new Set<string>();
    const queue = [node];
    while (queue.length > 0) {
      const current = queue.shift()!;
      if (visited.has(current.key)) {
        continue;
      }
      visited.add(current.key);
      const owner = owners.get(current.key);
      if (owner !== undefined) {
        result.add(owner);
        continue;
      }
      if (direction === "children" && isTerminalNode(current)) {
        continue;
      }
      queue.push(...linkedNodes(current, candidates, graph, [direction])
        .filter((neighbor) =>
          !(direction === "parents" && isTerminalNode(neighbor)) &&
          (neighbor.ownerKey === node.ownerKey || neighbor.ownerKey === null ||
            neighbor.record.generationId !== null),
        ));
    }
  }
  return result;
}

function generationLinksFor(
  nodes: GraphNode[],
  graph: AttemptGraph,
): Map<string, GenerationCandidates> {
  const links = new Map<string, GenerationCandidates>();
  for (const node of nodes) {
    if (node.record.generationId !== null) {
      continue;
    }
    const result: GenerationCandidates = {
      ids: new Set(),
      ambiguous: node.ownerKey === null,
    };
    links.set(node.key, result);
    if (result.ambiguous) {
      continue;
    }
    const uncertainIds = new Set<string>();
    // Terminal originals stop forward linkage. Missing continuation evidence
    // is uncertainty, not proof of either a new attempt or a later owner.
    for (const direction of ["parents", "children"] as const) {
      const visited = new Set<string>();
      const queue = [{ node, uncertain: false }];
      while (queue.length > 0) {
        const { node: current, uncertain } = queue.shift()!;
        const key = JSON.stringify([current.key, uncertain]);
        if (visited.has(key)) {
          continue;
        }
        visited.add(key);
        if (current.record.generationId !== null) {
          (uncertain ? uncertainIds : result.ids).add(current.record.generationId);
          continue;
        }
        if (direction === "children" && isTerminalNode(current)) {
          continue;
        }
        for (const candidate of linkedNodes(current, nodes, graph, [direction])) {
          if (direction === "parents" && isTerminalNode(candidate)) {
            continue;
          }
          const ownershipConflict = candidate.record.generationId === null &&
            candidate.ownerKey !== node.ownerKey;
          if (ownershipConflict && candidate.ownerKey !== null) {
            continue;
          }
          const requestConflict = node.record.requestId !== null &&
            candidate.record.requestId !== null &&
            candidate.record.requestId !== node.record.requestId;
          if (requestConflict && candidate.record.generationId === null) {
            continue;
          }
          queue.push({
            node: candidate,
            uncertain: uncertain || requestConflict || ownershipConflict ||
              (direction === "children" && !isGenerationFragment(current)),
          });
        }
      }
    }
    result.ambiguous ||= result.ids.size > 1 ||
      [...uncertainIds].some((id) => !result.ids.has(id));
  }

  const boundaries = new Map<string, string>();
  for (const node of nodes) {
    if (node.record.generationId !== null) {
      boundaries.set(node.key, JSON.stringify(["generation", node.record.generationId]));
      continue;
    }
    const result = links.get(node.key)!;
    if (node.record.requestId !== null || result.ambiguous) {
      boundaries.set(node.key, !result.ambiguous && result.ids.size === 1
        ? JSON.stringify(["generation", [...result.ids][0]])
        : JSON.stringify([
          result.ambiguous ? "unresolved" : "request",
          node.ownerKey,
          node.record.requestId ?? node.key,
        ]));
    }
  }
  // A sole generation candidate cannot consume a fragment that also borders
  // a different request owner or unresolved evidence.
  for (const node of nodes) {
    if (node.record.generationId !== null || node.record.requestId !== null) {
      continue;
    }
    const result = links.get(node.key)!;
    if (result.ambiguous || result.ids.size !== 1) {
      continue;
    }
    const expected = JSON.stringify(["generation", [...result.ids][0]]);
    const candidates = nearestOwnersFor(node, nodes, boundaries, graph);
    if (candidates.size !== 1 || !candidates.has(expected)) {
      result.ambiguous = true;
    }
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
  const byKey = new Map(nodes.map((node) => [node.key, node]));
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
      const node = byKey.get(key);
      if (!node) {
        continue;
      }
      for (const neighbor of graphNeighbors(node, nodes, graph)) {
        if (remaining.has(neighbor.key) && neighbor.ownerKey === node.ownerKey) {
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
  const eligible = (group: NodeGroup | null | undefined): NodeGroup | null =>
    group && (group.nodes.length === 0 ||
      group.identityBasis === "generation" || group.identityBasis === "request")
      ? group
      : null;
  if (user.generationId !== null) {
    const generationGroups = groups.filter((group) =>
      group.nodes.some((node) => node.record.generationId === user.generationId),
    );
    return generationGroups.length === 1 ? eligible(generationGroups[0]) : null;
  }
  if (user.requestId !== null) {
    const requestGroups = groups.filter((group) =>
      group.nodes.some((node) => node.record.requestId === user.requestId),
    );
    return requestGroups.length === 1 ? eligible(requestGroups[0]) : null;
  }
  if (groups.length === 1 && groups[0]?.nodes.length === 0) {
    return eligible(groups[0]);
  }

  // Identity strength does not identify the original response. Prefer graph
  // precedence across all groups, then strictly ordered sibling evidence.
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
    return eligible(roots[0]);
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
  return eligible(first.group);
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
): string | null {
  const queue = [node];
  const visited = new Set<string>();
  const reachableRoots = new Set<string>();
  while (queue.length > 0) {
    const current = queue.shift()!;
    if (visited.has(current.key)) {
      continue;
    }
    visited.add(current.key);
    if (roots.has(current.key)) {
      reachableRoots.add(current.key);
    }
    for (const neighbor of linkedNodes(current, nodes, graph, ["parents"])) {
      if (!visited.has(neighbor.key)) {
        queue.push(neighbor);
      }
    }
  }
  return reachableRoots.size === 1 ? [...reachableRoots][0]! : null;
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
