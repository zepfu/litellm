# LiteLLM Repository Rules

Global `~/.codex/AGENTS.md` governs scope, authorization, testing, orchestration,
delivery, and cross-repository reporting. This file adds LiteLLM-specific
requirements. Authorized implementation includes synchronizing documentation
for the affected behavior; it does not waive global test approval.

## Ownership and Routing

Implement LiteLLM-owned behavior here and preserve compatibility with the
unmodified stock Codex client. A Codex fork, patch, custom build, or deployment
requires explicit operator authorization for work in that repository.

Anthropic/Claude models must use Claude-native or Anthropic-native egress with
credentials accepted for that route. Never route those models through Codex,
ChatGPT OAuth, an OpenAI/Codex adapter, or another provider's egress. This
restriction follows the resolved model/provider, not the inbound wire format:
Claude Code may select non-Anthropic models through supported adapters.

Apply that boundary to aliases, retries, fallbacks, cooldown recovery, and all
authorized probes or acceptance checks. If native Anthropic credentials or
routing are unavailable, fail closed with an attributable routing/authentication
error. Do not substitute Codex egress to recover availability or pass acceptance.

## Spawned-Agent Failure Intake

This fork owns AAWM aliases and their routing, injected instructions, tool
advertisement/transformation, response handling, and observability. For incoming
`investigate-codex-*.md` reports, assess these layers rather than closing the
report merely because the final provider call is not defective.

Apply the global reporting exclusions. Disposition caller prompting/argument
mistakes and ordinary upstream capacity errors without turning them into
implementation work. Distinguish those from evidenced defects in LiteLLM's
injected instructions, tool schemas, retry classification, or response handling.

For qualifying reports, identify the owning layer and smallest useful remedy.
Consider read-only/final-answer contracts, malformed or missing completions,
setup-only or wrong-domain output, session-history/Langfuse attribution, and
redispatch accounting where relevant. Record evidence, disposition, and an
existing or proposed work-item reference; deduplicate reports of the same defect.
Intake and reviewer suggestions do not authorize remediation or expand a goal.

## Reusable Tool Failures

Record a LiteLLM-owned reusable tool failure only when it recurs or affects
multiple workflows. Use `.analysis/tool-failure-[repo]-[YYYYMMDDhhmmss].md`
with the attempt, failure, proposed owner/change, reuse scope, and why it is
not a one-off caller error. Do not use this category to bypass dispatch-report
exclusions or an operator prohibition on investigation artifacts.

## Discovery and Evaluation

For broad delegated alias or investigation discovery, include the inventory
contract in the prompt: list commands/sources and matching candidates, mark
each inspected/omitted/unavailable, explain omissions, classify relevance, and
report coverage gaps. For narrow work, name the exact inputs instead.

Scores, evals, and session-history flags must judge only the communicated
contract and evidence visible to the agent. Do not penalize agents for hidden
filesystem state or unstated discovery requirements.

## Backend Constraints

- Use existing `BaseConfig` provider transformations in `litellm/llms/`, typed
  contracts in `litellm/types/`, provider-specific exceptions, and consistent
  OpenAI-format output. Preserve affected sync/async and streaming paths,
  environment/programmatic configuration, and Pydantic v1/v2 compatibility.
- Put model capability flags in `model_prices_and_context_window.json`; use
  `get_model_info` or existing capability helpers instead of model-name checks.
- Use Prisma model methods for proxy database access, not raw
  `execute_raw`/`query_raw` SQL.
- Never close HTTP/SDK clients during cache eviction, including inside
  `LLMClientCache._remove_key()`: in-flight requests may still own them.
  Keep shutdown cleanup in `close_litellm_async_clients()`.
- Preserve affected open-source/enterprise interfaces; inspect `enterprise/`
  when changing a shared contract.

## Dashboard Constraints

- Reuse `ui/litellm-dashboard` common components. Do not introduce Tremor
  components except its Table and required Table subcomponents.
- Match UI selection cardinality to backend scalar/array contracts.
- The proxy serves prebuilt UI from `litellm/proxy/_experimental/out/`.
  For authorized UI builds, build from `ui/litellm-dashboard/` and synchronize
  the resulting assets to the served directory.
- Provider logos live in `ui/litellm-dashboard/public/assets/logos/` and the
  corresponding served `out/assets/logos/`. SVGs loaded through `<img>` need
  explicit colors or a color variant, not `fill="currentColor"`.
- For approved UI tests, use Vitest/React Testing Library, existing setup mocks,
  `screen`, semantic role/label/text queries, `query*` for absence, `act` for
  direct interaction updates, and `waitFor` for asynchronous state. Follow
  existing `should ...` names. Do not mandate render-only tests or new test
  cases merely because an entity was added; apply global candidate pruning.

## Local Tooling

Use `./.venv/bin/python` for repository Python scripts. Read `CLAUDE.md` and
`Makefile` for relevant commands, not as authorization to run suites.
Approved Python tests belong in existing `tests/` locations; UI tests stay
with the dashboard. Use repository Ruff configuration for scoped lint.

Provider documentation is under `docs/my-website/docs/providers/`. Use existing
`.github/ISSUE_TEMPLATE/` and pull-request templates when those artifacts are
requested; template test instructions remain subject to operator approval.

### Cursor Cloud Only

These details do not describe Thoth's deployment. In Cursor Cloud, use the
repo-local `.venv` and system Python; `uv` is under `~/.local/bin`.
For an authorized local proxy launch, use
`./.venv/bin/litellm --config dev_config.yaml --port 4000`; startup runs
migrations, so confirm the intended `DATABASE_URL` first and await `/health`.
Do not rely on an embedded/default database target.

For approved checks, follow the current dependency configuration:
`psycopg-binary` is needed by pytest-postgresql and `openapi-core` by the
interactions OpenAPI checks. Do not pass unsupported `pytest --timeout`.
Dependency installation, server startup, and suite commands are not automatic
setup steps.
