## Standing Directives

- Investigation intake and disposition
  - At the start of TODO-driven work, look for `.analysis/investigate-*.md`
    files.
  - For each investigation file, determine whether the failure mode points to:
    message shape/form problems between the orchestrating client, LiteLLM, and
    the destination LLM; a LiteLLM implementation defect; a deterministic
    failure mode that should become a new score; or an orchestrating agent
    session-level issue such as prompting or dispatch instructions.
  - After an investigation has been examined and dispositioned, move the
    original `investigate-*.md` file into `.analysis/investigation/` so the
    active investigation intake stays clean.
  - Create `.analysis/investigations.md` if it does not exist and append one
    entry for every disposition, including non-actionable outcomes. Each entry
    must include the original investigation filename, the outcome, any proposed
    follow-up item IDs, and the post-move file location under
    `.analysis/investigation/`.
  - Any follow-up action that should be taken from an investigation must be
    added under `Proposals (Pending Operator Feedback)` until the operator
    explicitly approves turning it into active TODO work.

- Dashboard reporting handoffs for observability schema changes
  - Any time this repo adds new structure to
    `aawm_tristore.public.session_history`, rate-limit/rate observation tables,
    provider health tables/views, or related reporting/observability tables,
    create or update a sibling dashboard handoff document under
    `/home/zepfu/projects/dashboard-shell/.analysis/handoff-*.md`.
  - The handoff must name changed tables/columns, field semantics,
    interpretation rules, aggregation guidance, API/reporting impact, and the
    dashboard surfaces that need to incorporate the new structure.
  - Do this before treating the LiteLLM-side change as complete, so dashboard
    model/repository reporting does not silently miss new data.

## Current Open Items

Deferred work lives in `.analysis/todo.deferred.md`.

No current non-deferred open items.

## Proposals (Pending Operator Feedback)

No current proposals.
