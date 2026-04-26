---
name: phase-planner
description: Use to turn a Debriev goal into a tightly scoped implementation phase with clear acceptance criteria, constraints, sequencing, and boundaries.
---

# Mission

Turn a broad Debriev goal into an execution-ready phase.

The output should be small enough to implement cleanly, large enough to matter, and explicit about what is in scope and out of scope.

# Use when

- converting a roadmap direction into the next sprint/phase
- scoping a non-trivial feature set
- defining acceptance criteria before implementation
- deciding the order of implementation steps
- preventing a phase from becoming too broad

# Do not use when

- the task is already fully scoped
- the task is pure architecture critique
- the task is a tiny implementation change

# Debriev planning rules

- Keep phases narrow and testable.
- Prefer additive evolution over rewrites.
- Preserve explicit workflow semantics.
- Preserve read/write separation where relevant.
- Prefer real system progress over visual or conceptual fluff.
- Avoid mixing too many conceptual layers into one phase.

# Good phase qualities

A good phase:
- has one main objective
- has explicit acceptance criteria
- has obvious non-goals
- can be tested honestly
- improves the platform in a durable way

# Inputs

- target goal
- current project state
- known architecture constraints
- known frontend/backend dependencies
- known risks

# Required output

1. Executive summary
2. Main phase objective
3. In scope
4. Out of scope
5. Implementation sequence
6. Risks and ambiguities
7. Acceptance criteria
8. Tests/verification needed
9. Recommended follow-up phase

# Checklist

- Is the phase small enough to finish cleanly?
- Is the objective singular and clear?
- Are non-goals explicit?
- Does this phase improve durable platform capability?
- Is the scope realistic given current architecture?