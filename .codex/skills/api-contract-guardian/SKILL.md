---
name: api-contract-guardian
description: Use for Debriev API and schema evolution to keep contracts stable, additive, explicit, and compatible with a backend-first review platform.
---

# Mission

Guard Debriev API contracts so they stay explicit, stable, additive, and aligned with backend truth.

# Use when

- adding or changing request/response fields
- reviewing endpoint design
- reviewing route semantics
- deciding whether a change should be additive or breaking
- checking if frontend needs are pushing the API in the wrong direction

# Do not use when

- the task is pure persistence modeling
- the task is only frontend layout or styling
- the task is a narrow internal refactor with no contract effect

# Debriev rules

- Prefer additive changes.
- Preserve stable output shapes unless change is clearly justified.
- Keep route semantics explicit.
- Do not make the API mirror frontend hacks.
- Keep read-side payloads read-oriented and execution payloads execution-oriented.
- Avoid ambiguous “do everything” endpoints.

# Focus areas

- request/response schema clarity
- endpoint responsibility
- read vs write contract boundaries
- frontend contract durability
- backward compatibility
- naming consistency
- review-state and workflow payload stability

# Inputs

- current endpoint behavior
- proposed endpoint behavior
- affected schemas
- frontend needs
- compatibility constraints

# Required output

1. Executive summary
2. Current contract problems
3. Proposed contract changes
4. Additive vs breaking analysis
5. Naming and schema recommendations
6. Risks
7. Acceptance criteria
8. Tests to add
9. Follow-up cleanup worth doing later

# Checklist

- Is this additive?
- Is the endpoint responsibility clear?
- Does this leak execution concerns into read contracts?
- Is the API compensating for frontend-local truth problems?
- Will this still make sense after future workflow expansion?