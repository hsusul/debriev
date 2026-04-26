---
name: frontend-foundation
description: Use for Debriev frontend implementation before or during UI construction. Best for layout scaffolding, component selection, page shells, API-facing state structure, and frontend architecture decisions.
---

# Mission

Build the Debriev frontend as a review workspace, not a chatbot.

# Product rules

- Default to a dense, professional workbench UI.
- Prefer review and inspection patterns over marketing or chat patterns.
- Do not introduce chat-first UX.
- Do not introduce hero sections, splash sections, or generic dashboard filler unless explicitly requested.
- Default to dark-mode-friendly, high-density, trust-oriented design.

# Frontend stack rules

- Prefer React + Tailwind + shadcn/ui patterns.
- Reuse shadcn primitives before inventing custom ones.
- Keep components composable and inspectable.
- Prefer explicit prop-driven components over tangled local state.

# Debriev UI rules

Every screen should answer:
1. what am I looking at?
2. why is it flagged?
3. what can I do next?

- Prioritize orientation, status, and action.
- Use progressive disclosure for dense evidence or audit details.
- Keep the primary action path obvious.
- Treat evidence and provenance as first-class UI content, not footnotes.

# State rules

- Keep backend truth primary.
- Do not invent queue truth locally.
- Keep API adapters explicit.
- Separate page shell, pane layout, and inspector content.
- Avoid giant all-in-one screen components.

# Use when

- building a new workbench screen
- wiring frontend state to backend payloads
- choosing component structure
- scaffolding a new page or pane
- cleaning frontend architecture before polish

# Do not use when

- the task is backend-only
- the task is schema design only
- the task is post-build hierarchy refinement only

# Required output

1. Screen goal
2. Layout structure
3. Component breakdown
4. State shape
5. API/data dependencies
6. Files to create or edit
7. Implementation notes
8. Risks or follow-ups

# Checklist

- Is the main work surface obvious?
- Is evidence/provenance surfaced clearly?
- Is there one clear next action?
- Is state cleanly tied to backend truth?
- Is the screen a workbench, not a dashboard or chatbot?