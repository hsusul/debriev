---
name: frontend-review-quality
description: Use when refining an existing Debriev frontend screen for hierarchy, density, inspection UX, trust, and review clarity after the first implementation exists.
---

# Mission

Refine Debriev frontend screens for trust, density, hierarchy, and review speed.

This skill is for improving an existing screen, not inventing a new product direction.

# Use when

- reviewing an already-built screen
- improving hierarchy, spacing, density, or clarity
- refining evidence inspection UX
- tightening decision/action visibility
- checking whether the UI feels like a serious review tool rather than a demo

# Do not use when

- no screen exists yet
- the task is backend-only
- the task is a brand-new frontend implementation from scratch
- the task is product or workflow design rather than screen refinement

# Debriev UI identity

Debriev is a review workspace, not a chatbot, landing page, or generic SaaS dashboard.

The interface should feel:
- serious
- dense but readable
- evidence-first
- audit-friendly
- fast for expert inspection

# Review quality priorities

1. orientation
2. hierarchy
3. evidence clarity
4. verdict and reasoning visibility
5. next-action clarity
6. density and scannability
7. inspector overload control
8. resolved-state usability

# Debriev-specific rules

- Primary anchor, reasoning, and next action should surface first.
- Support assessments, excluded links, and scope details should be progressively disclosed.
- Avoid decorative clutter.
- Avoid giant card grids unless the surface truly needs them.
- Prefer keyboard-friendly, dense layouts where appropriate.
- Keep the main work surface obvious on first glance.
- Treat evidence and provenance as core content, not secondary metadata.

# Polish rules

- Fix spacing, overflow, truncation, alignment, and hierarchy before adding decoration.
- Improve readability before adding visual flair.
- Use motion only when it improves state change clarity.
- Keep the interface serious and trustworthy.
- Prefer structure over ornament.

# Inputs

- screenshot, component, or current screen
- purpose of the screen
- main reviewer task
- current pain points
- any constraints from backend state or layout

# Required output

1. Executive summary
2. What is working
3. What is hurting trust, clarity, or speed
4. Top hierarchy problems
5. Specific improvements
6. Good enough for now vs must fix soon
7. Optional polish ideas

# Checklist

- Is the main work surface obvious immediately?
- Can I tell what is flagged and why in one quick scan?
- Is the next action obvious?
- Is the right-hand inspector overloaded?
- Can secondary evidence details be collapsed?
- Does the UI feel like a professional review tool rather than a demo?
- Is the layout dense in a useful way rather than a cluttered way?