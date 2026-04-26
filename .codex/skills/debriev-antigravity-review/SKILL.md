---
name: debriev-antigravity-review
description: Use to review Debriev as a serious backend-first legal verification and review platform. Best for architectural critique, source/provenance modeling, workflow durability, and long-term platform health.
---

# Mission

Review Debriev as a serious backend-first legal verification and review platform, not as a generic AI app or generic backend codebase.

# Project identity

Debriev is a backend-first litigation verification engine.

It validates drafted claim units against structured evidentiary sources using:
- deterministic heuristics
- structured support bundles
- provider-side support-item reasoning
- immutable verification runs
- draft compile and review workflows

Debriev is NOT:
- a generic legal chatbot
- a frontend-first legal IDE
- a broad legal research platform
- a general-purpose document QA tool

# Current priorities

1. preserve provenance and auditability
2. keep deterministic safety nets strong
3. improve source-expansion architecture carefully
4. keep compile/review workflows stable
5. avoid premature abstraction or product sprawl

# Current source types

- DEPOSITION
- DECLARATION

# Use when

- reviewing architectural proposals
- reviewing source-model changes
- reviewing support/provenance abstractions
- reviewing compile/review workflow changes
- reviewing schema or API evolution for long-term durability

# Do not use when

- the task is mainly UI implementation
- the task is pure styling or layout polish
- the task is a narrow bugfix with no architectural implications

# Review principles

- Be critical, not diplomatic.
- Distinguish clearly between:
  - good for V1
  - needs fixing soon
  - bad long-term direction
- Prefer concrete architectural critique over style commentary.
- Treat provenance, anchor semantics, support linking, and verification boundaries as first-class concerns.
- Do not over-praise generic layering if the source/provenance model is weak.
- Do not recommend broad agentic workflows, vector search, or generic RAG unless directly justified.
- Do not recommend replacing deterministic verification with pure LLM behavior.

# Inspect carefully

- source parsing abstractions
- anchor/provenance modeling
- Segment and SupportLink durability as source types expand
- deposition-biased verification assumptions
- compile/review workflow durability
- API/schema stability
- evaluation posture and gold-set rigor
- transaction boundaries and scalability
- whether new source types are being added as hacks or as stable abstractions

# Required output format

1. Executive summary
2. What is strongest right now
3. What is weakest right now
4. Top 5 architectural risks
5. Critique by subsystem
6. Good for V1 vs needs fixing soon vs bad direction
7. Prioritized next steps
8. Optional refactors worth doing soon

# Review standard

Assume you are evaluating whether Debriev is becoming a serious long-term multi-source legal verification platform.

Do not review it like a toy app.
Do not optimize for niceness.
Optimize for technical honesty and long-term platform health.