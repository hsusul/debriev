# Debriev AGENTS.md

Debriev is a backend-first legal verification and review platform.

Core rules:
- Do not turn Debriev into a generic chatbot or generic RAG app.
- Prioritize deterministic-first, provenance-aware, audit-friendly behavior.
- Prefer explicit workflows over hidden automation.
- Preserve stable output shapes unless a change is clearly justified.

Backend rules:
- Keep verification deterministic-first.
- Preserve historical reproducibility.
- Do not silently mutate persisted historical state.
- Prefer additive schema changes over broad churn.

Frontend rules:
- Treat Debriev as a review workspace, not a chat product.
- Prioritize draft review, claim inspection, evidence inspection, and the resolution loop.
- Use progressive disclosure for dense inspector details.
- Do not build hero pages, marketing dashboard filler, or chat-first UX.

Workflow rules:
- Keep route handlers thin.
- Prefer small, explicit service/workflow modules.
- Keep adapter/mapper responsibilities separated on read paths.

Testing rules:
- Run focused tests first, then broader regressions.
- Respect the gold-set evaluation harness when changing extraction or verification behavior.