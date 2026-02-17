# Debriev Working Agreement

- v1 scope is US case citations only.
- Statutes, regulations, and Bluebook edge cases are out of scope for v1.
- Stable contracts live in `packages/core/src/debriev_core/types.py`: `DebrievReport`, `CitationSpan`, `VerificationResult`.
- Domain logic belongs in `packages/core`.
- `apps/api` and `apps/web` are thin wrappers around core contracts.
- DoD for this scaffold: upload -> stored document -> stored stub report -> report view works end-to-end.
