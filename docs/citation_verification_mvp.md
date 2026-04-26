# Debriev Citation Verification MVP

Debriev’s MVP is a citation verification system, not a broad legal IDE.

## Core MVP backend

- Accept draft text.
- Extract cited propositions from draft assertions.
- Detect court citations deterministically.
- Run deterministic-first verification for each cited proposition.
- Persist immutable verification runs.
- Return a narrow draft-level citation verification summary.

## Core MVP route surface

- `POST /api/v1/citation-verification`
  - Input: draft text
  - Behavior: create draft, extract claims, run fresh verification, return only citation-level results plus a draft summary
- Existing routes still exist, but they are no longer the primary demo surface for MVP:
  - `POST /api/v1/drafts`
  - `POST /api/v1/drafts/{draft_id}/review`
  - `GET /api/v1/drafts/{draft_id}/review-state`

## Keep for MVP because they strengthen trust

- Draft intake from text
- Claim extraction persistence
- Immutable verification runs
- Deterministic heuristic verification
- Structured reasoning categories
- Verification support snapshots

## Defer from the primary MVP surface

- Review queue and reviewer decision workflows
- Draft review run history and advanced freshness UI
- Claim graph and cross-claim intelligence
- Re-extraction compare/apply admin workflows
- Audit rendering beyond concise verification reasoning

These systems are still useful infrastructure. They should remain in the repo for now, but they are not the main product loop.

## Out of scope for MVP

- Broad evidence bundle management as a user-facing workflow
- General legal research assistant behavior
- Chat UX
- Draft editor features
- Broad legal IDE workspace administration

## Honest capability boundary

The current backend can classify cited propositions, parse ordinary case-citation structure deterministically, and match a limited set of authorities against a small built-in MVP catalog.

It does **not** yet do full court-authority identity resolution against a comprehensive authority database.

For MVP, the product should expose that boundary honestly through `authority_status`, parsed authority fields, and match results rather than overclaiming broad case verification coverage.

## Smallest frontend surface for MVP

- Paste draft text
- Run citation verification
- Review flagged cited propositions
- See citation text, proposition text, verdict, reasoning, and support snippet

No broad workspace administration is required for the MVP demo.
