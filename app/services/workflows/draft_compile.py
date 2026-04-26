"""Draft-wide compile workflow built on top of per-claim verification."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload, sessionmaker

from app.core.enums import ClaimGraphRelationshipType, SupportStatus
from app.core.exceptions import NotFoundError
from app.models import Assertion, ClaimUnit, Draft
from app.services.llm.base import ProviderSupportAssessment
from app.services.verification.classifier import ClaimVerificationService, VerificationExecution
from app.services.verification.snapshot_adapter import parse_verification_support_snapshot

FLAGGED_VERDICTS = frozenset(
    {
        SupportStatus.OVERSTATED,
        SupportStatus.AMBIGUOUS,
        SupportStatus.UNSUPPORTED,
        SupportStatus.UNVERIFIED,
    }
)
DEFAULT_COMPILE_MAX_CONCURRENCY = 5


@dataclass(slots=True)
class DraftCompileVerdictCounts:
    supported: int = 0
    partially_supported: int = 0
    overstated: int = 0
    ambiguous: int = 0
    unsupported: int = 0
    unverified: int = 0

    def increment(self, verdict: SupportStatus) -> None:
        if verdict == SupportStatus.SUPPORTED:
            self.supported += 1
        elif verdict == SupportStatus.PARTIALLY_SUPPORTED:
            self.partially_supported += 1
        elif verdict == SupportStatus.OVERSTATED:
            self.overstated += 1
        elif verdict == SupportStatus.AMBIGUOUS:
            self.ambiguous += 1
        elif verdict == SupportStatus.UNSUPPORTED:
            self.unsupported += 1
        elif verdict == SupportStatus.UNVERIFIED:
            self.unverified += 1


@dataclass(slots=True)
class DraftCompileFlaggedClaim:
    @dataclass(slots=True)
    class ChangeSummary:
        current_verdict: SupportStatus
        previous_verdict: SupportStatus | None = None
        verdict_changed: bool = False
        current_confidence_score: float | None = None
        previous_confidence_score: float | None = None
        confidence_changed: bool = False
        current_primary_anchor: str | None = None
        previous_primary_anchor: str | None = None
        support_changed: bool = False
        current_flags: list[str] = field(default_factory=list)
        previous_flags: list[str] = field(default_factory=list)
        flags_changed: bool = False
        current_reasoning_categories: list[str] = field(default_factory=list)
        previous_reasoning_categories: list[str] = field(default_factory=list)
        reasoning_categories_changed: bool = False

    @dataclass(slots=True)
    class Relationship:
        relationship_type: ClaimGraphRelationshipType
        related_claim_id: UUID
        related_claim_text: str
        reason_code: str | None = None
        reason_text: str | None = None
        confidence_score: float | None = None

    claim_id: UUID
    claim_text: str
    verdict: SupportStatus
    deterministic_flags: list[str]
    primary_anchor: str | None
    draft_sequence: int = 0
    assertion_context: str | None = None
    reasoning: str | None = None
    excluded_links: list[dict[str, str | None]] = field(default_factory=list)
    scope: dict[str, object] | None = None
    latest_verification_run_id: UUID | None = None
    latest_verification_run_at: datetime | None = None
    support_assessments: list[ProviderSupportAssessment] = field(default_factory=list)
    suggested_fix: str | None = None
    confidence_score: float | None = None
    reasoning_categories: list[str] = field(default_factory=list)
    changed_since_last_run: bool = False
    change_summary: ChangeSummary | None = None
    contradiction_flags: list[str] = field(default_factory=list)
    claim_relationships: list[Relationship] = field(default_factory=list)


@dataclass(slots=True)
class DraftCompileResult:
    draft_id: UUID
    total_claims: int
    counts: DraftCompileVerdictCounts
    flagged_claims: list[DraftCompileFlaggedClaim] = field(default_factory=list)


@dataclass(slots=True)
class DraftCompileClaimInput:
    claim_id: UUID
    claim_text: str
    draft_sequence: int
    assertion_context: str


@dataclass(slots=True)
class _CompileClaimOutcome:
    claim: DraftCompileClaimInput
    execution: VerificationExecution


class DraftCompileService:
    """Verify every claim in a draft and aggregate the results into a draft-wide summary."""

    def __init__(
        self,
        session: Session,
        verification_service: ClaimVerificationService | None = None,
    ) -> None:
        self.session = session
        self.verification_service = verification_service or ClaimVerificationService(session)
        self._session_factory = sessionmaker(
            bind=session.get_bind(),
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
        )

    def compile_draft(
        self,
        draft_id: UUID,
        *,
        model_version: str | None = None,
        prompt_version: str | None = None,
        max_concurrency: int = DEFAULT_COMPILE_MAX_CONCURRENCY,
    ) -> DraftCompileResult:
        prepared = self._prepare_draft_compile(draft_id)
        outcomes = self._verify_claims(
            prepared.claims,
            model_version=model_version,
            prompt_version=prompt_version,
            max_concurrency=max_concurrency,
        )
        counts = DraftCompileVerdictCounts()
        flagged_claims: list[DraftCompileFlaggedClaim] = []

        for outcome in outcomes:
            counts.increment(outcome.execution.result.verdict)
            flagged_claim = self._build_flagged_claim(outcome.claim, outcome.execution)
            if flagged_claim is not None:
                flagged_claims.append(flagged_claim)

        return DraftCompileResult(
            draft_id=prepared.draft_id,
            total_claims=len(prepared.claims),
            counts=counts,
            flagged_claims=flagged_claims,
        )

    def _verify_claims(
        self,
        claims: list[DraftCompileClaimInput],
        *,
        model_version: str | None,
        prompt_version: str | None,
        max_concurrency: int,
    ) -> list[_CompileClaimOutcome]:
        resolved_max_concurrency = self._resolve_max_concurrency(len(claims), max_concurrency)
        if resolved_max_concurrency <= 1:
            return [
                self._verify_single_claim(
                    claim,
                    model_version=model_version,
                    prompt_version=prompt_version,
                )
                for claim in claims
            ]

        outcomes_by_index: dict[int, _CompileClaimOutcome] = {}
        with ThreadPoolExecutor(max_workers=resolved_max_concurrency) as executor:
            futures = {
                executor.submit(
                    self._verify_single_claim,
                    claim,
                    model_version=model_version,
                    prompt_version=prompt_version,
                ): index
                for index, claim in enumerate(claims)
            }
            for future in as_completed(futures):
                outcomes_by_index[futures[future]] = future.result()

        return [outcomes_by_index[index] for index in range(len(claims))]

    def _resolve_max_concurrency(self, claim_count: int, requested_max_concurrency: int) -> int:
        if claim_count <= 1:
            return 1
        if requested_max_concurrency <= 1:
            return 1

        bind = self.session.get_bind()
        if bind.dialect.name == "sqlite" and isinstance(self.verification_service, ClaimVerificationService):
            # SQLite-backed tests use shared in-memory connections that are not a good fit
            # for concurrent read/write verification workers.
            return 1

        return min(claim_count, max(1, requested_max_concurrency))

    def _verify_single_claim(
        self,
        claim: DraftCompileClaimInput,
        *,
        model_version: str | None,
        prompt_version: str | None,
    ) -> _CompileClaimOutcome:
        execution = self.verification_service.verify_claim(
            claim.claim_id,
            model_version=model_version,
            prompt_version=prompt_version,
        )
        return _CompileClaimOutcome(
            claim=claim,
            execution=execution,
        )

    def _prepare_draft_compile(self, draft_id: UUID):
        load_session = self._session_factory()
        try:
            draft = self._get_draft(load_session, draft_id)
            if draft is None:
                raise NotFoundError("Draft not found.")
            claims = self._ordered_claims(draft)
            return _PreparedDraftCompile(
                draft_id=draft.id,
                claims=[
                    DraftCompileClaimInput(
                        claim_id=claim.id,
                        claim_text=claim.text,
                        draft_sequence=index,
                        assertion_context=claim.assertion.raw_text,
                    )
                    for index, claim in enumerate(claims, start=1)
                ],
            )
        finally:
            load_session.close()

    def _get_draft(self, session: Session, draft_id: UUID) -> Draft | None:
        stmt = (
            select(Draft)
            .options(
                selectinload(Draft.evidence_bundle),
                selectinload(Draft.assertions).selectinload(Assertion.claim_units),
            )
            .where(Draft.id == draft_id)
        )
        return session.scalar(stmt)

    def _ordered_claims(self, draft: Draft) -> list[ClaimUnit]:
        ordered_assertions = sorted(
            draft.assertions,
            key=lambda assertion: (
                assertion.paragraph_index is None,
                assertion.paragraph_index or 0,
                assertion.sentence_index is None,
                assertion.sentence_index or 0,
                assertion.created_at,
                str(assertion.id),
            ),
        )
        ordered_claims: list[ClaimUnit] = []
        for assertion in ordered_assertions:
            ordered_claims.extend(
                sorted(
                    assertion.claim_units,
                    key=lambda claim: (claim.sequence_order, claim.created_at, str(claim.id)),
                )
            )
        return ordered_claims

    def _build_flagged_claim(
        self,
        claim: DraftCompileClaimInput,
        execution: VerificationExecution,
    ) -> DraftCompileFlaggedClaim | None:
        if execution.result.verdict not in FLAGGED_VERDICTS:
            return None

        parsed_snapshot = parse_verification_support_snapshot(
            execution.run.support_snapshot,
            execution.run.support_snapshot_version,
        )
        scope = None
        if parsed_snapshot.claim_scope is not None:
            scope = {
                "scope_kind": parsed_snapshot.claim_scope.scope_kind,
                "allowed_source_document_count": len(parsed_snapshot.claim_scope.allowed_source_document_ids),
            }

        return DraftCompileFlaggedClaim(
            claim_id=claim.claim_id,
            claim_text=claim.claim_text,
            verdict=execution.result.verdict,
            deterministic_flags=list(execution.result.deterministic_flags),
            primary_anchor=execution.result.primary_anchor,
            draft_sequence=claim.draft_sequence,
            assertion_context=claim.assertion_context,
            reasoning=execution.result.reasoning,
            excluded_links=[
                {
                    "code": link.code,
                    "message": link.message,
                }
                for link in parsed_snapshot.excluded_support_links
            ],
            scope=scope,
            latest_verification_run_id=execution.run.id,
            latest_verification_run_at=execution.run.created_at,
            support_assessments=list(execution.result.support_assessments),
            suggested_fix=execution.result.suggested_fix,
            confidence_score=execution.result.confidence_score,
            reasoning_categories=list(execution.result.reasoning_categories),
        )


@dataclass(slots=True)
class _PreparedDraftCompile:
    draft_id: UUID
    claims: list[DraftCompileClaimInput]
