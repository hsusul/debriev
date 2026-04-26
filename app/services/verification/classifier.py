"""Verification orchestration and persistence."""

from dataclasses import dataclass, field
from uuid import UUID

from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings, get_settings
from app.core.enums import SupportStatus
from app.core.exceptions import NotFoundError
from app.models import (
    CURRENT_SUPPORT_SNAPSHOT_VERSION,
    ClaimUnit,
    Segment,
    SupportLink,
    VerificationRun,
)
from app.repositories.claims import ClaimsRepository
from app.repositories.evidence_bundles import EvidenceBundleRepository
from app.repositories.links import LinksRepository
from app.repositories.verification import VerificationRepository
from app.services.llm.anthropic_provider import AnthropicProvider
from app.services.llm.base import (
    ProviderRequest,
    ProviderResponse,
    ProviderSupportAssessment,
    ProviderSupportItem,
    ProviderUnavailableError,
    VerificationProvider,
)
from app.services.llm.openai_provider import OpenAIProvider
from app.services.linking.validation import ClaimScopeContext, LinkRevalidationResult, LinkValidationService
from app.services.verification.context_builder import ClaimContext, build_claim_context
from app.services.verification.heuristics import evaluate_heuristics
from app.services.verification.reasoning_categories import classify_reasoning_categories

BLOCKING_FLAGS = frozenset({"missing_citation", "invalid_anchor", "quote_mismatch_placeholder"})
WARNING_VERDICT_CEILINGS: dict[str, SupportStatus] = {
    "absolute_qualifier_mismatch": SupportStatus.PARTIALLY_SUPPORTED,
    "temporal_scope_mismatch": SupportStatus.PARTIALLY_SUPPORTED,
    "subject_mismatch": SupportStatus.AMBIGUOUS,
    "knowledge_escalation": SupportStatus.AMBIGUOUS,
    "causation_escalation": SupportStatus.AMBIGUOUS,
    "contextual_support_only": SupportStatus.AMBIGUOUS,
    "narrow_support": SupportStatus.PARTIALLY_SUPPORTED,
}
SUPPORT_STRENGTH = {
    SupportStatus.UNVERIFIED: 0,
    SupportStatus.UNSUPPORTED: 1,
    SupportStatus.CONTRADICTED: 1,
    SupportStatus.AMBIGUOUS: 2,
    SupportStatus.OVERSTATED: 2,
    SupportStatus.PARTIALLY_SUPPORTED: 3,
    SupportStatus.SUPPORTED: 4,
}


@dataclass(slots=True)
class VerificationResult:
    """Normalized verification result stored in the database."""

    model_version: str
    prompt_version: str
    deterministic_flags: list[str]
    verdict: SupportStatus
    reasoning: str
    suggested_fix: str | None
    confidence_score: float | None
    support_assessments: list[ProviderSupportAssessment] = field(default_factory=list)
    primary_anchor: str | None = None
    reasoning_categories: list[str] = field(default_factory=list)


@dataclass(slots=True)
class VerificationExecution:
    """Fresh verification execution with persisted run and structured result."""

    run: VerificationRun
    result: VerificationResult


@dataclass(slots=True)
class PreparedVerificationInput:
    """All data needed to verify a claim outside of a database session."""

    claim_id: UUID
    claim: ClaimUnit
    claim_scope: ClaimScopeContext
    allowed_source_document_ids: list[UUID]
    links: list[SupportLink]
    segments: list[Segment]
    context: ClaimContext
    invalid_links: list[LinkRevalidationResult] = field(default_factory=list)


class VerificationClassifier:
    """Hybrid verification classifier with deterministic-first behavior."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def verify(
        self,
        claim: ClaimUnit,
        links: list[SupportLink],
        segments: list[Segment],
        *,
        model_version: str | None = None,
        prompt_version: str | None = None,
    ) -> VerificationResult:
        context = build_claim_context(claim, links, segments)
        return self.verify_prepared(
            PreparedVerificationInput(
                claim_id=claim.id,
                claim=claim,
                claim_scope=_build_inline_claim_scope(claim),
                allowed_source_document_ids=[],
                links=links,
                segments=segments,
                context=context,
            ),
            model_version=model_version,
            prompt_version=prompt_version,
        )

    def verify_prepared(
        self,
        prepared: PreparedVerificationInput,
        *,
        model_version: str | None = None,
        prompt_version: str | None = None,
    ) -> VerificationResult:
        heuristic = evaluate_heuristics(prepared.claim, prepared.links, prepared.segments)
        invalid_link_summary = _apply_invalid_link_revalidation(heuristic, prepared.invalid_links)
        reasoning_parts = list(heuristic.reasoning)
        if invalid_link_summary:
            reasoning_parts.insert(0, invalid_link_summary)
        resolved_model_version = model_version or self.settings.verification_model_version
        resolved_prompt_version = prompt_version or self.settings.prompt_version
        support_assessments: list[ProviderSupportAssessment] = []
        primary_anchor: str | None = None

        provider = self._build_provider(model_version=resolved_model_version)
        if provider is None:
            reasoning_parts.append("No LLM provider configured; returning deterministic placeholder output.")
        elif self._has_blocking_flags(heuristic.flags):
            reasoning_parts.append("Blocking deterministic flags prevent provider refinement for this claim.")
        else:
            request = _build_provider_request(prepared.context, heuristic.flags)
            try:
                provider_result = provider.verify(request)
            except ProviderUnavailableError as exc:
                heuristic.flags.append("llm_provider_stubbed")
                reasoning_parts.append(str(exc))
            else:
                resolved_model_version = provider.model_version
                support_assessments = provider_result.support_assessments
                primary_anchor = provider_result.primary_anchor
                reasoning_parts.append(provider_result.reasoning)
                support_summary = _render_provider_support_summary(provider_result)
                if support_summary:
                    reasoning_parts.append(support_summary)
                provider_verdict = self._apply_support_assessment_caps(
                    provider_result.verdict,
                    provider_result.support_assessments,
                )
                heuristic.verdict = self._apply_warning_caps(provider_verdict, heuristic.flags)
                heuristic.suggested_fix = provider_result.suggested_fix or heuristic.suggested_fix
                if provider_result.confidence_score is not None:
                    heuristic.confidence_score = provider_result.confidence_score

        return VerificationResult(
            model_version=resolved_model_version,
            prompt_version=resolved_prompt_version,
            deterministic_flags=heuristic.flags,
            reasoning_categories=classify_reasoning_categories(
                deterministic_flags=heuristic.flags,
                verdict=heuristic.verdict,
            ),
            verdict=heuristic.verdict,
            reasoning=" ".join(reasoning_parts),
            suggested_fix=heuristic.suggested_fix,
            confidence_score=heuristic.confidence_score,
            support_assessments=support_assessments,
            primary_anchor=primary_anchor,
        )

    def _build_provider(self, *, model_version: str) -> VerificationProvider | None:
        if self.settings.llm_provider == "openai" and self.settings.openai_api_key:
            return OpenAIProvider(api_key=self.settings.openai_api_key, model_version=model_version)
        if self.settings.llm_provider == "anthropic" and self.settings.anthropic_api_key:
            return AnthropicProvider(api_key=self.settings.anthropic_api_key, model_version=model_version)
        return None

    def _has_blocking_flags(self, flags: list[str]) -> bool:
        return any(flag in BLOCKING_FLAGS for flag in flags)

    def _apply_warning_caps(self, verdict: SupportStatus, flags: list[str]) -> SupportStatus:
        capped_verdict = verdict
        for flag in flags:
            ceiling = WARNING_VERDICT_CEILINGS.get(flag)
            if ceiling is None:
                continue
            if SUPPORT_STRENGTH[capped_verdict] > SUPPORT_STRENGTH[ceiling]:
                capped_verdict = ceiling
        return capped_verdict

    def _apply_support_assessment_caps(
        self,
        verdict: SupportStatus,
        assessments: list[ProviderSupportAssessment],
    ) -> SupportStatus:
        if assessments and all(assessment.role == "contextual" for assessment in assessments):
            if SUPPORT_STRENGTH[verdict] > SUPPORT_STRENGTH[SupportStatus.AMBIGUOUS]:
                return SupportStatus.AMBIGUOUS
        return verdict


def _build_provider_request(context: ClaimContext, heuristic_flags: list[str]) -> ProviderRequest:
    return ProviderRequest(
        claim_text=context.claim_text,
        support_items=[
            ProviderSupportItem(
                segment_id=item.segment_id,
                anchor=item.anchor,
                evidence_role=item.evidence_role,
                speaker=item.speaker,
                segment_type=item.segment_type,
                raw_text=item.raw_text,
                normalized_text=item.normalized_text,
            )
            for item in context.support_items
        ],
        context=context.segment_bundle,
        citations=context.citations,
        heuristic_flags=heuristic_flags,
    )


def _render_provider_support_summary(provider_result: ProviderResponse) -> str:
    parts: list[str] = []
    if provider_result.primary_anchor:
        parts.append(f"Primary support anchor: {provider_result.primary_anchor}.")
    if provider_result.support_assessments:
        rendered = "; ".join(
            f"{assessment.anchor} [{assessment.role}]: {assessment.contribution}"
            for assessment in provider_result.support_assessments
        )
        parts.append(f"Structured support reasoning: {rendered}.")
    return " ".join(parts)


def _apply_invalid_link_revalidation(
    heuristic,
    invalid_links: list[LinkRevalidationResult],
) -> str | None:
    if not invalid_links:
        return None

    for result in invalid_links:
        if result.code and result.code not in heuristic.flags:
            heuristic.flags.append(result.code)

    if "missing_citation" in heuristic.flags:
        heuristic.reasoning = [
            "No valid support links remain after excluding invalid existing links."
        ]
        heuristic.suggested_fix = (
            "Remove invalid support links and relink the claim to in-scope record support."
        )
    elif heuristic.suggested_fix is None:
        heuristic.suggested_fix = "Review invalid support links and relink the claim to in-scope record support."

    counts_by_code: dict[str, int] = {}
    for result in invalid_links:
        code = result.code or "invalid_support_link"
        counts_by_code[code] = counts_by_code.get(code, 0) + 1

    rendered_counts = ", ".join(
        f"{count} {code.replace('_', ' ')}"
        for code, count in sorted(counts_by_code.items())
    )
    return f"Excluded {len(invalid_links)} invalid support link(s) from reasoning: {rendered_counts}."


def _build_inline_claim_scope(claim: ClaimUnit) -> ClaimScopeContext:
    assertion = getattr(claim, "assertion", None)
    draft = getattr(assertion, "draft", None) if assertion is not None else None
    draft_id = getattr(draft, "id", None) or getattr(assertion, "draft_id", None) or claim.id
    matter_id = getattr(draft, "matter_id", None) or claim.id
    evidence_bundle_id = getattr(draft, "evidence_bundle_id", None)
    return ClaimScopeContext(
        claim_id=claim.id,
        draft_id=draft_id,
        matter_id=matter_id,
        evidence_bundle_id=evidence_bundle_id,
    )


def _build_support_snapshot(
    prepared: PreparedVerificationInput,
    result: VerificationResult,
) -> dict[str, object]:
    return _build_support_snapshot_v1(prepared, result)


def _build_support_snapshot_v1(
    prepared: PreparedVerificationInput,
    result: VerificationResult,
) -> dict[str, object]:
    support_item_by_segment_id = {item.segment_id: item for item in prepared.context.support_items}
    segment_by_id = {segment.id: segment for segment in prepared.segments}

    valid_support_links: list[dict[str, object]] = []
    for link in prepared.links:
        segment = getattr(link, "segment", None)
        support_item = support_item_by_segment_id.get(link.segment_id)
        valid_support_links.append(
            {
                "link_id": str(link.id),
                "claim_id": str(link.claim_unit_id),
                "segment_id": str(link.segment_id),
                "source_document_id": str(segment.source_document_id) if segment is not None else None,
                "sequence_order": link.sequence_order,
                "link_type": link.link_type.value,
                "citation_text": link.citation_text,
                "user_confirmed": link.user_confirmed,
                "anchor": support_item.anchor if support_item is not None else (link.citation_text or None),
                "evidence_role": support_item.evidence_role if support_item is not None else None,
            }
        )

    support_items = [
        {
            "order": index,
            "segment_id": str(item.segment_id),
            "source_document_id": str(segment_by_id[item.segment_id].source_document_id),
            "anchor": item.anchor,
            "evidence_role": item.evidence_role,
            "speaker": item.speaker,
            "segment_type": item.segment_type,
            "raw_text": item.raw_text,
            "normalized_text": item.normalized_text,
        }
        for index, item in enumerate(prepared.context.support_items, start=1)
    ]

    excluded_support_links = [
        {
            "link_id": str(result_.link_id),
            "claim_id": str(result_.claim_id),
            "segment_id": str(result_.segment_id),
            "code": result_.code,
            "message": result_.message,
        }
        for result_ in prepared.invalid_links
    ]

    support_assessments = [
        {
            "segment_id": str(assessment.segment_id),
            "anchor": assessment.anchor,
            "role": assessment.role,
            "contribution": assessment.contribution,
        }
        for assessment in result.support_assessments
    ]

    scope_kind = "bundle" if prepared.claim_scope.evidence_bundle_id is not None else "matter_fallback"
    return {
        "claim_scope": {
            "claim_id": str(prepared.claim_scope.claim_id),
            "draft_id": str(prepared.claim_scope.draft_id),
            "matter_id": str(prepared.claim_scope.matter_id),
            "evidence_bundle_id": (
                str(prepared.claim_scope.evidence_bundle_id)
                if prepared.claim_scope.evidence_bundle_id is not None
                else None
            ),
            "scope_kind": scope_kind,
            "allowed_source_document_ids": [str(source_id) for source_id in prepared.allowed_source_document_ids],
        },
        "valid_support_links": valid_support_links,
        "excluded_support_links": excluded_support_links,
        "support_items": support_items,
        "citations": list(prepared.context.citations),
        "provider_output": {
            "primary_anchor": result.primary_anchor,
            "support_assessments": support_assessments,
        },
    }


class ClaimVerificationService:
    """Coordinates claim verification and immutable run persistence."""

    def __init__(self, session: Session, classifier: VerificationClassifier | None = None) -> None:
        self.session = session
        self.classifier = classifier or VerificationClassifier()
        self._session_factory = sessionmaker(
            bind=session.get_bind(),
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
        )

    def verify_claim(
        self,
        claim_id: UUID,
        *,
        model_version: str | None = None,
        prompt_version: str | None = None,
    ) -> VerificationExecution:
        prepared = self.prepare_verification_input(claim_id)
        result = self.classifier.verify_prepared(
            prepared,
            model_version=model_version,
            prompt_version=prompt_version,
        )
        run = self.persist_verification_result(prepared, result)
        return VerificationExecution(run=run, result=result)

    def prepare_verification_input(self, claim_id: UUID) -> PreparedVerificationInput:
        load_session = self._session_factory()
        try:
            claims = ClaimsRepository(load_session)
            evidence_bundles = EvidenceBundleRepository(load_session)
            links_repository = LinksRepository(load_session)
            link_validation = LinkValidationService(load_session)
            claim = claims.get(claim_id)
            if claim is None:
                raise NotFoundError("Claim unit not found.")
            claim_scope = link_validation.get_claim_scope_context(claim_id)
            if claim_scope is None:
                raise NotFoundError("Claim unit not found.")

            links = links_repository.list_by_claim(claim_id)
            revalidation_results = link_validation.revalidate_links(links)
            valid_links: list[SupportLink] = []
            invalid_links: list[LinkRevalidationResult] = []
            for link, result in zip(links, revalidation_results, strict=False):
                if result.is_valid:
                    valid_links.append(link)
                else:
                    invalid_links.append(result)

            segments = [link.segment for link in valid_links if link.segment is not None]
            context = build_claim_context(claim, valid_links, segments)
            allowed_source_document_ids = evidence_bundles.resolve_allowed_source_document_ids_for_draft(
                claim_scope.draft_id
            )
            load_session.expunge_all()
            return PreparedVerificationInput(
                claim_id=claim.id,
                claim=claim,
                claim_scope=claim_scope,
                allowed_source_document_ids=allowed_source_document_ids,
                links=valid_links,
                segments=segments,
                context=context,
                invalid_links=invalid_links,
            )
        finally:
            load_session.close()

    def persist_verification_result(
        self,
        prepared: PreparedVerificationInput,
        result: VerificationResult,
    ) -> VerificationRun:
        support_snapshot_version = CURRENT_SUPPORT_SNAPSHOT_VERSION
        support_snapshot = _build_support_snapshot(prepared, result)
        write_session = self._session_factory()
        try:
            claims = ClaimsRepository(write_session)
            runs = VerificationRepository(write_session)
            claim = claims.get(prepared.claim_id)
            if claim is None:
                raise NotFoundError("Claim unit not found.")

            claim.support_status = result.verdict
            run = runs.create(
                prepared.claim_id,
                model_version=result.model_version,
                prompt_version=result.prompt_version,
                deterministic_flags=result.deterministic_flags,
                reasoning_categories=result.reasoning_categories,
                verdict=result.verdict,
                reasoning=result.reasoning,
                support_snapshot_version=support_snapshot_version,
                support_snapshot=support_snapshot,
                suggested_fix=result.suggested_fix,
                confidence_score=result.confidence_score,
            )
            write_session.commit()
            return run
        except Exception:
            write_session.rollback()
            raise
        finally:
            write_session.close()
