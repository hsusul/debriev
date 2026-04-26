from dataclasses import dataclass, field

from app.core.enums import ClaimType, SupportStatus


@dataclass(frozen=True, slots=True)
class ExtractionGoldCase:
    name: str
    category: str
    assertion_text: str
    expected_claim_texts: tuple[str, ...]
    expected_claim_types: tuple[ClaimType, ...]


@dataclass(frozen=True, slots=True)
class VerificationSupportItemGold:
    raw_text: str
    page_start: int | None
    line_start: int | None
    page_end: int | None
    line_end: int | None
    speaker: str = "A"
    segment_type: str = "ANSWER_BLOCK"

    @property
    def anchor(self) -> str | None:
        if None in {self.page_start, self.line_start, self.page_end, self.line_end}:
            return None
        return f"p.{self.page_start}:{self.line_start}-{self.page_end}:{self.line_end}"


@dataclass(frozen=True, slots=True)
class VerificationGoldCase:
    name: str
    category: str
    claim_text: str
    support_items: tuple[VerificationSupportItemGold, ...] = field(default_factory=tuple)
    expected_flags: tuple[str, ...] = field(default_factory=tuple)
    expected_verdict: SupportStatus = SupportStatus.UNVERIFIED
    expected_provider_verdict: SupportStatus | None = None
    expected_primary_anchor: str | None = None
    expected_support_roles: tuple[str, ...] | None = None


EXTRACTION_GOLD_CASES: tuple[ExtractionGoldCase, ...] = (
    ExtractionGoldCase(
        name="shared-subject-distinct-predicates",
        category="shared_subject_predicates",
        assertion_text="Doe signed the contract and approved the invoice.",
        expected_claim_texts=("Doe signed the contract", "Doe approved the invoice."),
        expected_claim_types=(ClaimType.FACT, ClaimType.FACT),
    ),
    ExtractionGoldCase(
        name="shared-subject-follow-on-predicate",
        category="shared_subject_predicates",
        assertion_text="Doe signed the contract and resigned.",
        expected_claim_texts=("Doe signed the contract", "Doe resigned."),
        expected_claim_types=(ClaimType.FACT, ClaimType.FACT),
    ),
    ExtractionGoldCase(
        name="shared-subject-comma-predicate-list",
        category="shared_subject_predicates",
        assertion_text="Doe reviewed the contract, signed the declaration, and emailed counsel.",
        expected_claim_texts=(
            "Doe reviewed the contract",
            "Doe signed the declaration",
            "Doe emailed counsel.",
        ),
        expected_claim_types=(ClaimType.FACT, ClaimType.FACT, ClaimType.FACT),
    ),
    ExtractionGoldCase(
        name="shared-object-compound-stays-intact",
        category="coordination_guards",
        assertion_text="Doe reviewed and approved the agreement.",
        expected_claim_texts=("Doe reviewed and approved the agreement.",),
        expected_claim_types=(ClaimType.FACT,),
    ),
    ExtractionGoldCase(
        name="noun-phrase-conjunction-with-later-predicate",
        category="coordination_guards",
        assertion_text="Doe reviewed the purchase and sale agreement and signed the affidavit.",
        expected_claim_texts=(
            "Doe reviewed the purchase and sale agreement",
            "Doe signed the affidavit.",
        ),
        expected_claim_types=(ClaimType.FACT, ClaimType.FACT),
    ),
    ExtractionGoldCase(
        name="quoted-testimony-plus-gloss",
        category="quote_gloss",
        assertion_text='Doe testified, "I signed the agreement," which suggests he knew the terms.',
        expected_claim_texts=('Doe testified, "I signed the agreement,"', "which suggests he knew the terms."),
        expected_claim_types=(ClaimType.QUOTE, ClaimType.INFERENCE),
    ),
    ExtractionGoldCase(
        name="mixed-fact-and-inference",
        category="mixed_fact_inference",
        assertion_text="Doe signed the contract, which suggests he knew the terms.",
        expected_claim_texts=("Doe signed the contract, which suggests he knew the terms.",),
        expected_claim_types=(ClaimType.MIXED,),
    ),
    ExtractionGoldCase(
        name="mixed-fact-and-inference-suggesting",
        category="mixed_fact_inference",
        assertion_text="Doe signed the release, suggesting he understood the waiver.",
        expected_claim_texts=("Doe signed the release, suggesting he understood the waiver.",),
        expected_claim_types=(ClaimType.MIXED,),
    ),
    ExtractionGoldCase(
        name="quoted-conjunction-stays-intact",
        category="quote_gloss",
        assertion_text='Doe testified, "I reviewed the file and signed the agreement."',
        expected_claim_texts=('Doe testified, "I reviewed the file and signed the agreement."',),
        expected_claim_types=(ClaimType.QUOTE,),
    ),
    ExtractionGoldCase(
        name="attorney-paraphrase-embedded-quote-plus-gloss",
        category="quote_gloss",
        assertion_text='Counsel noted that Doe testified, "I signed the agreement," which suggests he accepted the terms.',
        expected_claim_texts=(
            'Counsel noted that Doe testified, "I signed the agreement,"',
            "which suggests he accepted the terms.",
        ),
        expected_claim_types=(ClaimType.QUOTE, ClaimType.INFERENCE),
    ),
    ExtractionGoldCase(
        name="attorney-paraphrase-embedded-quote-without-gloss",
        category="quote_gloss",
        assertion_text='Counsel noted that Doe testified, "I reviewed the file and signed the declaration."',
        expected_claim_texts=('Counsel noted that Doe testified, "I reviewed the file and signed the declaration."',),
        expected_claim_types=(ClaimType.QUOTE,),
    ),
    ExtractionGoldCase(
        name="object-list-conjunction-stays-intact",
        category="coordination_guards",
        assertion_text="Doe met with Smith and Jones at the office.",
        expected_claim_texts=("Doe met with Smith and Jones at the office.",),
        expected_claim_types=(ClaimType.FACT,),
    ),
    ExtractionGoldCase(
        name="short-follow-on-predicate-chain",
        category="shared_subject_predicates",
        assertion_text="Doe signed the contract, resigned, and moved overseas.",
        expected_claim_texts=("Doe signed the contract", "Doe resigned", "Doe moved overseas."),
        expected_claim_types=(ClaimType.FACT, ClaimType.FACT, ClaimType.FACT),
    ),
    ExtractionGoldCase(
        name="later-follow-on-predicate",
        category="structural_assist",
        assertion_text="Doe reviewed the agreement and later signed the declaration.",
        expected_claim_texts=("Doe reviewed the agreement", "Doe later signed the declaration."),
        expected_claim_types=(ClaimType.FACT, ClaimType.FACT),
    ),
    ExtractionGoldCase(
        name="quote-followed-by-later-fact",
        category="structural_assist",
        assertion_text='Doe testified, "I signed the agreement," and later emailed counsel about the closing.',
        expected_claim_texts=('Doe testified, "I signed the agreement,"', "Doe later emailed counsel about the closing."),
        expected_claim_types=(ClaimType.QUOTE, ClaimType.FACT),
    ),
    ExtractionGoldCase(
        name="noun-phrase-coordination-with-later-predicate",
        category="structural_assist",
        assertion_text="Doe reviewed the purchase and sale agreement and related schedules and later signed the affidavit.",
        expected_claim_texts=(
            "Doe reviewed the purchase and sale agreement and related schedules",
            "Doe later signed the affidavit.",
        ),
        expected_claim_types=(ClaimType.FACT, ClaimType.FACT),
    ),
    ExtractionGoldCase(
        name="attorney-paraphrase-quote-then-procedural-step",
        category="structural_assist",
        assertion_text='Counsel wrote that Doe testified, "I reviewed the file," and later moved to strike the objection.',
        expected_claim_texts=(
            'Counsel wrote that Doe testified, "I reviewed the file,"',
            "Counsel later moved to strike the objection.",
        ),
        expected_claim_types=(ClaimType.QUOTE, ClaimType.FACT),
    ),
    ExtractionGoldCase(
        name="shared-subject-predicate-chain-with-noun-phrase-coordination",
        category="structural_assist",
        assertion_text="Doe reviewed the agreement and related schedules and circulated the revised declaration.",
        expected_claim_texts=(
            "Doe reviewed the agreement and related schedules",
            "Doe circulated the revised declaration.",
        ),
        expected_claim_types=(ClaimType.FACT, ClaimType.FACT),
    ),
    ExtractionGoldCase(
        name="embedded-quote-with-inferential-commentary",
        category="structural_assist",
        assertion_text='Counsel argued that Doe testified, "I reviewed the file," suggesting he knew the discrepancy.',
        expected_claim_texts=(
            'Counsel argued that Doe testified, "I reviewed the file,"',
            "suggesting he knew the discrepancy.",
        ),
        expected_claim_types=(ClaimType.QUOTE, ClaimType.INFERENCE),
    ),
    ExtractionGoldCase(
        name="comma-separated-object-list-stays-intact",
        category="coordination_guards",
        assertion_text="Doe reviewed the agreement, related schedules, and draft disclosures.",
        expected_claim_texts=("Doe reviewed the agreement, related schedules, and draft disclosures.",),
        expected_claim_types=(ClaimType.FACT,),
    ),
    ExtractionGoldCase(
        name="temporal-follow-on-predicate-chain",
        category="structural_assist",
        assertion_text="Doe reviewed the agreement, then signed the declaration and emailed counsel.",
        expected_claim_texts=(
            "Doe reviewed the agreement",
            "Doe then signed the declaration",
            "Doe emailed counsel.",
        ),
        expected_claim_types=(ClaimType.FACT, ClaimType.FACT, ClaimType.FACT),
    ),
    ExtractionGoldCase(
        name="exhibit-style-label-preamble-with-predicate-chain",
        category="hybrid_fallback",
        assertion_text="Subject: Contract Status. Doe signed the agreement and later emailed counsel regarding payment.",
        expected_claim_texts=(
            "Doe signed the agreement",
            "Doe later emailed counsel regarding payment.",
        ),
        expected_claim_types=(ClaimType.FACT, ClaimType.FACT),
    ),
    ExtractionGoldCase(
        name="reported-exhibit-predicate-chain",
        category="hybrid_fallback",
        assertion_text=(
            "Counsel contended that the exhibit reflects Doe signed the contract, approved the invoice, "
            "and thereafter forwarded the executed copy to counsel."
        ),
        expected_claim_texts=(
            "Counsel contended that the exhibit reflects Doe signed the contract",
            "Doe approved the invoice",
            "Doe thereafter forwarded the executed copy to counsel.",
        ),
        expected_claim_types=(ClaimType.FACT, ClaimType.FACT, ClaimType.FACT),
    ),
    ExtractionGoldCase(
        name="quote-gloss-with-follow-on-procedural-step",
        category="hybrid_fallback",
        assertion_text=(
            'Counsel argued that Doe testified, "I reviewed the file and signed the agreement," '
            "which suggests he knew the discrepancy, and later moved to compel production."
        ),
        expected_claim_texts=(
            'Counsel argued that Doe testified, "I reviewed the file and signed the agreement,"',
            "which suggests he knew the discrepancy",
            "Counsel later moved to compel production.",
        ),
        expected_claim_types=(ClaimType.QUOTE, ClaimType.INFERENCE, ClaimType.FACT),
    ),
    ExtractionGoldCase(
        name="exhibit-header-quote-follow-on-fact",
        category="structured_extractor",
        assertion_text=(
            'From: Jane Doe. In the March 1 email, Doe wrote, "I signed the release," '
            "and later sent the executed copy to counsel."
        ),
        expected_claim_texts=(
            'In the March 1 email, Doe wrote, "I signed the release,"',
            "Doe later sent the executed copy to counsel.",
        ),
        expected_claim_types=(ClaimType.QUOTE, ClaimType.FACT),
    ),
    ExtractionGoldCase(
        name="nested-reporting-predicate-chain",
        category="structured_extractor",
        assertion_text=(
            "Counsel argued that the exhibit indicates Doe reviewed the agreement, approved the invoice, "
            "and later forwarded the executed copy."
        ),
        expected_claim_texts=(
            "Counsel argued that the exhibit indicates Doe reviewed the agreement",
            "Doe approved the invoice",
            "Doe later forwarded the executed copy.",
        ),
        expected_claim_types=(ClaimType.FACT, ClaimType.FACT, ClaimType.FACT),
    ),
    ExtractionGoldCase(
        name="negated-clause-with-follow-on-fact",
        category="structured_extractor",
        assertion_text="Doe signed the agreement but not the guaranty, and later emailed counsel about the exception.",
        expected_claim_texts=(
            "Doe signed the agreement but not the guaranty",
            "Doe later emailed counsel about the exception.",
        ),
        expected_claim_types=(ClaimType.FACT, ClaimType.FACT),
    ),
)


VERIFICATION_GOLD_CASES: tuple[VerificationGoldCase, ...] = (
    VerificationGoldCase(
        name="missing-citation",
        category="blocking_flags",
        claim_text="Doe signed the contract.",
        expected_flags=("missing_citation",),
        expected_verdict=SupportStatus.UNVERIFIED,
    ),
    VerificationGoldCase(
        name="invalid-anchor",
        category="blocking_flags",
        claim_text="Doe signed the contract.",
        support_items=(
            VerificationSupportItemGold(
                raw_text="A. Doe signed the contract.",
                page_start=10,
                line_start=4,
                page_end=9,
                line_end=1,
            ),
        ),
        expected_flags=("invalid_anchor",),
        expected_verdict=SupportStatus.AMBIGUOUS,
    ),
    VerificationGoldCase(
        name="absolute-qualifier-mismatch",
        category="overstatement",
        claim_text="Smith always approved invoices.",
        support_items=(
            VerificationSupportItemGold(
                raw_text="A. Smith approved the invoice on March 1.",
                page_start=10,
                line_start=4,
                page_end=10,
                line_end=6,
            ),
        ),
        expected_flags=("absolute_qualifier_mismatch", "temporal_scope_mismatch"),
        expected_verdict=SupportStatus.OVERSTATED,
    ),
    VerificationGoldCase(
        name="knowledge-escalation",
        category="escalation",
        claim_text="Doe knew the contract was fraudulent.",
        support_items=(
            VerificationSupportItemGold(
                raw_text="A. Doe signed the contract.",
                page_start=11,
                line_start=3,
                page_end=11,
                line_end=4,
            ),
        ),
        expected_flags=("knowledge_escalation", "needs_human_review"),
        expected_verdict=SupportStatus.AMBIGUOUS,
    ),
    VerificationGoldCase(
        name="causation-escalation",
        category="escalation",
        claim_text="Doe's conduct caused the spill.",
        support_items=(
            VerificationSupportItemGold(
                raw_text="A. Doe opened the valve and the spill happened later.",
                page_start=12,
                line_start=5,
                page_end=12,
                line_end=7,
            ),
        ),
        expected_flags=("causation_escalation", "needs_human_review"),
        expected_verdict=SupportStatus.AMBIGUOUS,
    ),
    VerificationGoldCase(
        name="narrow-support",
        category="narrow_support",
        claim_text="Doe reviewed the contract and approved the invoice.",
        support_items=(
            VerificationSupportItemGold(
                raw_text="A. Doe reviewed the contract.",
                page_start=13,
                line_start=2,
                page_end=13,
                line_end=3,
            ),
        ),
        expected_flags=("narrow_support",),
        expected_verdict=SupportStatus.OVERSTATED,
    ),
    VerificationGoldCase(
        name="contextual-question-only-support",
        category="contextual_support",
        claim_text="Doe signed the contract.",
        support_items=(
            VerificationSupportItemGold(
                raw_text="Q. Did Doe sign the contract?",
                page_start=14,
                line_start=1,
                page_end=14,
                line_end=2,
                speaker="Q",
                segment_type="QUESTION_BLOCK",
            ),
        ),
        expected_flags=("contextual_support_only",),
        expected_verdict=SupportStatus.AMBIGUOUS,
        expected_provider_verdict=SupportStatus.AMBIGUOUS,
        expected_primary_anchor="p.14:1-14:2",
        expected_support_roles=("contextual",),
    ),
    VerificationGoldCase(
        name="strong-answer-support",
        category="direct_support",
        claim_text="Smith signed the contract.",
        support_items=(
            VerificationSupportItemGold(
                raw_text="Q. Did you review the contract?",
                page_start=15,
                line_start=1,
                page_end=15,
                line_end=2,
                speaker="Q",
                segment_type="QUESTION_BLOCK",
            ),
            VerificationSupportItemGold(
                raw_text="A. Smith signed the contract on March 1.",
                page_start=15,
                line_start=3,
                page_end=15,
                line_end=4,
            ),
        ),
        expected_flags=(),
        expected_verdict=SupportStatus.SUPPORTED,
        expected_provider_verdict=SupportStatus.SUPPORTED,
        expected_primary_anchor="p.15:3-15:4",
        expected_support_roles=("contextual", "primary"),
    ),
    VerificationGoldCase(
        name="distributed-multi-support-bundle",
        category="distributed_support",
        claim_text="Doe reviewed the contract and approved the invoice.",
        support_items=(
            VerificationSupportItemGold(
                raw_text="A. Doe reviewed the contract.",
                page_start=16,
                line_start=1,
                page_end=16,
                line_end=2,
            ),
            VerificationSupportItemGold(
                raw_text="A. Doe approved the invoice.",
                page_start=16,
                line_start=3,
                page_end=16,
                line_end=4,
            ),
        ),
        expected_flags=("narrow_support",),
        expected_verdict=SupportStatus.OVERSTATED,
        expected_provider_verdict=SupportStatus.PARTIALLY_SUPPORTED,
        expected_primary_anchor="p.16:1-16:2",
        expected_support_roles=("primary", "secondary"),
    ),
    VerificationGoldCase(
        name="broad-multi-predicate-single-direct-support",
        category="narrow_support",
        claim_text="Doe reviewed the contract, approved the invoice, and emailed counsel.",
        support_items=(
            VerificationSupportItemGold(
                raw_text="A. Doe approved the invoice.",
                page_start=17,
                line_start=1,
                page_end=17,
                line_end=2,
            ),
        ),
        expected_flags=("narrow_support",),
        expected_verdict=SupportStatus.OVERSTATED,
        expected_provider_verdict=SupportStatus.PARTIALLY_SUPPORTED,
        expected_primary_anchor="p.17:1-17:2",
        expected_support_roles=("primary",),
    ),
    VerificationGoldCase(
        name="strong-question-overlap-weak-answer-support",
        category="contextual_support",
        claim_text="Doe signed the contract on March 1.",
        support_items=(
            VerificationSupportItemGold(
                raw_text="Q. Did Doe sign the contract on March 1?",
                page_start=18,
                line_start=1,
                page_end=18,
                line_end=2,
                speaker="Q",
                segment_type="QUESTION_BLOCK",
            ),
            VerificationSupportItemGold(
                raw_text="A. I do not remember whether Doe signed it.",
                page_start=18,
                line_start=3,
                page_end=18,
                line_end=4,
            ),
        ),
        expected_flags=("contextual_support_only",),
        expected_verdict=SupportStatus.AMBIGUOUS,
        expected_provider_verdict=SupportStatus.AMBIGUOUS,
        expected_primary_anchor="p.18:1-18:2",
        expected_support_roles=("contextual", "contextual"),
    ),
    VerificationGoldCase(
        name="distributed-support-no-single-strong-anchor",
        category="distributed_support",
        claim_text="Doe reviewed the contract, approved the invoice, and emailed counsel.",
        support_items=(
            VerificationSupportItemGold(
                raw_text="A. Doe reviewed the contract.",
                page_start=19,
                line_start=1,
                page_end=19,
                line_end=2,
            ),
            VerificationSupportItemGold(
                raw_text="A. Doe approved the invoice.",
                page_start=19,
                line_start=3,
                page_end=19,
                line_end=4,
            ),
            VerificationSupportItemGold(
                raw_text="A. Doe emailed counsel.",
                page_start=19,
                line_start=5,
                page_end=19,
                line_end=6,
            ),
        ),
        expected_flags=("narrow_support",),
        expected_verdict=SupportStatus.OVERSTATED,
        expected_provider_verdict=SupportStatus.PARTIALLY_SUPPORTED,
        expected_primary_anchor="p.19:1-19:2",
        expected_support_roles=("primary", "secondary", "secondary"),
    ),
    VerificationGoldCase(
        name="knowledge-language-circumstantial-support-stays-cautious",
        category="escalation",
        claim_text="Doe knew the shipment was late.",
        support_items=(
            VerificationSupportItemGold(
                raw_text="A. Doe received the email about the shipment delay.",
                page_start=20,
                line_start=1,
                page_end=20,
                line_end=2,
            ),
        ),
        expected_flags=("knowledge_escalation", "needs_human_review"),
        expected_verdict=SupportStatus.AMBIGUOUS,
    ),
    VerificationGoldCase(
        name="causation-language-circumstantial-support-stays-cautious",
        category="escalation",
        claim_text="Doe's delay caused the shutdown.",
        support_items=(
            VerificationSupportItemGold(
                raw_text="A. Doe's delay preceded the shutdown.",
                page_start=21,
                line_start=1,
                page_end=21,
                line_end=2,
            ),
        ),
        expected_flags=("causation_escalation", "needs_human_review"),
        expected_verdict=SupportStatus.AMBIGUOUS,
    ),
    VerificationGoldCase(
        name="partial-support-for-one-clause-of-broader-claim",
        category="partial_clause_support",
        claim_text="Doe signed the contract and delivered the notice the same day.",
        support_items=(
            VerificationSupportItemGold(
                raw_text="A. Doe signed the contract that day.",
                page_start=22,
                line_start=1,
                page_end=22,
                line_end=2,
            ),
        ),
        expected_flags=("narrow_support",),
        expected_verdict=SupportStatus.OVERSTATED,
        expected_provider_verdict=SupportStatus.PARTIALLY_SUPPORTED,
        expected_primary_anchor="p.22:1-22:2",
        expected_support_roles=("primary",),
    ),
    VerificationGoldCase(
        name="multiple-weak-supports-stay-cautious",
        category="weak_support",
        claim_text="Doe approved the invoice on March 1.",
        support_items=(
            VerificationSupportItemGold(
                raw_text="A. I do not remember whether Doe approved the invoice.",
                page_start=23,
                line_start=1,
                page_end=23,
                line_end=2,
            ),
            VerificationSupportItemGold(
                raw_text="A. I am not sure if it happened on March 1.",
                page_start=23,
                line_start=3,
                page_end=23,
                line_end=4,
            ),
        ),
        expected_flags=(),
        expected_verdict=SupportStatus.AMBIGUOUS,
        expected_provider_verdict=SupportStatus.AMBIGUOUS,
        expected_primary_anchor="p.23:1-23:2",
        expected_support_roles=("contextual", "contextual"),
    ),
    VerificationGoldCase(
        name="near-contradiction-between-support-items",
        category="support_tension",
        claim_text="Doe signed the contract.",
        support_items=(
            VerificationSupportItemGold(
                raw_text="A. Doe signed the contract.",
                page_start=24,
                line_start=1,
                page_end=24,
                line_end=2,
            ),
            VerificationSupportItemGold(
                raw_text="A. I do not recall whether Doe signed the contract.",
                page_start=24,
                line_start=3,
                page_end=24,
                line_end=4,
            ),
        ),
        expected_flags=(),
        expected_verdict=SupportStatus.SUPPORTED,
        expected_provider_verdict=SupportStatus.PARTIALLY_SUPPORTED,
        expected_primary_anchor="p.24:1-24:2",
        expected_support_roles=("primary", "contextual"),
    ),
    VerificationGoldCase(
        name="question-framing-strong-answer-equivocal",
        category="contextual_support",
        claim_text="Doe delivered the notice on March 1.",
        support_items=(
            VerificationSupportItemGold(
                raw_text="Q. Did Doe deliver the notice on March 1?",
                page_start=25,
                line_start=1,
                page_end=25,
                line_end=2,
                speaker="Q",
                segment_type="QUESTION_BLOCK",
            ),
            VerificationSupportItemGold(
                raw_text="A. I cannot say whether he delivered it that day.",
                page_start=25,
                line_start=3,
                page_end=25,
                line_end=4,
            ),
        ),
        expected_flags=("contextual_support_only",),
        expected_verdict=SupportStatus.AMBIGUOUS,
        expected_provider_verdict=SupportStatus.AMBIGUOUS,
        expected_primary_anchor="p.25:1-25:2",
        expected_support_roles=("contextual", "contextual"),
    ),
    VerificationGoldCase(
        name="temporal-relation-not-causal-sufficiency",
        category="temporal_vs_causal",
        claim_text="Doe's email caused the shutdown.",
        support_items=(
            VerificationSupportItemGold(
                raw_text="A. Doe sent the email before the shutdown happened.",
                page_start=26,
                line_start=1,
                page_end=26,
                line_end=2,
            ),
        ),
        expected_flags=("causation_escalation", "needs_human_review"),
        expected_verdict=SupportStatus.AMBIGUOUS,
    ),
)
