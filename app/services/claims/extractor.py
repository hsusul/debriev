"""Lightweight claim extraction heuristics."""

from collections.abc import Sequence
from dataclasses import dataclass
import re
from typing import Literal
from uuid import UUID

from sqlalchemy.orm import Session

from app.core.enums import ClaimType
from app.core.exceptions import NotFoundError
from app.repositories.assertions import AssertionRepository
from app.repositories.claims import ClaimsRepository
from app.services.claims.structured_extractor import extract_structured_parts
from app.services.parsing.normalization import normalize_for_match

DOUBLE_QUOTE_RE = re.compile(r'"([^"\n]+)"|“([^”\n]+)”')
LEADING_ADVERBIAL_RE = re.compile(r"^(?:later|then|thereafter|subsequently|promptly|immediately|also)\b[\s,]*", re.IGNORECASE)
LEADING_TEMPORAL_PHRASE_RE = re.compile(
    r"^(?:after|before|during|following|upon)\b[^,]{0,80},\s*",
    re.IGNORECASE,
)
LABEL_PREAMBLE_RE = re.compile(r"^\s*[A-Z][A-Za-z0-9 /&()#._-]{1,24}:\s*[^.?!\n]{0,80}\s*$")
REPORTING_PREFIX_RE = re.compile(
    r"^(?:(?:[A-Z][A-Za-z']+(?:\s+[A-Z][A-Za-z']+){0,2})|"
    r"(?:the|this)\s+(?:email|exhibit|letter|memo|message|document|record))\s+"
    r"(?:states?|stated|shows?|showed|reflects?|reflected|indicates?|indicated|"
    r"notes?|noted|says?|said|wrote|argued|contended|asserted|maintained)\s+(?:that\s+)?",
    re.IGNORECASE,
)
INFERENCE_PATTERNS = (
    re.compile(r"\bbecause\b"),
    re.compile(r"\btherefore\b"),
    re.compile(r"\bthus\b"),
    re.compile(r"\bso\b"),
    re.compile(r"\bsuggest(?:s|ed|ing)?\b"),
    re.compile(r"\bshow(?:s|ed|ing)?\b"),
    re.compile(r"\bdemonstrat(?:es|ed|ing)?\b"),
    re.compile(r"\bimpl(?:ies|ied|y)\b"),
    re.compile(r"\blikely\b"),
    re.compile(r"\bappears?\b"),
    re.compile(r"\bknew\b"),
    re.compile(r"\bunderstood\b"),
    re.compile(r"\bintended\b"),
    re.compile(r"\bshould have known\b"),
)
GLOSS_INFERENCE_PATTERNS = (
    re.compile(r"^\s*(?:which|suggesting|showing|meaning|therefore|thus|so)\b"),
    re.compile(r"\b(suggest(?:s|ed|ing)?|show(?:s|ed|ing)?|demonstrat(?:es|ed|ing)?|impl(?:ies|ied|y))\b"),
)
CLAUSE_SPLIT_RE = re.compile(r"\s*;\s*")
WORD_RE = re.compile(r"[a-z0-9']+")
RAW_WORD_RE = re.compile(r"\b[\w']+\b")
LEADING_COORDINATOR_RE = re.compile(r"^(?:and|but)\s+", re.IGNORECASE)
VERB_HINTS = {
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "am",
    "said",
    "told",
    "knew",
    "signed",
    "approved",
    "admitted",
    "caused",
    "emailed",
    "received",
    "sent",
    "observed",
    "saw",
    "testified",
    "explained",
    "wrote",
    "argued",
    "asserted",
    "contended",
    "maintained",
    "ordered",
    "failed",
    "did",
    "had",
    "has",
    "have",
    "knew",
    "understood",
    "intended",
    "reviewed",
    "circulated",
    "met",
    "resigned",
    "opened",
    "closed",
    "paid",
    "spoke",
    "agreed",
}
AMBIGUOUS_PARTICIPLE_HINTS = {"related", "alleged", "revised", "attached", "written", "pending", "underlying"}
TRANSITIVE_VERB_HINTS = {
    "signed",
    "approved",
    "reviewed",
    "emailed",
    "received",
    "sent",
    "observed",
    "ordered",
    "circulated",
    "opened",
    "closed",
}
SUBJECT_PRONOUNS = {"he", "she", "they", "it", "we", "i", "you"}
ExtractionStrategy = Literal["legacy", "structured"]
ExtractionMode = Literal["auto", "legacy", "structured"]
DEFAULT_EXTRACTION_MODE: ExtractionMode = "auto"
CURRENT_EXTRACTION_VERSION = 1


@dataclass(slots=True)
class ExtractedClaim:
    """Normalized, ordered claim candidate."""

    text: str
    normalized_text: str
    claim_type: ClaimType
    sequence_order: int


@dataclass(slots=True)
class ExtractionPlan:
    """Structured extraction output with the strategy used."""

    strategy: ExtractionStrategy
    claims: list[ExtractedClaim]


def classify_claim_type(text: str) -> ClaimType:
    """Classify a claim with lightweight heuristics."""

    lowered = normalize_for_match(text)
    has_quote = bool(extract_double_quoted_spans(text))
    has_inference = any(pattern.search(lowered) for pattern in INFERENCE_PATTERNS)
    has_fact_and_inference = bool(
        re.search(r",\s*(?:which|showing|suggesting|therefore|thus|so)\b", lowered)
        or re.search(r"\b(?:and|but)\s+(?:therefore|thus|so)\b", lowered)
    )
    if has_quote and has_inference:
        return ClaimType.MIXED
    if has_fact_and_inference:
        return ClaimType.MIXED
    if has_quote:
        return ClaimType.QUOTE
    if has_inference:
        return ClaimType.INFERENCE
    return ClaimType.FACT


def extract_claim_candidates(
    text: str,
    *,
    mode: ExtractionMode = DEFAULT_EXTRACTION_MODE,
) -> list[ExtractedClaim]:
    """Split an assertion into narrow claim candidates."""

    return extract_claim_candidates_with_strategy(text, mode=mode).claims


def build_extraction_snapshot(
    *,
    assertion_id: UUID,
    source_assertion_text: str,
    normalized_assertion_text: str,
    extraction_strategy: ExtractionStrategy,
    extraction_version: int,
    claims: Sequence[object],
) -> dict[str, object]:
    """Build a persisted explanation of the extracted claim set."""

    return {
        "assertion_id": str(assertion_id),
        "source_assertion_text": source_assertion_text,
        "normalized_assertion_text": normalized_assertion_text,
        "extraction_strategy": extraction_strategy,
        "extraction_version": extraction_version,
        "claims": [
            {
                "claim_id": str(getattr(claim, "id")) if getattr(claim, "id", None) is not None else None,
                "text": claim.text,
                "normalized_text": claim.normalized_text,
                "claim_type": claim.claim_type.value,
                "sequence_order": claim.sequence_order,
            }
            for claim in claims
        ],
    }


def extract_claim_candidates_with_strategy(
    text: str,
    *,
    mode: ExtractionMode = DEFAULT_EXTRACTION_MODE,
) -> ExtractionPlan:
    """Run claim extraction with an explicit legacy/structured strategy boundary."""

    strategy = select_extraction_strategy(text, mode=mode)
    if strategy == "structured":
        raw_parts = extract_structured_parts(text)
    else:
        raw_parts = _extract_legacy_parts(text)

    if not raw_parts:
        raw_parts = [text.strip()]

    claims = _build_extracted_claims(raw_parts, original_text=text)
    return ExtractionPlan(strategy=strategy, claims=claims)


def select_extraction_strategy(
    text: str,
    *,
    mode: ExtractionMode = DEFAULT_EXTRACTION_MODE,
) -> ExtractionStrategy:
    """Choose the extraction strategy used for the current assertion."""

    if mode == "legacy":
        return "legacy"
    if mode == "structured":
        return "structured"
    return "structured" if _should_use_structured_strategy(text) else "legacy"


def extract_legacy_claim_candidates(text: str) -> list[ExtractedClaim]:
    """Return claim candidates from the legacy heuristic-heavy extractor."""

    return _build_extracted_claims(_extract_legacy_parts(text), original_text=text)


def extract_structured_claim_candidates(text: str) -> list[ExtractedClaim]:
    """Return claim candidates from the structured extractor."""

    return _build_extracted_claims(extract_structured_parts(text), original_text=text)


def _should_use_structured_strategy(text: str) -> bool:
    """Choose the structured extractor for harder or higher-risk assertion shapes."""

    delimiter_count = len(_find_delimiter_positions_outside_quotes(text, ",")) + len(
        _find_delimiter_positions_outside_quotes(text, " and ")
    )

    if _has_nonpropositional_label_preamble(text):
        return True
    if extract_double_quoted_spans(text) and delimiter_count >= 1:
        return True
    if _has_complex_quote_gloss_follow_on(text):
        return True
    if delimiter_count >= 2 and _has_nested_reporting_prefix(text):
        return True
    if delimiter_count >= 3 and _has_dense_predicate_chain(text):
        return True
    if delimiter_count >= 1 and re.search(r"\b(?:later|then|thereafter|except|but not)\b", normalize_for_match(text)):
        return True
    return False


def _extract_legacy_parts(text: str) -> list[str]:
    """Run the legacy heuristic-first extractor, including its local fallback logic."""

    strategy = _select_legacy_internal_strategy(text)
    if strategy == "structured_fallback":
        raw_parts = _extract_structured_fallback_parts(text)
        if raw_parts:
            return raw_parts
    return _extract_heuristic_parts(text)


def _select_legacy_internal_strategy(text: str) -> Literal["heuristic", "structured_fallback"]:
    """Choose between the historical heuristic path and its narrow local fallback."""

    delimiter_count = len(_find_delimiter_positions_outside_quotes(text, ",")) + len(
        _find_delimiter_positions_outside_quotes(text, " and ")
    )

    if _has_nonpropositional_label_preamble(text):
        return "structured_fallback"
    if _has_complex_quote_gloss_follow_on(text):
        return "structured_fallback"
    if delimiter_count >= 2 and _has_nested_reporting_prefix(text) and not extract_double_quoted_spans(text):
        return "structured_fallback"
    if delimiter_count >= 3 and _has_dense_predicate_chain(text):
        return "structured_fallback"
    return "heuristic"


def _extract_heuristic_parts(text: str) -> list[str]:
    raw_parts: list[str] = []
    for part in CLAUSE_SPLIT_RE.split(text):
        stripped = part.strip(" ,")
        if not stripped:
            continue
        for quote_part in _maybe_split_quote_gloss(stripped):
            for conjunction_part in _maybe_split_on_conjunction(quote_part):
                raw_parts.extend(_maybe_split_on_comma_predicates(conjunction_part))

    if raw_parts:
        return raw_parts
    if text.strip():
        return [text.strip()]
    return []


def _extract_structured_fallback_parts(text: str) -> list[str]:
    working_text = _strip_nonpropositional_label_preamble(text)
    if not working_text:
        working_text = text.strip()
    if not working_text:
        return []

    quote_gloss_parts = _maybe_split_quote_gloss(working_text)
    if len(quote_gloss_parts) == 2 and _has_follow_on_fact_clause(
        quote_gloss_parts[1],
        default_subject_prefix=_extract_outer_subject_prefix(quote_gloss_parts[0]),
    ):
        return [quote_gloss_parts[0], *_structured_split_piece(
            quote_gloss_parts[1],
            default_subject_prefix=_extract_outer_subject_prefix(quote_gloss_parts[0]),
        )]

    return _structured_split_piece(working_text)


def _structured_split_piece(text: str, *, default_subject_prefix: str | None = None) -> list[str]:
    pieces = [text.strip(" ,")]
    structured_parts: list[str] = []

    for piece in pieces:
        if not piece:
            continue

        subject_prefix = _extract_structured_subject_prefix(piece) or default_subject_prefix or _extract_subject_prefix(piece)
        comma_parts = _maybe_split_on_comma_predicates(piece, subject_prefix_override=subject_prefix)

        conjunction_parts: list[str] = []
        for comma_part in comma_parts:
            per_part_subject = (
                _extract_structured_subject_prefix(comma_part)
                or default_subject_prefix
                or _extract_subject_prefix(comma_part)
                or subject_prefix
            )
            conjunction_parts.extend(
                _maybe_split_on_conjunction(comma_part, subject_prefix_override=per_part_subject)
            )

        structured_parts.extend(conjunction_parts or comma_parts)

    cleaned_parts = [part.strip(" ,") for part in structured_parts if part.strip(" ,")]
    return cleaned_parts or [working for working in [text.strip(" ,")] if working]


def _build_extracted_claims(raw_parts: list[str], *, original_text: str) -> list[ExtractedClaim]:
    claims: list[ExtractedClaim] = []
    for index, part in enumerate(raw_parts, start=1):
        cleaned = part.strip(" ,")
        if not cleaned:
            continue
        claims.append(
            ExtractedClaim(
                text=cleaned,
                normalized_text=normalize_for_match(cleaned),
                claim_type=classify_claim_type(cleaned),
                sequence_order=index,
            )
        )

    if claims:
        return claims

    fallback_text = original_text.strip()
    if not fallback_text:
        return []
    return [
        ExtractedClaim(
            text=fallback_text,
            normalized_text=normalize_for_match(fallback_text),
            claim_type=classify_claim_type(fallback_text),
            sequence_order=1,
        )
    ]


def _maybe_split_on_conjunction(text: str, *, subject_prefix_override: str | None = None) -> list[str]:
    split_indexes = _find_delimiter_positions_outside_quotes(text, " and ")
    if not split_indexes:
        return [text]

    subject_prefix = subject_prefix_override or _extract_subject_prefix(text)
    parts: list[str] = []
    start_index = 0

    for split_index in split_indexes:
        left_piece = text[start_index:split_index].strip(" ,")
        right_piece = text[split_index + 5 :].strip(" ,")
        if not left_piece or not right_piece:
            continue
        if not _can_split_at_boundary(left_piece, right_piece, subject_prefix=subject_prefix):
            continue

        candidate = _apply_subject_prefix(left_piece, subject_prefix=subject_prefix, carry_subject=start_index > 0)
        parts.append(candidate)
        start_index = split_index + 5

    if not parts:
        return [text]

    remainder = text[start_index:].strip(" ,")
    candidate = _apply_subject_prefix(remainder, subject_prefix=subject_prefix, carry_subject=start_index > 0)
    if not _looks_like_clause(candidate) or not _has_complete_predicate(candidate):
        return [text]
    return parts + [candidate]


def _maybe_split_on_comma_predicates(text: str, *, subject_prefix_override: str | None = None) -> list[str]:
    split_indexes = _find_delimiter_positions_outside_quotes(text, ",")
    if not split_indexes:
        return [text]

    subject_prefix = subject_prefix_override or _extract_subject_prefix(text)
    parts: list[str] = []
    start_index = 0

    for split_index in split_indexes:
        left_piece = text[start_index:split_index].strip(" ,")
        right_piece = text[split_index + 1 :].strip(" ,")
        if not left_piece or not right_piece:
            continue
        if not _can_split_at_boundary(left_piece, right_piece, subject_prefix=subject_prefix):
            continue

        candidate = _apply_subject_prefix(left_piece, subject_prefix=subject_prefix, carry_subject=start_index > 0)
        parts.append(candidate)
        start_index = split_index + 1

    if not parts:
        return [text]

    remainder = text[start_index:].strip(" ,")
    candidate = _apply_subject_prefix(remainder, subject_prefix=subject_prefix, carry_subject=start_index > 0)
    if not _looks_like_clause(candidate) or not _has_complete_predicate(candidate):
        return [text]
    return parts + [candidate]


def _maybe_split_quote_gloss(text: str) -> list[str]:
    match = DOUBLE_QUOTE_RE.search(text)
    if match is None:
        return [text]

    quoted_text = text[: match.end()].strip(" ,")
    trailing_text = text[match.end() :].strip(" ,")
    if not quoted_text or not trailing_text:
        return [text]
    if not any(pattern.search(normalize_for_match(trailing_text)) for pattern in GLOSS_INFERENCE_PATTERNS):
        return [text]
    return [quoted_text, trailing_text]


def _expand_conjunction_pieces(pieces: list[str]) -> list[str]:
    if not pieces:
        return []

    if _looks_like_shared_object_pattern(pieces):
        return []

    expanded: list[str] = []
    subject_prefix = _extract_subject_prefix(pieces[0])
    previous_subject_prefix = subject_prefix

    for index, piece in enumerate(pieces):
        normalized_piece = piece.strip(" ,")
        if not normalized_piece:
            return []

        candidate = normalized_piece
        if index > 0 and previous_subject_prefix and _starts_with_verb_phrase(normalized_piece):
            candidate = f"{previous_subject_prefix} {normalized_piece}"

        if not _looks_like_clause(candidate) or not _has_complete_predicate(candidate):
            return []

        expanded.append(candidate)
        previous_subject_prefix = _extract_subject_prefix(candidate) or previous_subject_prefix

    return expanded


def _can_split_at_boundary(left_piece: str, right_piece: str, *, subject_prefix: str | None) -> bool:
    left_candidate = _apply_subject_prefix(left_piece, subject_prefix=subject_prefix, carry_subject=False)
    if not _looks_like_clause(left_candidate) or not _has_complete_predicate(left_candidate):
        return False

    right_candidate = _apply_subject_prefix(right_piece, subject_prefix=subject_prefix, carry_subject=True)
    if not _can_start_clause(right_piece, subject_prefix=subject_prefix):
        return False
    if not _looks_like_clause(right_candidate) or not _has_complete_predicate(right_candidate):
        return False
    return True


def _apply_subject_prefix(text: str, *, subject_prefix: str | None, carry_subject: bool) -> str:
    stripped = _strip_leading_coordinator(text.strip(" ,"))
    if not stripped:
        return stripped
    if carry_subject and subject_prefix and _starts_with_verb_phrase(stripped):
        return f"{subject_prefix} {stripped}"
    return stripped


def _can_start_clause(text: str, *, subject_prefix: str | None) -> bool:
    stripped = _strip_leading_coordinator(text.strip(" ,"))
    structural_text = _strip_structural_lead_in(stripped)
    if structural_text.startswith(('"', "“")):
        return False
    words = RAW_WORD_RE.findall(structural_text)
    if not words:
        return False

    normalized_words = [normalize_for_match(word) for word in words]
    if _is_clause_initial_verb(normalized_words):
        return subject_prefix is not None

    if len(normalized_words) >= 2 and _is_subject_token(words[0]) and any(
        _is_clause_initial_verb([word]) or _is_verb_hint(word) for word in normalized_words[1:3]
    ):
        return True

    return False


def _looks_like_clause(text: str) -> bool:
    words = WORD_RE.findall(text.lower())
    if len(words) < 4:
        return _looks_like_short_clause(words)
    return any(_is_verb_hint(word) for word in words)


def _has_complete_predicate(text: str) -> bool:
    words = WORD_RE.findall(normalize_for_match(text))
    verb_index = _find_first_verb_index(words)
    if verb_index is None:
        return False

    tokens_after_verb = words[verb_index + 1 :]
    if tokens_after_verb:
        return True
    return words[verb_index] not in TRANSITIVE_VERB_HINTS


def _looks_like_shared_object_pattern(pieces: list[str]) -> bool:
    if len(pieces) != 2:
        return False

    left_tokens = WORD_RE.findall(normalize_for_match(pieces[0]))
    right_tokens = WORD_RE.findall(normalize_for_match(pieces[1]))
    left_verb_index = _find_first_verb_index(left_tokens)
    right_verb_index = _find_first_verb_index(right_tokens)
    if left_verb_index is None or right_verb_index is None:
        return False

    left_verb = left_tokens[left_verb_index]
    if left_verb not in TRANSITIVE_VERB_HINTS:
        return False
    if len(left_tokens[left_verb_index + 1 :]) > 0:
        return False
    if len(right_tokens[right_verb_index + 1 :]) == 0:
        return False
    return True


def _extract_subject_prefix(text: str) -> str | None:
    matches = list(RAW_WORD_RE.finditer(text))
    normalized_words = [normalize_for_match(match.group()) for match in matches]
    verb_index = _find_first_verb_index(normalized_words)
    if verb_index is None or verb_index == 0:
        return None

    subject_words = [match.group() for match in matches[:verb_index]]
    if len(subject_words) > 4:
        return None
    if subject_words[0].lower() not in SUBJECT_PRONOUNS and not subject_words[0][0].isupper():
        return None
    return " ".join(subject_words)


def _extract_structured_subject_prefix(text: str) -> str | None:
    stripped_reporting = _strip_reporting_prefixes(text)
    if stripped_reporting != text:
        return _extract_subject_prefix(stripped_reporting) or _extract_subject_prefix(text)
    return _extract_subject_prefix(text)


def _extract_outer_subject_prefix(text: str) -> str | None:
    pre_quote = text.split('"', 1)[0].split("“", 1)[0].strip(" ,")
    if not pre_quote:
        return _extract_subject_prefix(text)
    return _extract_subject_prefix(pre_quote) or _extract_subject_prefix(text)


def _is_subject_token(word: str) -> bool:
    return word.lower() in SUBJECT_PRONOUNS or word[0].isupper()


def _starts_with_verb_phrase(text: str) -> bool:
    words = WORD_RE.findall(normalize_for_match(_strip_structural_lead_in(text)))
    if not words:
        return False
    return _is_clause_initial_verb(words)


def _find_first_verb_index(words: list[str]) -> int | None:
    for index, word in enumerate(words):
        if _is_verb_hint(word):
            return index
    return None


def _is_verb_hint(word: str) -> bool:
    return word in VERB_HINTS or word.endswith("ed")


def _is_clause_initial_verb(words: list[str]) -> bool:
    if not words:
        return False
    first_word = words[0]
    if first_word in AMBIGUOUS_PARTICIPLE_HINTS:
        return False
    if first_word in VERB_HINTS:
        return True
    return first_word.endswith("ed")


def _looks_like_short_clause(words: list[str]) -> bool:
    if len(words) < 2:
        return False
    return not _is_verb_hint(words[0]) and _is_verb_hint(words[1])


def extract_double_quoted_spans(text: str) -> list[str]:
    """Return double-quoted spans without treating apostrophes as quotations."""

    spans: list[str] = []
    for straight_quote, curly_quote in DOUBLE_QUOTE_RE.findall(text):
        span = straight_quote or curly_quote
        cleaned = span.strip()
        if cleaned:
            spans.append(cleaned)
    return spans


def _find_delimiter_positions_outside_quotes(text: str, delimiter: str) -> list[int]:
    lower_text = text.lower()
    positions: list[int] = []
    index = 0
    in_quote = False
    delimiter_length = len(delimiter)
    normalized_delimiter = delimiter.lower()

    while index < len(text):
        char = text[index]
        if char in {'"', "“", "”"}:
            in_quote = not in_quote
            index += 1
            continue

        if not in_quote and lower_text[index : index + delimiter_length] == normalized_delimiter:
            positions.append(index)
            index += delimiter_length
            continue

        index += 1

    return positions


def _strip_leading_coordinator(text: str) -> str:
    return LEADING_COORDINATOR_RE.sub("", text).strip()


def _strip_structural_lead_in(text: str) -> str:
    stripped = text.strip(" ,")
    if not stripped:
        return stripped

    while True:
        adverbial_match = LEADING_ADVERBIAL_RE.match(stripped)
        if adverbial_match is not None:
            stripped = stripped[adverbial_match.end() :].strip(" ,")
            continue

        temporal_match = LEADING_TEMPORAL_PHRASE_RE.match(stripped)
        if temporal_match is not None:
            stripped = stripped[temporal_match.end() :].strip(" ,")
            continue

        break

    return stripped


def _strip_nonpropositional_label_preamble(text: str) -> str:
    sentence_end_index = _find_first_sentence_boundary(text)
    if sentence_end_index is None:
        return text.strip()

    lead = text[:sentence_end_index].strip(" ,")
    remainder = text[sentence_end_index + 1 :].strip()
    if not remainder:
        return text.strip()
    if _is_nonpropositional_label(lead):
        return remainder
    return text.strip()


def _has_nonpropositional_label_preamble(text: str) -> bool:
    if _find_first_sentence_boundary(text) is None:
        return False
    lead = text[: _find_first_sentence_boundary(text)].strip(" ,")
    return _is_nonpropositional_label(lead)


def _is_nonpropositional_label(text: str) -> bool:
    if not LABEL_PREAMBLE_RE.match(text):
        return False
    return not _has_complete_predicate(text)


def _has_complex_quote_gloss_follow_on(text: str) -> bool:
    quote_gloss_parts = _maybe_split_quote_gloss(text.strip())
    if len(quote_gloss_parts) != 2:
        return False
    default_subject_prefix = _extract_outer_subject_prefix(quote_gloss_parts[0])
    return _has_follow_on_fact_clause(quote_gloss_parts[1], default_subject_prefix=default_subject_prefix)


def _has_follow_on_fact_clause(text: str, *, default_subject_prefix: str | None = None) -> bool:
    split_parts = _structured_split_piece(text, default_subject_prefix=default_subject_prefix)
    return len(split_parts) > 1


def _has_nested_reporting_prefix(text: str) -> bool:
    stripped = _strip_reporting_prefixes(text)
    return stripped != text.strip(" ,")


def _strip_reporting_prefixes(text: str) -> str:
    stripped = text.strip(" ,")
    while True:
        match = REPORTING_PREFIX_RE.match(stripped)
        if match is None:
            return stripped
        remainder = stripped[match.end() :].strip(" ,")
        if not remainder or remainder == stripped:
            return stripped
        stripped = remainder


def _has_dense_predicate_chain(text: str) -> bool:
    words = WORD_RE.findall(normalize_for_match(text))
    verb_count = sum(1 for word in words if _is_verb_hint(word))
    return verb_count >= 4


def _find_first_sentence_boundary(text: str) -> int | None:
    in_quote = False
    for index, char in enumerate(text):
        if char in {'"', "“", "”"}:
            in_quote = not in_quote
            continue
        if not in_quote and char in {".", "?", "!"}:
            return index
    return None


class ClaimExtractionService:
    """Coordinates assertion loading and persisted claim extraction."""

    def __init__(
        self,
        session: Session,
        *,
        extraction_mode: ExtractionMode = DEFAULT_EXTRACTION_MODE,
    ) -> None:
        self.session = session
        self.extraction_mode = extraction_mode
        self.assertions = AssertionRepository(session)
        self.claims = ClaimsRepository(session)

    def extract_from_assertion(self, assertion_id: UUID):
        assertion = self.assertions.get(assertion_id)
        if assertion is None:
            raise NotFoundError("Assertion not found.")

        existing_claims = self.claims.list_by_assertion(assertion_id)
        if existing_claims:
            return existing_claims

        plan = extract_claim_candidates_with_strategy(assertion.raw_text, mode=self.extraction_mode)
        claims = self.claims.create_many(assertion_id, plan.claims)
        snapshot = build_extraction_snapshot(
            assertion_id=assertion.id,
            source_assertion_text=assertion.raw_text,
            normalized_assertion_text=assertion.normalized_text,
            extraction_strategy=plan.strategy,
            extraction_version=CURRENT_EXTRACTION_VERSION,
            claims=claims,
        )
        self.assertions.record_extraction(
            assertion,
            strategy=plan.strategy,
            version=CURRENT_EXTRACTION_VERSION,
            snapshot=snapshot,
        )
        self.session.flush()
        return claims
