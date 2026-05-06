"""Narrow MVP workflow for verifying a statement against an uploaded case PDF."""

from dataclasses import dataclass
from io import BytesIO
import re
import zlib

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.core.enums import SupportStatus
from app.core.exceptions import ValidationError
from app.services.parsing.citation_extraction import CitationExtractionService
from app.services.parsing.case_citations import (
    ParsedCaseCitation,
    parse_case_citation,
)
from app.services.parsing.normalization import normalize_for_match, normalize_text

STREAM_RE = re.compile(rb"<<(?P<dictionary>.*?)>>\s*stream\r?\n(?P<stream>.*?)\r?\nendstream", re.S)
TEXT_SECTION_RE = re.compile(r"BT(.*?)ET", re.S)
TJ_STRING_RE = re.compile(r"\((?P<text>(?:\\.|[^\\)])*)\)\s*Tj")
TJ_ARRAY_RE = re.compile(r"\[(?P<items>.*?)\]\s*TJ", re.S)
PDF_STRING_RE = re.compile(r"\((?P<text>(?:\\.|[^\\)])*)\)")
NON_ALPHANUMERIC_RE = re.compile(r"[^a-z0-9]+")
TOKEN_RE = re.compile(r"[a-z0-9']+")
PDF_LINE_WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")
STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "for",
    "that",
    "this",
    "these",
    "those",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "it",
    "its",
    "with",
    "from",
    "by",
    "as",
    "at",
    "into",
    "case",
    "court",
}
ABSOLUTE_QUALIFIERS = {"always", "never", "all", "only", "must", "every", "none"}


@dataclass(slots=True)
class ExtractedPdfAuthorityMetadata:
    case_name: str | None
    reporter_volume: str | None
    reporter_abbreviation: str | None
    first_page: str | None
    court: str | None
    year: int | None
    canonical_citation: str | None


@dataclass(slots=True)
class CasePdfVerificationResult:
    pdf_text_status: str
    extracted_authority_metadata: ExtractedPdfAuthorityMetadata | None
    extracted_character_count: int
    page_count: int | None
    extraction_warnings: list[str]
    extracted_text_preview: str | None
    citation_match_status: str
    statement_verdict: SupportStatus
    reasoning: str | None
    support_snippet: str | None
    confidence_score: float | None
    suggested_fix: str | None


class CasePdfVerificationService:
    """Verify a single statement and optional citation against one uploaded case PDF."""

    def verify_pdf(
        self,
        *,
        filename: str | None,
        pdf_bytes: bytes,
        statement_text: str,
        citation_text: str | None,
    ) -> CasePdfVerificationResult:
        if filename is not None and filename.lower().endswith(".pdf") is False:
            raise ValidationError("Uploaded file must be a PDF.")

        extracted_text = extract_pdf_text(pdf_bytes)
        metadata = _extract_authority_metadata(extracted_text.text)
        citation_match_status = _derive_citation_match_status(citation_text, metadata)

        if extracted_text.status != "text_extracted":
            reasoning = {
                "invalid_pdf": "Uploaded file is not a readable PDF.",
                "text_empty": "PDF was readable, but no authority text could be extracted.",
                "scanned_or_unreadable": "PDF appears scanned or otherwise unreadable without OCR.",
                "unsupported_pdf_layout": "PDF layout or encryption is not supported by the MVP extractor.",
            }.get(extracted_text.status, "PDF text could not be extracted.")
            return CasePdfVerificationResult(
                pdf_text_status=extracted_text.status,
                extracted_authority_metadata=metadata,
                extracted_character_count=extracted_text.character_count,
                page_count=extracted_text.page_count,
                extraction_warnings=extracted_text.warnings,
                extracted_text_preview=extracted_text.preview,
                citation_match_status=citation_match_status,
                statement_verdict=SupportStatus.UNVERIFIED,
                reasoning=reasoning,
                support_snippet=None,
                confidence_score=0.2 if extracted_text.status == "invalid_pdf" else 0.3,
                suggested_fix="Upload a text-based opinion PDF or add OCR support before verification.",
            )

        statement_evaluation = _evaluate_statement_against_pdf_text(statement_text, extracted_text.text)
        return CasePdfVerificationResult(
            pdf_text_status=extracted_text.status,
            extracted_authority_metadata=metadata,
            extracted_character_count=extracted_text.character_count,
            page_count=extracted_text.page_count,
            extraction_warnings=extracted_text.warnings,
            extracted_text_preview=extracted_text.preview,
            citation_match_status=citation_match_status,
            statement_verdict=statement_evaluation.statement_verdict,
            reasoning=statement_evaluation.reasoning,
            support_snippet=statement_evaluation.support_snippet,
            confidence_score=statement_evaluation.confidence_score,
            suggested_fix=statement_evaluation.suggested_fix,
        )


@dataclass(slots=True)
class ExtractedPdfText:
    status: str
    text: str
    character_count: int = 0
    page_count: int | None = None
    warnings: list[str] | None = None

    @property
    def preview(self) -> str | None:
        if not self.text:
            return None
        return self.text[:600]


@dataclass(slots=True)
class StatementEvaluation:
    statement_verdict: SupportStatus
    reasoning: str
    support_snippet: str | None
    confidence_score: float
    suggested_fix: str | None


def extract_pdf_text(pdf_bytes: bytes) -> ExtractedPdfText:
    if not pdf_bytes.startswith(b"%PDF"):
        return _build_extracted_pdf_text(
            status="invalid_pdf",
            text="",
            warnings=["Uploaded bytes do not start with a PDF header."],
        )

    pypdf_result = _extract_pdf_text_with_pypdf(pdf_bytes)
    if pypdf_result.status == "text_extracted":
        return pypdf_result
    if pypdf_result.status in {"invalid_pdf", "unsupported_pdf_layout"}:
        return pypdf_result

    legacy_result = _extract_pdf_text_with_legacy_stream_parser(pdf_bytes)
    warnings = [*(pypdf_result.warnings or [])]
    if legacy_result.status == "text_extracted":
        warnings.append("pypdf returned no extractable text; legacy PDF stream fallback extracted text.")
        return _build_extracted_pdf_text(
            status="text_extracted",
            text=legacy_result.text,
            page_count=pypdf_result.page_count,
            warnings=warnings,
        )
    if legacy_result.status == "text_empty":
        return _build_extracted_pdf_text(
            status="text_empty",
            text="",
            page_count=pypdf_result.page_count,
            warnings=warnings + ["PDF text operators were present, but no readable text was extracted."],
        )
    if legacy_result.status == "scanned_or_unreadable":
        return _build_extracted_pdf_text(
            status="scanned_or_unreadable",
            text="",
            page_count=pypdf_result.page_count,
            warnings=warnings + ["No extractable text was found. The PDF may be scanned or image-only."],
        )
    return pypdf_result


def _extract_pdf_text_with_pypdf(pdf_bytes: bytes) -> ExtractedPdfText:
    warnings: list[str] = []
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
    except PdfReadError as exc:
        return _build_extracted_pdf_text(
            status="invalid_pdf",
            text="",
            warnings=[f"pypdf could not read the PDF: {exc}"],
        )
    except Exception as exc:
        return _build_extracted_pdf_text(
            status="invalid_pdf",
            text="",
            warnings=[f"PDF reader failed before page extraction: {type(exc).__name__}"],
        )

    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception:
            return _build_extracted_pdf_text(
                status="unsupported_pdf_layout",
                text="",
                page_count=len(reader.pages),
                warnings=["PDF is encrypted and could not be decrypted with an empty password."],
            )

    page_texts: list[str] = []
    page_count = len(reader.pages)
    for index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            warnings.append(f"Page {index} text extraction failed: {type(exc).__name__}.")
            continue
        if text.strip():
            page_texts.append(text)

    normalized_text = _normalize_extracted_pdf_text("\n".join(page_texts))
    if normalized_text:
        return _build_extracted_pdf_text(
            status="text_extracted",
            text=normalized_text,
            page_count=page_count,
            warnings=warnings,
        )

    warnings.append("pypdf read the PDF but found no extractable page text.")
    return _build_extracted_pdf_text(
        status="text_empty",
        text="",
        page_count=page_count,
        warnings=warnings,
    )


def _extract_pdf_text_with_legacy_stream_parser(pdf_bytes: bytes) -> ExtractedPdfText:
    warnings: list[str] = ["Using legacy PDF stream parser fallback."]

    extracted_blocks: list[str] = []
    saw_text_operator = False
    saw_image_marker = b"/Subtype /Image" in pdf_bytes or b"/Image" in pdf_bytes

    for match in STREAM_RE.finditer(pdf_bytes):
        stream_bytes = match.group("stream")
        dictionary_bytes = match.group("dictionary")
        decoded_stream = _decode_pdf_stream(stream_bytes, dictionary_bytes)
        if decoded_stream is None:
            continue

        stream_text = decoded_stream.decode("latin-1", errors="ignore")
        sections = TEXT_SECTION_RE.findall(stream_text)
        if sections:
            saw_text_operator = True
        for section in sections:
            extracted = _extract_pdf_text_section(section)
            if extracted:
                extracted_blocks.append(extracted)

    normalized_text = _normalize_extracted_pdf_text("\n".join(block for block in extracted_blocks if block))
    if normalized_text:
        return _build_extracted_pdf_text(status="text_extracted", text=normalized_text, warnings=warnings)
    if saw_image_marker:
        return _build_extracted_pdf_text(status="scanned_or_unreadable", text="", warnings=warnings)
    if saw_text_operator:
        return _build_extracted_pdf_text(status="text_empty", text="", warnings=warnings)
    return _build_extracted_pdf_text(status="scanned_or_unreadable", text="", warnings=warnings)


def _build_extracted_pdf_text(
    *,
    status: str,
    text: str,
    page_count: int | None = None,
    warnings: list[str] | None = None,
) -> ExtractedPdfText:
    normalized_text = _normalize_extracted_pdf_text(text)
    return ExtractedPdfText(
        status=status,
        text=normalized_text,
        character_count=len(normalized_text),
        page_count=page_count,
        warnings=warnings or [],
    )


def _normalize_extracted_pdf_text(text: str) -> str:
    lines = [
        PDF_LINE_WHITESPACE_RE.sub(" ", line).strip()
        for line in text.splitlines()
    ]
    return "\n".join(line for line in lines if line)


def _decode_pdf_stream(stream_bytes: bytes, dictionary_bytes: bytes) -> bytes | None:
    if b"/FlateDecode" in dictionary_bytes:
        try:
            return zlib.decompress(stream_bytes)
        except zlib.error:
            return None
    return stream_bytes


def _extract_pdf_text_section(section: str) -> str:
    fragments: list[str] = []
    for match in TJ_STRING_RE.finditer(section):
        text = _decode_pdf_string(match.group("text"))
        if text:
            fragments.append(text)
    for match in TJ_ARRAY_RE.finditer(section):
        for string_match in PDF_STRING_RE.finditer(match.group("items")):
            text = _decode_pdf_string(string_match.group("text"))
            if text:
                fragments.append(text)
    return normalize_text(" ".join(fragments))


def _decode_pdf_string(value: str) -> str:
    replacements = {
        r"\(": "(",
        r"\)": ")",
        r"\\": "\\",
        r"\n": "\n",
        r"\r": "\r",
        r"\t": "\t",
    }
    decoded = value
    for needle, replacement in replacements.items():
        decoded = decoded.replace(needle, replacement)
    return normalize_text(decoded)


def _extract_authority_metadata(text: str) -> ExtractedPdfAuthorityMetadata | None:
    for candidate in CitationExtractionService().extract_full_case_citations(text[:4000]):
        parsed = parse_case_citation(candidate.citation_text)
        if not parsed.is_case_citation:
            continue
        return ExtractedPdfAuthorityMetadata(
            case_name=parsed.case_name,
            reporter_volume=parsed.reporter_volume,
            reporter_abbreviation=parsed.reporter_abbreviation,
            first_page=parsed.first_page,
            court=parsed.court,
            year=parsed.year,
            canonical_citation=_build_canonical_citation(parsed),
        )
    return None


def _build_canonical_citation(parsed: ParsedCaseCitation) -> str | None:
    if parsed.case_name is None:
        return None

    citation = parsed.case_name
    if parsed.reporter_volume and parsed.reporter_abbreviation and parsed.first_page:
        citation = f"{citation}, {parsed.reporter_volume} {parsed.reporter_abbreviation} {parsed.first_page}"
    if parsed.year is None and parsed.court is None:
        return citation
    if parsed.year is None:
        return f"{citation} ({parsed.court})"
    if parsed.court is None:
        return f"{citation} ({parsed.year})"
    return f"{citation} ({parsed.court} {parsed.year})"


def _derive_citation_match_status(
    citation_text: str | None,
    metadata: ExtractedPdfAuthorityMetadata | None,
) -> str:
    if citation_text is None or not citation_text.strip():
        return "citation_not_provided"

    candidate = CitationExtractionService().parse_full_case_citation(citation_text)
    if candidate is None:
        return "citation_unresolved"

    parsed = parse_case_citation(candidate.citation_text)
    if not parsed.is_case_citation:
        return "citation_unresolved"
    if metadata is None or metadata.case_name is None:
        return "citation_recognized"

    if _citation_conflicts_with_metadata(parsed, metadata):
        return "citation_mismatch"
    if _citation_matches_metadata(parsed, metadata):
        return "citation_matches_pdf"
    return "citation_unresolved"


def _citation_conflicts_with_metadata(
    parsed: ParsedCaseCitation,
    metadata: ExtractedPdfAuthorityMetadata,
) -> bool:
    if parsed.case_name and metadata.case_name and _normalize_name(parsed.case_name) != _normalize_name(metadata.case_name):
        return True
    if parsed.reporter_volume and metadata.reporter_volume and parsed.reporter_volume != metadata.reporter_volume:
        return True
    if (
        parsed.reporter_abbreviation
        and metadata.reporter_abbreviation
        and _normalize_reporter(parsed.reporter_abbreviation) != _normalize_reporter(metadata.reporter_abbreviation)
    ):
        return True
    if parsed.first_page and metadata.first_page and parsed.first_page != metadata.first_page:
        return True
    if parsed.year is not None and metadata.year is not None and parsed.year != metadata.year:
        return True
    return False


def _citation_matches_metadata(
    parsed: ParsedCaseCitation,
    metadata: ExtractedPdfAuthorityMetadata,
) -> bool:
    if parsed.case_name and metadata.case_name and _normalize_name(parsed.case_name) == _normalize_name(metadata.case_name):
        return True
    if (
        parsed.reporter_volume
        and parsed.reporter_abbreviation
        and parsed.first_page
        and metadata.reporter_volume
        and metadata.reporter_abbreviation
        and metadata.first_page
        and parsed.reporter_volume == metadata.reporter_volume
        and _normalize_reporter(parsed.reporter_abbreviation) == _normalize_reporter(metadata.reporter_abbreviation)
        and parsed.first_page == metadata.first_page
    ):
        return True
    return False


def _evaluate_statement_against_pdf_text(statement_text: str, pdf_text: str) -> StatementEvaluation:
    snippets = _build_text_snippets(pdf_text)
    statement_tokens = _content_tokens(statement_text)
    best_snippet = None
    best_overlap = 0.0

    for snippet in snippets:
        overlap = _lexical_overlap(statement_tokens, _content_tokens(snippet))
        if overlap > best_overlap:
            best_overlap = overlap
            best_snippet = snippet

    best_tokens = _content_tokens(best_snippet or "")
    qualifier_mismatch = _has_qualifier_mismatch(statement_tokens, best_tokens)

    if best_snippet and best_overlap >= 0.58 and not qualifier_mismatch:
        return StatementEvaluation(
            statement_verdict=SupportStatus.SUPPORTED,
            reasoning="Extracted PDF text materially overlaps the provided statement.",
            support_snippet=best_snippet,
            confidence_score=0.84,
            suggested_fix=None,
        )
    if best_snippet and best_overlap >= 0.42 and not qualifier_mismatch:
        return StatementEvaluation(
            statement_verdict=SupportStatus.PARTIALLY_SUPPORTED,
            reasoning="Extracted PDF text supports part of the statement, but not the full phrasing.",
            support_snippet=best_snippet,
            confidence_score=0.68,
            suggested_fix="Narrow the statement to the portion stated more directly in the opinion text.",
        )
    if best_snippet and best_overlap >= 0.3 and qualifier_mismatch:
        return StatementEvaluation(
            statement_verdict=SupportStatus.OVERSTATED,
            reasoning="The statement uses broader qualifier language than the matched PDF excerpt appears to support.",
            support_snippet=best_snippet,
            confidence_score=0.66,
            suggested_fix="Soften the qualifier or quote the narrower language from the opinion.",
        )
    if best_snippet and best_overlap >= 0.22:
        return StatementEvaluation(
            statement_verdict=SupportStatus.AMBIGUOUS,
            reasoning="Extracted PDF text is related to the statement, but support is not clear enough to verify deterministically.",
            support_snippet=best_snippet,
            confidence_score=0.52,
            suggested_fix="Quote the narrower holding language or verify the proposition manually.",
        )
    return StatementEvaluation(
        statement_verdict=SupportStatus.UNSUPPORTED,
        reasoning="No extracted PDF snippet clearly supports the provided statement.",
        support_snippet=best_snippet,
        confidence_score=0.74,
        suggested_fix="Revise the statement to track the opinion text more closely or verify it manually.",
    )


def _build_text_snippets(pdf_text: str) -> list[str]:
    lines = [normalize_text(line) for line in pdf_text.splitlines() if normalize_text(line)]
    if lines:
        snippets = list(lines)
    else:
        snippets = [normalize_text(part) for part in re.split(r"(?<=[.?!])\s+", pdf_text) if normalize_text(part)]

    windows: list[str] = list(snippets)
    for index in range(len(snippets) - 1):
        windows.append(normalize_text(f"{snippets[index]} {snippets[index + 1]}"))
    return windows


def _content_tokens(text: str) -> set[str]:
    normalized = normalize_for_match(text)
    return {
        token
        for token in TOKEN_RE.findall(normalized)
        if token not in STOPWORDS and len(token) > 2
    }


def _lexical_overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left)


def _has_qualifier_mismatch(statement_tokens: set[str], snippet_tokens: set[str]) -> bool:
    qualifiers = statement_tokens & ABSOLUTE_QUALIFIERS
    if not qualifiers:
        return False
    return not qualifiers.issubset(snippet_tokens)


def _normalize_name(value: str) -> str:
    return NON_ALPHANUMERIC_RE.sub(" ", normalize_for_match(value)).strip()


def _normalize_reporter(value: str) -> str:
    return NON_ALPHANUMERIC_RE.sub("", normalize_for_match(value))
