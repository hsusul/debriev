"""Structured claim splitting helpers."""

from dataclasses import dataclass
import re

from app.services.parsing.normalization import normalize_for_match

DOUBLE_QUOTE_RE = re.compile(r'"([^"\n]+)"|“([^”\n]+)”')
LABEL_PREAMBLE_RE = re.compile(r"^\s*[A-Z][A-Za-z0-9 /&()#._-]{1,24}:\s*[^.?!\n]{0,80}\s*$")
REPORTING_PREFIX_RE = re.compile(
    r"^(?:(?:[A-Z][A-Za-z']+(?:\s+[A-Z][A-Za-z']+){0,2})|"
    r"(?:the|this)\s+(?:email|exhibit|letter|memo|message|document|record))\s+"
    r"(?:states?|stated|shows?|showed|reflects?|reflected|indicates?|indicated|"
    r"notes?|noted|says?|said|wrote|argued|contended|asserted|maintained)\s+(?:that\s+)?",
    re.IGNORECASE,
)
LEADING_COORDINATOR_RE = re.compile(r"^(?:and|but)\s+", re.IGNORECASE)
LEADING_ADVERBIAL_RE = re.compile(
    r"^(?:later|then|thereafter|subsequently|promptly|immediately|also)\b[\s,]*",
    re.IGNORECASE,
)
LEADING_TEMPORAL_PHRASE_RE = re.compile(
    r"^(?:after|before|during|following|upon)\b[^,]{0,80},\s*",
    re.IGNORECASE,
)
LEADING_CONTEXTUAL_PHRASE_RE = re.compile(
    r"^(?:in|on|at|by|from|within)\b[^,]{0,80},\s*",
    re.IGNORECASE,
)
INFERENCE_OPENERS = (
    re.compile(r"^\s*(?:which|suggesting|showing|meaning|therefore|thus|so)\b", re.IGNORECASE),
)
MIXED_INFERENCE_RE = re.compile(r",\s*(?:which|suggesting|showing|meaning|therefore|thus|so)\b", re.IGNORECASE)
WORD_RE = re.compile(r"[a-z0-9']+")
RAW_WORD_RE = re.compile(r"\b[\w']+\b")
SUBJECT_PRONOUNS = {"he", "she", "they", "it", "we", "i", "you", "which", "who", "that"}
ADVERBIAL_TOKENS = {"later", "then", "thereafter", "subsequently", "promptly", "immediately", "also"}
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
    "forwarded",
    "moved",
    "compelled",
    "reflects",
    "indicates",
    "notes",
    "suggests",
    "shows",
}
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
    "forwarded",
    "moved",
    "compelled",
}
AMBIGUOUS_PARTICIPLE_HINTS = {"related", "alleged", "revised", "attached", "written", "pending", "underlying"}


@dataclass(slots=True)
class ClauseBoundary:
    index: int
    token: str
    kind: str


def extract_structured_parts(text: str) -> list[str]:
    """Return ordered structured claim parts for harder assertions."""

    working_text = _strip_nonpropositional_label_preamble(text.strip())
    if not working_text:
        return []
    if not DOUBLE_QUOTE_RE.search(working_text) and _should_keep_mixed_statement_intact(working_text):
        return [working_text.strip(" ,")]

    parts = _extract_structured_parts(
        working_text,
        default_subject_prefix=_extract_structured_subject_prefix(working_text),
    )
    cleaned = [part.strip(" ,") for part in parts if part.strip(" ,")]
    return cleaned or [working_text.strip(" ,")]


def _extract_structured_parts(text: str, *, default_subject_prefix: str | None) -> list[str]:
    stripped = text.strip(" ,")
    if not stripped:
        return []

    quote_parts = _split_quote_and_trailing_parts(stripped, default_subject_prefix=default_subject_prefix)
    if len(quote_parts) > 1:
        parts: list[str] = []
        quote_subject_prefix = _extract_outer_subject_prefix(quote_parts[0]) or default_subject_prefix
        for index, part in enumerate(quote_parts):
            cleaned = part.strip(" ,")
            if not cleaned:
                continue
            if index == 0:
                parts.append(cleaned)
                continue
            parts.extend(
                _split_clause_chain(
                    cleaned,
                    default_subject_prefix=quote_subject_prefix,
                )
            )
        return parts

    return _split_clause_chain(stripped, default_subject_prefix=default_subject_prefix)


def _split_quote_and_trailing_parts(text: str, *, default_subject_prefix: str | None) -> list[str]:
    match = DOUBLE_QUOTE_RE.search(text)
    if match is None:
        return [text]

    quoted_piece = text[: match.end()].strip(" ,")
    trailing_text = text[match.end() :].strip(" ,")
    if not quoted_piece or not trailing_text:
        return [text]

    outer_subject_prefix = default_subject_prefix or _extract_outer_subject_prefix(quoted_piece)
    if _starts_with_inferential_gloss(trailing_text):
        return [quoted_piece, trailing_text]
    if _can_start_clause(trailing_text, default_subject_prefix=outer_subject_prefix):
        return [quoted_piece, trailing_text]
    return [text]


def _split_clause_chain(text: str, *, default_subject_prefix: str | None) -> list[str]:
    boundaries = _collect_clause_boundaries(text)
    if not boundaries:
        candidate = _normalize_piece(
            text.strip(" ,"),
            default_subject_prefix=default_subject_prefix,
            carry_subject=True,
        )
        return [candidate] if candidate else []

    parts: list[str] = []
    start_index = 0
    subject_prefix = default_subject_prefix or _extract_structured_subject_prefix(text)

    for boundary in boundaries:
        left_piece = text[start_index:boundary.index].strip(" ,")
        right_piece = text[boundary.index + len(boundary.token) :].strip(" ,")
        if not left_piece or not right_piece:
            continue
        if not _can_split_at_boundary(left_piece, right_piece, default_subject_prefix=subject_prefix):
            continue

        left_candidate = _normalize_piece(
            left_piece,
            default_subject_prefix=subject_prefix,
            carry_subject=start_index > 0,
        )
        parts.append(left_candidate)
        if not _starts_with_inferential_gloss(left_candidate):
            subject_prefix = _extract_structured_subject_prefix(left_candidate) or subject_prefix
        start_index = boundary.index + len(boundary.token)

    if not parts:
        return [text.strip(" ,")]

    remainder = text[start_index:].strip(" ,")
    remainder_candidate = _normalize_piece(
        remainder,
        default_subject_prefix=subject_prefix,
        carry_subject=start_index > 0,
    )
    if not _looks_like_complete_clause(remainder_candidate):
        return [text.strip(" ,")]
    return parts + [remainder_candidate]


def _can_split_at_boundary(left_piece: str, right_piece: str, *, default_subject_prefix: str | None) -> bool:
    left_candidate = _normalize_piece(
        left_piece,
        default_subject_prefix=default_subject_prefix,
        carry_subject=_needs_subject_prefix(left_piece),
    )
    if not _looks_like_complete_clause(left_candidate):
        return False

    if _starts_with_inferential_gloss(right_piece):
        return True

    if not _can_start_clause(right_piece, default_subject_prefix=default_subject_prefix):
        return False

    right_candidate = _normalize_piece(right_piece, default_subject_prefix=default_subject_prefix, carry_subject=True)
    if not _looks_like_complete_clause(right_candidate):
        return False

    return True


def _normalize_piece(text: str, *, default_subject_prefix: str | None, carry_subject: bool) -> str:
    stripped = _strip_leading_coordinator(text.strip(" ,"))
    if not stripped:
        return stripped
    if carry_subject and default_subject_prefix and _needs_subject_prefix(stripped):
        return f"{default_subject_prefix} {stripped}"
    return stripped


def _needs_subject_prefix(text: str) -> bool:
    if _starts_with_inferential_gloss(text):
        return False
    structural = _strip_structural_lead_in(text)
    words = [normalize_for_match(word) for word in RAW_WORD_RE.findall(structural)]
    if not words:
        return False
    if _starts_with_explicit_subject(structural):
        return False
    return _is_clause_initial_verb(words)


def _can_start_clause(text: str, *, default_subject_prefix: str | None) -> bool:
    stripped = _strip_leading_coordinator(text.strip(" ,"))
    if not stripped or stripped.startswith(('"', "“")):
        return False
    if _starts_with_inferential_gloss(stripped):
        return True

    structural = _strip_structural_lead_in(stripped)
    if not structural:
        return False
    if _starts_with_explicit_subject(structural):
        return True

    words = [normalize_for_match(word) for word in RAW_WORD_RE.findall(structural)]
    if not words:
        return False
    return default_subject_prefix is not None and _is_clause_initial_verb(words)


def _looks_like_complete_clause(text: str) -> bool:
    if _starts_with_inferential_gloss(text):
        words = WORD_RE.findall(normalize_for_match(text))
        return len(words) >= 2 and any(_is_verb_hint(word) for word in words)

    words = WORD_RE.findall(normalize_for_match(_strip_structural_lead_in(text)))
    if len(words) < 2:
        return False
    verb_index = _find_first_verb_index(words)
    if verb_index is None:
        return False
    tokens_after_verb = words[verb_index + 1 :]
    if tokens_after_verb:
        return True
    return words[verb_index] not in TRANSITIVE_VERB_HINTS


def _starts_with_explicit_subject(text: str) -> bool:
    stripped = _strip_structural_lead_in(text)
    words = RAW_WORD_RE.findall(stripped)
    normalized_words = [normalize_for_match(word) for word in words]
    if len(words) < 2:
        return False
    if not _is_subject_token(words[0]):
        return False
    return any(_is_clause_initial_verb([word]) or _is_verb_hint(word) for word in normalized_words[1:4])


def _extract_subject_prefix(text: str) -> str | None:
    stripped = _strip_structural_lead_in(text)
    matches = list(RAW_WORD_RE.finditer(stripped))
    normalized_words = [normalize_for_match(match.group()) for match in matches]
    verb_index = _find_first_verb_index(normalized_words)
    if verb_index is None or verb_index == 0:
        return None

    subject_words = [match.group() for match in matches[:verb_index]]
    subject_normalized = normalized_words[:verb_index]
    if subject_normalized:
        first_adverbial_index = next(
            (index for index, word in enumerate(subject_normalized[1:], start=1) if word in ADVERBIAL_TOKENS),
            None,
        )
        if first_adverbial_index is not None:
            subject_words = subject_words[:first_adverbial_index]
    if len(subject_words) > 6:
        return None
    if subject_words[0].lower() not in SUBJECT_PRONOUNS and not subject_words[0][0].isupper():
        return None
    return " ".join(subject_words)


def _extract_structured_subject_prefix(text: str) -> str | None:
    stripped_reporting = _strip_reporting_prefixes(text)
    return _extract_subject_prefix(stripped_reporting) or _extract_subject_prefix(text)


def _extract_outer_subject_prefix(text: str) -> str | None:
    pre_quote = text.split('"', 1)[0].split("“", 1)[0].strip(" ,")
    return _extract_subject_prefix(pre_quote) or _extract_subject_prefix(text)


def _collect_clause_boundaries(text: str) -> list[ClauseBoundary]:
    boundaries: list[ClauseBoundary] = []
    lower_text = text.lower()
    index = 0
    in_quote = False

    while index < len(text):
        char = text[index]
        if char in {'"', "“", "”"}:
            in_quote = not in_quote
            index += 1
            continue

        if in_quote:
            index += 1
            continue

        if char == ";":
            boundaries.append(ClauseBoundary(index=index, token=";", kind="semicolon"))
            index += 1
            continue

        if char == ",":
            boundaries.append(ClauseBoundary(index=index, token=",", kind="comma"))
            index += 1
            continue

        if lower_text[index : index + 5] == " and ":
            boundaries.append(ClauseBoundary(index=index, token=" and ", kind="coordinator"))
            index += 5
            continue

        index += 1

    return boundaries


def _starts_with_inferential_gloss(text: str) -> bool:
    return any(pattern.search(text) for pattern in INFERENCE_OPENERS)


def _should_keep_mixed_statement_intact(text: str) -> bool:
    if MIXED_INFERENCE_RE.search(text) is None:
        return False
    return " and later " not in text.lower() and " and then " not in text.lower()


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


def _strip_leading_coordinator(text: str) -> str:
    return LEADING_COORDINATOR_RE.sub("", text).strip()


def _strip_structural_lead_in(text: str) -> str:
    stripped = text.strip(" ,")
    if not stripped:
        return stripped

    while True:
        updated = stripped
        for pattern in (
            LEADING_ADVERBIAL_RE,
            LEADING_TEMPORAL_PHRASE_RE,
            LEADING_CONTEXTUAL_PHRASE_RE,
        ):
            match = pattern.match(updated)
            if match is not None:
                updated = updated[match.end() :].strip(" ,")
        if updated == stripped:
            return stripped
        stripped = updated


def _strip_nonpropositional_label_preamble(text: str) -> str:
    boundary = _find_first_sentence_boundary(text)
    if boundary is None:
        return text.strip()
    lead = text[:boundary].strip(" ,")
    remainder = text[boundary + 1 :].strip()
    if not remainder:
        return text.strip()
    if _is_nonpropositional_label(lead):
        return remainder
    return text.strip()


def _find_first_sentence_boundary(text: str) -> int | None:
    in_quote = False
    for index, char in enumerate(text):
        if char in {'"', "“", "”"}:
            in_quote = not in_quote
            continue
        if not in_quote and char in {".", "?", "!"}:
            return index
    return None


def _is_nonpropositional_label(text: str) -> bool:
    if not LABEL_PREAMBLE_RE.match(text):
        return False
    return not _looks_like_complete_clause(text)


def _is_subject_token(word: str) -> bool:
    return word.lower() in SUBJECT_PRONOUNS or word[0].isupper()


def _find_first_verb_index(words: list[str]) -> int | None:
    for index, word in enumerate(words):
        if _is_verb_hint(word):
            return index
    return None


def _is_clause_initial_verb(words: list[str]) -> bool:
    if not words:
        return False
    first_word = words[0]
    if first_word in AMBIGUOUS_PARTICIPLE_HINTS:
        return False
    if first_word in VERB_HINTS:
        return True
    return first_word.endswith("ed")


def _is_verb_hint(word: str) -> bool:
    return word in VERB_HINTS or word.endswith("ed")
