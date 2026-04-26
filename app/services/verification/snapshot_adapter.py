"""Version-aware read-side adapter for persisted verification support snapshots."""

from dataclasses import dataclass, field
from typing import Literal

from app.models import CURRENT_SUPPORT_SNAPSHOT_VERSION

SnapshotReadStatus = Literal[
    "missing",
    "versioned_v1",
    "legacy_unversioned_v1",
    "unsupported_version",
]

LEGACY_UNVERSIONED_SNAPSHOT_NOTE = (
    "Snapshot: legacy/unversioned persisted support snapshot interpreted as v1-compatible."
)


@dataclass(slots=True)
class ParsedSupportAssessment:
    segment_id: str | None
    anchor: str
    role: str
    contribution: str


@dataclass(slots=True)
class ParsedExcludedSupportLink:
    link_id: str | None
    claim_id: str | None
    segment_id: str | None
    code: str | None
    message: str | None


@dataclass(slots=True)
class ParsedVerificationSupportLink:
    link_id: str | None
    claim_id: str | None
    segment_id: str | None
    source_document_id: str | None
    sequence_order: int | None
    link_type: str | None
    citation_text: str | None
    user_confirmed: bool
    anchor: str | None
    evidence_role: str | None


@dataclass(slots=True)
class ParsedSupportItem:
    order: int | None
    segment_id: str | None
    source_document_id: str | None
    anchor: str | None
    evidence_role: str | None
    speaker: str | None
    segment_type: str | None
    raw_text: str | None
    normalized_text: str | None


@dataclass(slots=True)
class ParsedClaimScope:
    claim_id: str | None
    draft_id: str | None
    matter_id: str | None
    evidence_bundle_id: str | None
    scope_kind: str | None
    allowed_source_document_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ParsedProviderOutput:
    primary_anchor: str | None = None
    support_assessments: list[ParsedSupportAssessment] = field(default_factory=list)


@dataclass(slots=True)
class ParsedVerificationSupportSnapshot:
    status: SnapshotReadStatus
    version: int | None
    note: str | None = None
    claim_scope: ParsedClaimScope | None = None
    valid_support_links: list[ParsedVerificationSupportLink] = field(default_factory=list)
    excluded_support_links: list[ParsedExcludedSupportLink] = field(default_factory=list)
    support_items: list[ParsedSupportItem] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    provider_output: ParsedProviderOutput = field(default_factory=ParsedProviderOutput)


def parse_verification_support_snapshot(
    support_snapshot: dict[str, object] | None,
    support_snapshot_version: int | None,
) -> ParsedVerificationSupportSnapshot:
    """Normalize a persisted support snapshot into a stable typed read model."""

    if support_snapshot is None:
        return ParsedVerificationSupportSnapshot(
            status="missing",
            version=support_snapshot_version,
        )

    if support_snapshot_version is None:
        return _parse_v1_snapshot(
            support_snapshot,
            status="legacy_unversioned_v1",
            version=None,
            note=LEGACY_UNVERSIONED_SNAPSHOT_NOTE,
        )

    if support_snapshot_version == CURRENT_SUPPORT_SNAPSHOT_VERSION:
        return _parse_v1_snapshot(
            support_snapshot,
            status="versioned_v1",
            version=support_snapshot_version,
            note=None,
        )

    return ParsedVerificationSupportSnapshot(
        status="unsupported_version",
        version=support_snapshot_version,
        note=(
            f"Snapshot: persisted support snapshot version {support_snapshot_version} "
            "is not supported for structured rendering."
        ),
    )


def _parse_v1_snapshot(
    support_snapshot: dict[str, object],
    *,
    status: SnapshotReadStatus,
    version: int | None,
    note: str | None,
) -> ParsedVerificationSupportSnapshot:
    snapshot = _as_dict(support_snapshot)
    claim_scope = _parse_claim_scope(snapshot.get("claim_scope"))
    valid_support_links = [_parse_valid_support_link(entry) for entry in _as_list(snapshot.get("valid_support_links"))]
    excluded_support_links = [
        _parse_excluded_support_link(entry) for entry in _as_list(snapshot.get("excluded_support_links"))
    ]
    support_items = [_parse_support_item(entry) for entry in _as_list(snapshot.get("support_items"))]
    citations = [value for value in (_as_str(entry) for entry in _as_list(snapshot.get("citations"))) if value]
    provider_output = _parse_provider_output(snapshot.get("provider_output"))

    return ParsedVerificationSupportSnapshot(
        status=status,
        version=version,
        note=note,
        claim_scope=claim_scope,
        valid_support_links=valid_support_links,
        excluded_support_links=excluded_support_links,
        support_items=support_items,
        citations=citations,
        provider_output=provider_output,
    )


def _parse_claim_scope(value: object) -> ParsedClaimScope | None:
    data = _as_dict(value)
    if not data:
        return None
    return ParsedClaimScope(
        claim_id=_as_str(data.get("claim_id")),
        draft_id=_as_str(data.get("draft_id")),
        matter_id=_as_str(data.get("matter_id")),
        evidence_bundle_id=_as_str(data.get("evidence_bundle_id")),
        scope_kind=_as_str(data.get("scope_kind")),
        allowed_source_document_ids=[
            entry for entry in (_as_str(item) for item in _as_list(data.get("allowed_source_document_ids"))) if entry
        ],
    )


def _parse_valid_support_link(value: object) -> ParsedVerificationSupportLink:
    data = _as_dict(value)
    return ParsedVerificationSupportLink(
        link_id=_as_str(data.get("link_id")),
        claim_id=_as_str(data.get("claim_id")),
        segment_id=_as_str(data.get("segment_id")),
        source_document_id=_as_str(data.get("source_document_id")),
        sequence_order=_as_int(data.get("sequence_order")),
        link_type=_as_str(data.get("link_type")),
        citation_text=_as_str(data.get("citation_text")),
        user_confirmed=bool(data.get("user_confirmed", False)),
        anchor=_as_str(data.get("anchor")),
        evidence_role=_as_str(data.get("evidence_role")),
    )


def _parse_excluded_support_link(value: object) -> ParsedExcludedSupportLink:
    data = _as_dict(value)
    return ParsedExcludedSupportLink(
        link_id=_as_str(data.get("link_id")),
        claim_id=_as_str(data.get("claim_id")),
        segment_id=_as_str(data.get("segment_id")),
        code=_as_str(data.get("code")),
        message=_as_str(data.get("message")),
    )


def _parse_support_item(value: object) -> ParsedSupportItem:
    data = _as_dict(value)
    return ParsedSupportItem(
        order=_as_int(data.get("order")),
        segment_id=_as_str(data.get("segment_id")),
        source_document_id=_as_str(data.get("source_document_id")),
        anchor=_as_str(data.get("anchor")),
        evidence_role=_as_str(data.get("evidence_role")),
        speaker=_as_str(data.get("speaker")),
        segment_type=_as_str(data.get("segment_type")),
        raw_text=_as_str(data.get("raw_text")),
        normalized_text=_as_str(data.get("normalized_text")),
    )


def _parse_provider_output(value: object) -> ParsedProviderOutput:
    data = _as_dict(value)
    return ParsedProviderOutput(
        primary_anchor=_as_str(data.get("primary_anchor")),
        support_assessments=[
            _parse_support_assessment(entry) for entry in _as_list(data.get("support_assessments"))
        ],
    )


def _parse_support_assessment(value: object) -> ParsedSupportAssessment:
    data = _as_dict(value)
    return ParsedSupportAssessment(
        segment_id=_as_str(data.get("segment_id")),
        anchor=_as_str(data.get("anchor")) or "",
        role=_as_str(data.get("role")) or "",
        contribution=_as_str(data.get("contribution")) or "",
    )


def _as_dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _as_list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _as_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _as_int(value: object) -> int | None:
    return value if isinstance(value, int) else None
