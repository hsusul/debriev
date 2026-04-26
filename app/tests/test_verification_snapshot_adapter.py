from uuid import uuid4

from app.services.verification.snapshot_adapter import (
    LEGACY_UNVERSIONED_SNAPSHOT_NOTE,
    parse_verification_support_snapshot,
)


def build_support_snapshot() -> dict[str, object]:
    return {
        "claim_scope": {
            "claim_id": str(uuid4()),
            "draft_id": str(uuid4()),
            "matter_id": str(uuid4()),
            "evidence_bundle_id": str(uuid4()),
            "scope_kind": "bundle",
            "allowed_source_document_ids": [str(uuid4()), str(uuid4())],
        },
        "valid_support_links": [
            {
                "link_id": str(uuid4()),
                "claim_id": str(uuid4()),
                "segment_id": str(uuid4()),
                "source_document_id": str(uuid4()),
                "sequence_order": 1,
                "link_type": "MANUAL",
                "citation_text": None,
                "user_confirmed": True,
                "anchor": "p.10:3-10:4",
                "evidence_role": "substantive",
            }
        ],
        "excluded_support_links": [
            {
                "link_id": str(uuid4()),
                "claim_id": str(uuid4()),
                "segment_id": str(uuid4()),
                "code": "out_of_scope_support_link",
                "message": "Segment source document is outside the draft evidence bundle.",
            }
        ],
        "support_items": [
            {
                "order": 1,
                "segment_id": str(uuid4()),
                "source_document_id": str(uuid4()),
                "anchor": "p.10:3-10:4",
                "evidence_role": "substantive",
                "speaker": "A",
                "segment_type": "ANSWER_BLOCK",
                "raw_text": "A. Smith signed the contract.",
                "normalized_text": "a smith signed the contract",
            }
        ],
        "citations": ["p.10:3-10:4"],
        "provider_output": {
            "primary_anchor": "p.10:3-10:4",
            "support_assessments": [
                {
                    "segment_id": str(uuid4()),
                    "anchor": "p.10:3-10:4",
                    "role": "primary",
                    "contribution": "Direct answer testimony states the proposition.",
                }
            ],
        },
    }


def test_parse_v1_support_snapshot() -> None:
    parsed = parse_verification_support_snapshot(build_support_snapshot(), 1)

    assert parsed.status == "versioned_v1"
    assert parsed.version == 1
    assert parsed.note is None
    assert parsed.claim_scope is not None
    assert parsed.claim_scope.scope_kind == "bundle"
    assert len(parsed.claim_scope.allowed_source_document_ids) == 2
    assert parsed.provider_output.primary_anchor == "p.10:3-10:4"
    assert parsed.provider_output.support_assessments[0].role == "primary"
    assert parsed.excluded_support_links[0].code == "out_of_scope_support_link"


def test_parse_legacy_unversioned_snapshot_as_v1_compatible() -> None:
    parsed = parse_verification_support_snapshot(build_support_snapshot(), None)

    assert parsed.status == "legacy_unversioned_v1"
    assert parsed.version is None
    assert parsed.note == LEGACY_UNVERSIONED_SNAPSHOT_NOTE
    assert parsed.provider_output.primary_anchor == "p.10:3-10:4"


def test_parse_unsupported_future_snapshot_version_conservatively() -> None:
    parsed = parse_verification_support_snapshot(build_support_snapshot(), 99)

    assert parsed.status == "unsupported_version"
    assert parsed.version == 99
    assert parsed.note == "Snapshot: persisted support snapshot version 99 is not supported for structured rendering."
    assert parsed.claim_scope is None
    assert parsed.provider_output.primary_anchor is None
    assert parsed.support_items == []


def test_parse_missing_snapshot_returns_explicit_missing_result() -> None:
    parsed = parse_verification_support_snapshot(None, None)

    assert parsed.status == "missing"
    assert parsed.version is None
    assert parsed.note is None
    assert parsed.claim_scope is None
    assert parsed.support_items == []
