from uuid import uuid4

from app.services.verification.snapshot_adapter import (
    LEGACY_UNVERSIONED_SNAPSHOT_NOTE,
    parse_verification_support_snapshot,
)
from app.services.verification.snapshot_mapper import (
    map_parsed_snapshot_to_audit_view,
    map_parsed_snapshot_to_history_fields,
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


def test_map_v1_snapshot_to_history_fields() -> None:
    parsed = parse_verification_support_snapshot(build_support_snapshot(), 1)

    mapped = map_parsed_snapshot_to_history_fields(parsed)

    assert mapped.support_snapshot_status == "versioned_v1"
    assert mapped.support_snapshot_note is None
    assert mapped.support_snapshot_version == 1
    assert mapped.support_snapshot is not None
    assert mapped.support_snapshot.provider_output.primary_anchor == "p.10:3-10:4"
    assert mapped.support_snapshot.support_items[0].anchor == "p.10:3-10:4"


def test_map_legacy_unversioned_snapshot_to_history_fields() -> None:
    parsed = parse_verification_support_snapshot(build_support_snapshot(), None)

    mapped = map_parsed_snapshot_to_history_fields(parsed)

    assert mapped.support_snapshot_status == "legacy_unversioned_v1"
    assert mapped.support_snapshot_note == LEGACY_UNVERSIONED_SNAPSHOT_NOTE
    assert mapped.support_snapshot_version is None
    assert mapped.support_snapshot is not None
    assert mapped.support_snapshot.claim_scope.scope_kind == "bundle"


def test_map_unsupported_snapshot_version_to_history_fields_conservatively() -> None:
    parsed = parse_verification_support_snapshot(build_support_snapshot(), 99)

    mapped = map_parsed_snapshot_to_history_fields(parsed)

    assert mapped.support_snapshot_status == "unsupported_version"
    assert mapped.support_snapshot_version == 99
    assert mapped.support_snapshot is None
    assert mapped.support_snapshot_note == (
        "Snapshot: persisted support snapshot version 99 is not supported for structured rendering."
    )


def test_map_v1_snapshot_to_audit_view() -> None:
    parsed = parse_verification_support_snapshot(build_support_snapshot(), 1)

    mapped = map_parsed_snapshot_to_audit_view(parsed)

    assert mapped is not None
    assert mapped.support_snapshot_version == 1
    assert mapped.scope_kind == "bundle"
    assert mapped.allowed_source_document_count == 2
    assert mapped.primary_anchor == "p.10:3-10:4"
    assert mapped.support_assessments[0].role == "primary"
    assert mapped.excluded_support_links[0].code == "out_of_scope_support_link"


def test_map_missing_snapshot_to_audit_view_returns_none() -> None:
    parsed = parse_verification_support_snapshot(None, None)

    mapped = map_parsed_snapshot_to_audit_view(parsed)

    assert mapped is None
