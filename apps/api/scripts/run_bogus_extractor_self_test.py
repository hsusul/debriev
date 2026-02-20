from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import (
    ChatResponse,
    _extract_bogus_case_findings,
    _is_bogus_case_request,
    _run_bogus_extractor_self_test,
    _simple_answer_from_chunks,
    _to_chat_findings,
)


if __name__ == "__main__":
    _run_bogus_extractor_self_test()
    findings = _extract_bogus_case_findings(
        [
            {
                "chunk_id": "chunk:script-check",
                "text": (
                    "Citation references Holliday v. Atl. Capital Corp., "
                    "which does not appear to exist. The case found at that citation is A.D. v Azar."
                ),
            }
        ]
    )
    assert findings, "Expected at least one bogus finding in script sanity check"
    for finding in findings:
        assert finding.reason_label.strip(), f"Missing reason_label for finding: {finding.case_name}"
        assert finding.evidence.strip(), f"Missing evidence for finding: {finding.case_name}"
        assert len(finding.evidence) <= 240, f"Evidence too long for finding: {finding.case_name}"

    message = "List bogus and non-existent cases from this order."
    rows = [
        {
            "doc_id": "doc:script-check",
            "chunk_id": "chunk:script-check",
            "text": (
                "Citation references Holliday v. Atl. Capital Corp., "
                "which does not appear to exist. The case found at that citation is A.D. v Azar."
            ),
        }
    ]
    response = ChatResponse(
        answer=_simple_answer_from_chunks(message, rows),
        sources=[],
        findings=_to_chat_findings(_extract_bogus_case_findings(rows)) if _is_bogus_case_request(message) else [],
    )
    payload = response.model_dump()
    assert isinstance(payload.get("findings"), list), "Expected findings array in chat response shape"
    assert payload["findings"], "Expected non-empty findings array for bogus-case request"
    finding_payload = payload["findings"][0]
    expected_keys = {"case_name", "reason_label", "reason_phrase", "evidence", "doc_id", "chunk_id"}
    assert expected_keys.issubset(finding_payload.keys()), "Missing keys in findings payload shape"
    print("ok")
