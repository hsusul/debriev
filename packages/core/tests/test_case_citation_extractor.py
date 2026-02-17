from debriev_core.citeextract.extract import extract_case_citations


def test_extract_case_citations_multiple_reporters() -> None:
    text = (
        "Roe v. Wade, 410 U.S. 113 (1973), and an appellate cite 123 F.3d 456. "
        "District court cite 12 F. Supp. 3d 45. Supreme Court cite 123 S. Ct. 456. "
        "Parallel cite 12 L. Ed. 2d 34. Ignore statute 18 U.S.C. 1030."
    )

    spans = extract_case_citations(text)
    raws = [span.raw for span in spans]

    assert raws == [
        "410 U.S. 113",
        "123 F.3d 456",
        "12 F. Supp. 3d 45",
        "123 S. Ct. 456",
        "12 L. Ed. 2d 34",
    ]
    assert "18 U.S.C. 1030" not in raws

    assert all(span.start is not None for span in spans)
    assert all(span.end is not None for span in spans)
    assert all(span.context_text for span in spans)
