from __future__ import annotations

from app.main import _extract_bogus_case_list


def test_bogus_extractor_real_chunk_regression() -> None:
    chunks = [
        {
            "text": (
                "Citation [26] references Varghese v China South Airlines Ltd., 925 F.3d 1339 "
                "(11th Cir. 2019), which does not appear to exist. The case appearing at that "
                "citation is titled Miccosukee Tribe v. United States, 716 F.3d 535 (11th Cir. 2013). "
                "Citation [27] references Zicherman v Korean Airlines Co., Ltd., 516 F. Supp. 2d "
                "1277 (N.D. Ga. 2008), which does not appear to exist."
            )
        },
        {
            "text": (
                "Citation [30] references Gibbs v. Maxwell House, a real case. The case found at "
                "that citation is A.D. v Azar. Docket number 18-12345 is for a case captioned "
                "George Cornea v. U.S. Attorney General."
            )
        },
        {
            "text": (
                "The following five decisions from Federal Reporter, Third, also appear to be fake as well:\n"
                "- Holliday v. Atl. Capital Corp., which does not appear to exist. "
                "The case found at that citation is A.D. v Azar.\n"
                "- Hyatt v. N. Cent. Airlines, which does not appear to exist. "
                "Docket number 18-12345 is for a case captioned George Cornea v. U.S. Attorney General.\n"
            )
        },
    ]

    found = _extract_bogus_case_list(chunks)
    found_set = {item.lower() for item in found}

    expected_includes = {
        "varghese v china south airlines ltd",
        "zicherman v korean airlines co., ltd",
        "holliday v. atl. capital corp",
        "hyatt v. n. cent. airlines",
    }
    expected_excludes = {
        "miccosukee tribe v. united states",
        "gibbs v. maxwell house",
        "a.d. v azar",
        "george cornea v. u.s. attorney general",
    }

    for case_name in expected_includes:
        assert case_name in found_set, f"Expected bogus case missing: {case_name}"
    for case_name in expected_excludes:
        assert case_name not in found_set, f"Unexpected comparator/real case included: {case_name}"
