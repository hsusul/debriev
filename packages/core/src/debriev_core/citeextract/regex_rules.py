from __future__ import annotations

import re
from re import Pattern

US_REPORTER: Pattern[str] = re.compile(r"\b\d{1,4}\s+U\.S\.\s+\d{1,4}\b")
F2D_F3D_REPORTER: Pattern[str] = re.compile(r"\b\d{1,4}\s+F\.(?:2d|3d)\s+\d{1,4}\b")
F_SUPP_REPORTER: Pattern[str] = re.compile(
    r"\b\d{1,4}\s+F\.\s+Supp\.\s+(?:(?:2d|3d)\s+)?\d{1,4}\b"
)
S_CT_REPORTER: Pattern[str] = re.compile(r"\b\d{1,4}\s+S\.\s+Ct\.\s+\d{1,4}\b")
L_ED_REPORTER: Pattern[str] = re.compile(r"\b\d{1,4}\s+L\.\s+Ed\.\s+(?:2d\s+)?\d{1,4}\b")

CASE_CITATION_PATTERNS: tuple[Pattern[str], ...] = (
    US_REPORTER,
    F2D_F3D_REPORTER,
    F_SUPP_REPORTER,
    S_CT_REPORTER,
    L_ED_REPORTER,
)
