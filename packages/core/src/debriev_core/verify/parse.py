from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Reporter = Literal["US", "F2D", "F3D", "F_SUPP", "S_CT", "L_ED"]


@dataclass(frozen=True)
class ParsedCitation:
    volume: int
    reporter: Reporter
    page: int


_US_RE = re.compile(r"^\s*(?P<volume>\d{1,4})\s+U\.S\.\s+(?P<page>\d{1,4})\s*$")
_F_RE = re.compile(r"^\s*(?P<volume>\d{1,4})\s+F\.(?P<series>2d|3d)\s+(?P<page>\d{1,4})\s*$")
_F_SUPP_RE = re.compile(
    r"^\s*(?P<volume>\d{1,4})\s+F\.\s+Supp\.\s+(?:(?:2d|3d)\s+)?(?P<page>\d{1,4})\s*$"
)
_S_CT_RE = re.compile(r"^\s*(?P<volume>\d{1,4})\s+S\.\s+Ct\.\s+(?P<page>\d{1,4})\s*$")
_L_ED_RE = re.compile(r"^\s*(?P<volume>\d{1,4})\s+L\.\s+Ed\.\s+(?:2d\s+)?(?P<page>\d{1,4})\s*$")


def parse_case_citation(raw: str) -> ParsedCitation | None:
    match = _US_RE.fullmatch(raw)
    if match:
        return ParsedCitation(
            volume=int(match.group("volume")),
            reporter="US",
            page=int(match.group("page")),
        )

    match = _F_RE.fullmatch(raw)
    if match:
        series = match.group("series")
        return ParsedCitation(
            volume=int(match.group("volume")),
            reporter="F2D" if series == "2d" else "F3D",
            page=int(match.group("page")),
        )

    match = _F_SUPP_RE.fullmatch(raw)
    if match:
        return ParsedCitation(
            volume=int(match.group("volume")),
            reporter="F_SUPP",
            page=int(match.group("page")),
        )

    match = _S_CT_RE.fullmatch(raw)
    if match:
        return ParsedCitation(
            volume=int(match.group("volume")),
            reporter="S_CT",
            page=int(match.group("page")),
        )

    match = _L_ED_RE.fullmatch(raw)
    if match:
        return ParsedCitation(
            volume=int(match.group("volume")),
            reporter="L_ED",
            page=int(match.group("page")),
        )

    return None
