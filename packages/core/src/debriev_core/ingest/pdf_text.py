from __future__ import annotations

import fitz


def extract_pdf_text(pdf_path: str) -> str:
    parts: list[str] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            parts.append(page.get_text())
    return "\n".join(parts)
