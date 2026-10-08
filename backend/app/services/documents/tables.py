"""Optional table extraction with pdfplumber. Used only to enrich pages whose
performance tables are poorly linearised by PyMuPDF."""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


def extract_tables_as_text(path: str | Path, page_numbers: list[int]) -> dict[int, str]:
    """Return {page_number: "cell | cell" rows} for tables found on the given 1-based pages."""
    try:
        import pdfplumber
    except ImportError:  # pragma: no cover
        return {}
    out: dict[int, str] = {}
    try:
        with pdfplumber.open(str(path)) as pdf:
            for n in page_numbers:
                if n < 1 or n > len(pdf.pages):
                    continue
                rows: list[str] = []
                for table in pdf.pages[n - 1].extract_tables() or []:
                    for row in table:
                        cells = [str(c).strip() for c in row if c not in (None, "")]
                        if cells:
                            rows.append(" | ".join(cells))
                if rows:
                    out[n] = "\n".join(rows)
    except Exception as exc:  # noqa: BLE001
        log.warning("pdfplumber table extraction failed for %s: %s", path, exc)
    return out
