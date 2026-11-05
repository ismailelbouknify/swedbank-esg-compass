"""Optional table extraction with pdfplumber. Used only to enrich pages whose
performance tables are poorly linearised by PyMuPDF."""
from __future__ import annotations

import io
import logging
import time
from pathlib import Path

log = logging.getLogger(__name__)


def extract_tables_as_text(source: str | Path | bytes, page_numbers: list[int], *, deadline: float | None = None) -> dict[int, str]:
    """Return {page_number: "cell | cell" rows} for tables found on the given 1-based pages.
    Tables are optional enrichment: at the deadline, the pages done so far are returned."""
    try:
        import pdfplumber
    except ImportError:  # pragma: no cover
        return {}
    out: dict[int, str] = {}
    try:
        with pdfplumber.open(io.BytesIO(source) if isinstance(source, bytes) else str(source)) as pdf:
            for n in page_numbers:
                if deadline is not None and time.monotonic() >= deadline:
                    break
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
        log.warning("pdfplumber table extraction failed: %s", type(exc).__name__)
    return out
