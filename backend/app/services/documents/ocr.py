"""OCR adapter stub. OCR is intentionally not part of the MVP pipeline:
image-only PDFs are marked OCR_REQUIRED and surfaced to the analyst."""
from __future__ import annotations

from pathlib import Path


class OCRProvider:
    name = "none"

    def is_available(self) -> bool:
        return False

    def extract_page_texts(self, path: str | Path) -> list[str]:
        raise NotImplementedError(
            "OCR is not enabled. Plug in e.g. Tesseract (pytesseract) or ocrmypdf here."
        )


def get_ocr_provider() -> OCRProvider:
    return OCRProvider()
