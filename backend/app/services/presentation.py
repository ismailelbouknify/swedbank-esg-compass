"""Analyst-facing wording. The backend keeps full technical detail (model names, extraction method,
rule ids, internal confidence scores) for audit and debugging; nothing here removes it from the
database. These helpers only decide what the normal UI and analyst exports show."""
from __future__ import annotations

import re

# Seven internal pipeline stages -> five business-facing stages shown while an analysis runs.
PUBLIC_STAGES = [
    "Finding company information",
    "Finding sustainability sources",
    "Reading reports",
    "Analysing ESG evidence",
    "Preparing assessment",
]
_INTERNAL_TO_PUBLIC = [1, 2, 3, 4, 4, 4, 5]  # index = internal stage - 1

_TECHNICAL = re.compile(
    r"\b(llm|llama|gguf|ollama|model|heuristic|regex|bm25|embedding|token|prompt|inference|ocr_required|keyword)",
    re.I,
)

# User-facing product branding (exports and report titles only; internal names are unchanged).
PRODUCT_NAME = "Swedbank ESG Compass"
PRODUCT_SUBTITLE = "Evidence-based sustainability assessment"
EXPORT_FILE_PREFIX = "swedbank-esg-compass"

REDUCED_MODE_NOTICE = ("Automated reading of the sources ran in a simplified mode for this analysis. "
                       "All answers should be reviewed by an analyst.")


def public_stage(internal_stage: int) -> int:
    if internal_stage <= 0:
        return 0
    return _INTERNAL_TO_PUBLIC[min(internal_stage, len(_INTERNAL_TO_PUBLIC)) - 1]


def is_technical(text: str) -> bool:
    return bool(_TECHNICAL.search(text or ""))


def analyst_notices(warnings: list[str] | None) -> list[str]:
    """Pipeline warnings rewritten for analysts; technical ones collapse into one plain notice."""
    out: list[str] = []
    for w in warnings or []:
        if "image-only" in w:
            m = re.match(r"'(.+?)'", w)
            msg = f"'{m.group(1)}' is a scanned (image-only) document and could not be read." if m else \
                "A scanned (image-only) document could not be read."
        elif w.startswith("Company website could not be fetched"):
            msg = "The company website could not be reached."
        elif w.startswith("Web search") and "no results" in w:
            msg = "The web search returned no public sources."
        elif w.startswith("No report uploaded"):
            msg = "No report was uploaded, no website was given and web search was off, so there was nothing to review."
        elif is_technical(w):
            msg = REDUCED_MODE_NOTICE
        else:
            msg = w
        if msg not in out:
            out.append(msg)
    return out


def analyst_notes(notes: list[str] | None) -> list[str]:
    return [n for n in (notes or []) if not is_technical(n)]


def review_state(review_status: str, evidence_status: str) -> str:
    """The only three (plus 'changed') states the UI uses."""
    if review_status == "APPROVED":
        return "approved"
    if review_status == "OVERRIDDEN":
        return "changed"
    if evidence_status == "NO_EVIDENCE_FOUND":
        return "no_evidence"
    return "review"


REVIEW_STATE_LABEL = {
    "approved": "Approved",
    "changed": "Changed by analyst",
    "no_evidence": "No evidence",
    "review": "Review",
}

SOURCE_STATUS_LABEL = {
    "PROCESSED": "Processed",
    "OCR_REQUIRED": "Could not be read (scanned document)",
    "FAILED": "Could not be read",
    "SKIPPED": "Skipped",
    "DISCOVERED": "Not processed",
    "DOWNLOADED": "Not processed",
}
