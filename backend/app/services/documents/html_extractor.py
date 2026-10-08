"""HTML main-text extraction (trafilatura, falling back to BeautifulSoup) and link discovery."""
from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from app.services.documents.pdf_extractor import clean_text

RELEVANT_LINK_TERMS = (
    "sustainab", "esg", "csr", "responsib", "environment", "climate", "energy", "diversity",
    "inclusion", "people", "employees", "privacy", "security", "annual-report", "annual report",
    "reports", "investor", "impact", "circular", "governance",
)


@dataclass
class HtmlContent:
    title: str | None
    text: str
    links: list[tuple[str, str]]  # (absolute url, anchor text)


def extract_html(html: str | bytes, base_url: str) -> HtmlContent:
    if isinstance(html, bytes):
        html = html.decode("utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else None

    text = None
    try:
        import trafilatura

        text = trafilatura.extract(html, include_tables=True, include_comments=False, favor_recall=True)
    except Exception:  # noqa: BLE001
        text = None
    if not text:
        for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "form", "svg"]):
            tag.decompose()
        text = soup.get_text("\n")
    text = clean_text(re.sub(r"\n\s*\n+", "\n\n", text))

    links: list[tuple[str, str]] = []
    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a["href"]).split("#")[0]
        if urlparse(href).scheme in ("http", "https"):
            links.append((href, a.get_text(" ", strip=True)[:200]))
    return HtmlContent(title=title, text=text, links=links)


def relevant_links(content: HtmlContent, domain: str | None, limit: int) -> list[str]:
    """Pick same-domain links whose URL or anchor looks ESG-related, PDFs first."""
    scored: dict[str, int] = {}
    for url, anchor in content.links:
        host = (urlparse(url).hostname or "").lower()
        if domain and not (host == domain or host.endswith("." + domain)):
            continue
        hay = (url + " " + anchor).lower()
        score = sum(term in hay for term in RELEVANT_LINK_TERMS)
        if score == 0:
            continue
        if url.lower().split("?")[0].endswith(".pdf"):
            score += 3
        if "report" in hay:
            score += 2
        scored[url] = max(score, scored.get(url, 0))
    return [u for u, _ in sorted(scored.items(), key=lambda kv: -kv[1])][:limit]
