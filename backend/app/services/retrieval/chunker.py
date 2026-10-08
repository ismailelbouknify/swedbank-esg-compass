"""Page-aware text chunking. Chunks never span pages, so every chunk has one page number."""
from __future__ import annotations

import re
from dataclasses import dataclass

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\u2022-])|\n{2,}")


@dataclass
class Chunk:
    page_number: int | None
    chunk_index: int
    text: str


def _units(text: str, max_words: int) -> list[str]:
    """Sentence-ish units; over-long units (tables, bullet walls) are split by line then words."""
    units: list[str] = []
    for part in SENTENCE_SPLIT.split(text):
        part = part.strip()
        if not part:
            continue
        if len(part.split()) <= max_words:
            units.append(part)
            continue
        for line in part.split("\n"):
            words = line.split()
            for i in range(0, len(words), max_words):
                piece = " ".join(words[i : i + max_words])
                if piece:
                    units.append(piece)
    return units


def chunk_pages(pages: list[tuple[int | None, str]], size_words: int = 180, overlap_words: int = 40) -> list[Chunk]:
    """pages: [(page_number, text)]. Returns overlapping chunks of roughly size_words words."""
    if size_words <= 0:
        raise ValueError("size_words must be positive")
    overlap_words = max(0, min(overlap_words, size_words // 2))
    chunks: list[Chunk] = []
    idx = 0
    for page_number, text in pages:
        units = _units(text, size_words)
        current: list[str] = []
        count = 0
        for unit in units:
            n = len(unit.split())
            if current and count + n > size_words:
                chunks.append(Chunk(page_number, idx, "\n".join(current)))
                idx += 1
                # carry over trailing units as overlap
                carry: list[str] = []
                carried = 0
                for prev in reversed(current):
                    if carried >= overlap_words:
                        break
                    carry.insert(0, prev)
                    carried += len(prev.split())
                if carried + n > size_words:
                    carry, carried = [], 0
                current, count = carry, carried
            current.append(unit)
            count += n
        if current:
            chunks.append(Chunk(page_number, idx, "\n".join(current)))
            idx += 1
    return chunks
