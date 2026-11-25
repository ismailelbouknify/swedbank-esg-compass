"""Page-aware, overlapping text chunking for retrieval.

Chunks never span pages, so every chunk has one page number. Sizes are in approximate model tokens (configurable;
default ~350 with ~60 overlap). Each chunk records the section heading it falls under. Table rows are kept whole, and
when a table continues into a new chunk its header row (years, units, column names) is repeated at the top, so a
number is never separated from the label and period that give it meaning."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(•-])")
_NUMBER = re.compile(r"(?<![A-Za-z])\d+(?:[.,]\d+)?%?")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
TABLE_MARKER = "[Table]"


def approx_tokens(text: str) -> int:
    """A conservative token estimate without a tokenizer: English prose is ~0.75 words per token, and number-heavy
    text costs more, so the larger of a word-based and a character-based estimate is used."""
    return max(1, round(max(len(text.split()) * 4 / 3, len(text) / 4)))


@dataclass
class Chunk:
    page_number: int | None
    chunk_index: int
    text: str
    section_heading: str | None = None
    is_table: bool = False


@dataclass
class _Unit:
    text: str
    heading: str | None
    is_table: bool = False
    table_header: str | None = None  # the header row this table row belongs to
    tokens: int = field(init=False)

    def __post_init__(self) -> None:
        self.tokens = approx_tokens(self.text)


def is_heading(line: str) -> bool:
    """A short title line: no sentence punctuation, few numbers, starts with a capital or a section number."""
    words = line.split()
    if not 1 <= len(words) <= 10 or len(line) > 90 or "|" in line:
        return False
    if line[-1] in ".,;:!?" or len(_NUMBER.findall(line)) > 1:
        return False
    return line[0].isupper() or bool(re.match(r"^\d+(\.\d+)*\s+[A-Z]", line))


def is_table_header(line: str) -> bool:
    """A row naming two or more periods with at most one other number, e.g. "Indicator 2023 2024 2025"."""
    years = set(_YEAR.findall(line))
    others = [n for n in _NUMBER.findall(line) if n not in years]
    return len(years) >= 2 and len(others) <= 1 and len(line.split()) <= 20 and not line.rstrip().endswith(".")


def is_table_row(line: str) -> bool:
    """Cells joined by " | " (pdfplumber), or a label followed by figures: numbers are a large share of the line.
    A sentence that happens to contain two numbers ("by 30% by 2030") is prose."""
    if " | " in line:
        return True
    numbers = len(_NUMBER.findall(line))
    return numbers >= 2 and numbers / max(1, len(line.split())) >= 0.25 and not line.rstrip().endswith(".")


def _page_units(text: str, heading: str | None, max_tokens: int) -> tuple[list[_Unit], str | None]:
    units: list[_Unit] = []
    prose: list[str] = []
    table_header: str | None = None
    in_table = False

    def flush_prose() -> None:
        if prose:
            for sentence in SENTENCE_SPLIT.split(" ".join(prose)):
                sentence = sentence.strip()
                if sentence:
                    units.extend(_split_long(sentence, heading, max_tokens))
            prose.clear()

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            flush_prose()
            in_table = False
            continue
        if line == TABLE_MARKER:
            flush_prose()
            in_table, table_header = True, None
            continue
        if in_table or is_table_header(line) or is_table_row(line):
            flush_prose()
            if is_table_header(line) or (in_table and table_header is None):
                table_header = line
                units.append(_Unit(line, heading, True, None))
            else:
                units.extend(_split_long(line, heading, max_tokens, table=True, header=table_header))
            continue
        if is_heading(line):
            flush_prose()
            heading, table_header = line, None
            units.append(_Unit(line, heading))
            continue
        table_header = None
        prose.append(line)
    flush_prose()
    return units, heading


def _split_long(text: str, heading: str | None, max_tokens: int, table: bool = False, header: str | None = None) -> list[_Unit]:
    if approx_tokens(text) <= max_tokens:
        return [_Unit(text, heading, table, header)]
    words = text.split()
    step = max(1, int(max_tokens * 3 / 4))
    return [_Unit(" ".join(words[i:i + step]), heading, table, header) for i in range(0, len(words), step)]


def chunk_pages(pages: list[tuple[int | None, str]], size_tokens: int = 350, overlap_tokens: int = 60) -> list[Chunk]:
    """pages: [(page_number, text)]. Returns overlapping chunks of roughly `size_tokens` tokens."""
    if size_tokens <= 0:
        raise ValueError("size_tokens must be positive")
    overlap_tokens = max(0, min(overlap_tokens, size_tokens // 2))
    chunks: list[Chunk] = []
    heading: str | None = None
    for page_number, text in pages:
        units, next_heading = _page_units(text or "", heading, size_tokens)
        current: list[_Unit] = []
        count = 0

        def emit() -> None:
            body = "\n".join(u.text for u in current)
            first_heading = next((u.heading for u in current if u.heading), heading)
            chunks.append(Chunk(page_number, len(chunks), body, first_heading, any(u.is_table for u in current)))

        for unit in units:
            if current and count + unit.tokens > size_tokens:
                emit()
                carry: list[_Unit] = []
                carried = 0
                for prev in reversed(current):
                    if carried + prev.tokens > overlap_tokens:
                        break
                    carry.insert(0, prev)
                    carried += prev.tokens
                if carried + unit.tokens > size_tokens:
                    carry, carried = [], 0
                current, count = carry, carried
                # a table continuing here keeps its header row: the numbers below stay interpretable
                if unit.table_header and all(u.text != unit.table_header for u in current):
                    header = _Unit(unit.table_header, unit.heading, True)
                    current.insert(0, header)
                    count += header.tokens
            current.append(unit)
            count += unit.tokens
        if current:
            emit()
        heading = next_heading
    return chunks
