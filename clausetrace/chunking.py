"""Split a contract into overlapping chunks that keep exact character offsets.

Offsets are the point: every retrieved chunk, and every quote a model cites,
can be traced back to an exact location in the source contract.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_BOUNDARY = re.compile(r"\n\s*\n|(?<=[.;:])\s+")


@dataclass(frozen=True)
class Chunk:
    contract_id: str
    index: int
    start: int
    end: int  # exclusive
    text: str

    @property
    def chunk_id(self) -> str:
        return f"{self.index}"


def chunk_contract(contract_id: str, text: str, size: int = 800, overlap: int = 200) -> list[Chunk]:
    """Greedy windows of ~`size` chars, snapped back to the nearest paragraph or
    sentence boundary so clauses are not cut mid-sentence where avoidable."""
    if size <= overlap:
        raise ValueError("size must exceed overlap")
    chunks: list[Chunk] = []
    n = len(text)
    start = 0
    while start < n:
        end = min(start + size, n)
        if end < n:
            window = text[start + size // 2:end]
            cuts = [m.end() for m in _BOUNDARY.finditer(window)]
            if cuts:
                end = start + size // 2 + cuts[-1]
        piece = text[start:end]
        if piece.strip():
            chunks.append(Chunk(contract_id, len(chunks), start, end, piece))
        if end >= n:
            break
        nxt = end - overlap
        # snap the next start forward to a word boundary
        while 0 < nxt < n and not text[nxt - 1].isspace():
            nxt += 1
        start = max(nxt, start + 1)
    return chunks
