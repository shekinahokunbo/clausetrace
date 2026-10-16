"""Span-level retrieval metrics against lawyer-labeled gold clauses.

A gold clause counts as retrieved at k when the union of the top-k chunks
covers at least `min_cover` of its characters (default 50%). This is stricter
than "any overlap" and matches what a reviewer needs: enough of the clause to
read it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .chunking import Chunk
from .data import Span


def overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def chunk_labels(chunks: list[Chunk], gold: list[Span], min_chars: int = 20) -> np.ndarray:
    """1 for chunks that contain a meaningful part of any gold span."""
    y = np.zeros(len(chunks), dtype=np.int8)
    for i, c in enumerate(chunks):
        for s in gold:
            ov = overlap(c.start, c.end, s.start, s.end)
            if ov >= min(min_chars, len(s)):
                y[i] = 1
                break
    return y


def span_coverage(span: Span, chunks: list[Chunk]) -> float:
    covered = np.zeros(len(span), dtype=bool)
    for c in chunks:
        lo, hi = max(c.start, span.start), min(c.end, span.end)
        if hi > lo:
            covered[lo - span.start:hi - span.start] = True
    return float(covered.mean()) if len(span) else 0.0


@dataclass
class QuestionResult:
    qid: str
    category: str
    n_spans: int
    spans_found_at: dict[int, int]       # k -> gold spans covered at k
    hit_at: dict[int, bool]              # k -> at least one gold span covered
    first_relevant_rank: int | None      # 1-based, chunk-overlap relevance
    context_fraction_at: dict[int, float]


def evaluate_question(qid, category, gold, chunks, scores, text_len, ks=(1, 3, 5, 10), min_cover=0.5):
    order = np.argsort(-scores, kind="stable")
    rel = chunk_labels(chunks, gold)
    hits = np.flatnonzero(rel[order])
    first = int(hits[0]) + 1 if len(hits) else None
    found, hit, ctx = {}, {}, {}
    for k in ks:
        top = [chunks[i] for i in order[:k]]
        n_found = sum(span_coverage(s, top) >= min_cover for s in gold)
        found[k], hit[k] = n_found, n_found > 0
        covered = np.zeros(text_len, dtype=bool)
        for c in top:
            covered[c.start:c.end] = True
        ctx[k] = float(covered.mean())
    return QuestionResult(qid, category, len(gold), found, hit, first, ctx)


@dataclass
class Summary:
    n_questions: int
    n_spans: int
    span_recall: dict[int, float]
    hit_rate: dict[int, float]
    mrr: float
    median_context_fraction: dict[int, float]
    per_category: dict[str, dict] = field(default_factory=dict)


def summarize(results: list[QuestionResult], ks=(1, 3, 5, 10)) -> Summary:
    n_spans = sum(r.n_spans for r in results)
    recall = {k: sum(r.spans_found_at[k] for r in results) / n_spans for k in ks}
    hit = {k: float(np.mean([r.hit_at[k] for r in results])) for k in ks}
    mrr = float(np.mean([1 / r.first_relevant_rank if r.first_relevant_rank else 0.0 for r in results]))
    ctx = {k: float(np.median([r.context_fraction_at[k] for r in results])) for k in ks}
    per_cat: dict[str, dict] = {}
    for cat in sorted({r.category for r in results}):
        rs = [r for r in results if r.category == cat]
        ns = sum(r.n_spans for r in rs)
        per_cat[cat] = {"questions": len(rs), "spans": ns,
                        **{f"recall@{k}": sum(r.spans_found_at[k] for r in rs) / ns for k in ks}}
    return Summary(len(results), n_spans, recall, hit, mrr, ctx, per_cat)
