"""Fit retrievers on one split, score another, and evaluate every configuration."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .chunking import Chunk, chunk_contract
from .data import Contract
from .metrics import chunk_labels, evaluate_question, summarize
from .retrieval import BM25Retriever, DenseRetriever, HybridRetriever, LearnedRetriever


def category_queries(contracts: list[Contract]) -> dict[str, str]:
    out: dict[str, str] = {}
    for c in contracts:
        for q in c.questions:
            out.setdefault(q.category, q.query)
    return dict(sorted(out.items()))


@dataclass
class Scored:
    contract: Contract
    chunks: list[Chunk]
    member_scores: dict[str, np.ndarray]  # retriever -> [n_categories, n_chunks]


def build_training_set(contracts, categories, size, overlap):
    chunks, labels = [], []
    for c in contracts:
        cs = chunk_contract(c.contract_id, c.text, size, overlap)
        by_cat = {q.category: q.gold for q in c.questions}
        y = np.zeros((len(cs), len(categories)), dtype=np.int8)
        for j, cat in enumerate(categories):
            if by_cat.get(cat):
                y[:, j] = chunk_labels(cs, by_cat[cat])
        chunks.extend(cs)
        labels.append(y)
    return chunks, np.vstack(labels)


def fit_and_score(fit_contracts, eval_contracts, size=800, overlap=200, members=("bm25", "dense", "learned"),
                  dense_embedder=None, log=print):
    queries = category_queries(fit_contracts + eval_contracts)
    categories, qtexts = list(queries), list(queries.values())
    t0 = time.time()
    train_chunks, train_y = build_training_set(fit_contracts, categories, size, overlap)
    log(f"  {len(train_chunks):,} training chunks, {int(train_y.sum()):,} positive labels")
    fitted = {}
    if "bm25" in members:
        fitted["bm25"] = BM25Retriever().fit(train_chunks)
    if "dense" in members:
        fitted["dense"] = DenseRetriever(dense_embedder).fit(train_chunks)
    if "learned" in members:
        fitted["learned"] = LearnedRetriever().fit(train_chunks, train_y, categories)
    log(f"  fit in {time.time() - t0:.1f}s")
    t0 = time.time()
    scored = []
    for c in eval_contracts:
        cs = chunk_contract(c.contract_id, c.text, size, overlap)
        ms = {}
        for name, r in fitted.items():
            ms[name] = r.score(cs, qtexts, categories) if name == "learned" else r.score(cs, qtexts)
        scored.append(Scored(c, cs, ms))
    elapsed = time.time() - t0
    n_q = len(eval_contracts) * len(categories)
    log(f"  scored {len(eval_contracts)} contracts in {elapsed:.1f}s ({1000 * elapsed / n_q:.2f} ms per query, all retrievers)")
    return categories, scored, {"score_seconds": elapsed, "queries": n_q}


def evaluate(categories, scored, combine, ks=(1, 3, 5, 10)):
    """combine: member_scores -> [n_cat, n_chunk] score matrix."""
    cat_index = {c: i for i, c in enumerate(categories)}
    results = []
    for s in scored:
        S = combine(s.member_scores)
        for q in s.contract.questions:
            if not q.has_answer:
                continue
            results.append(evaluate_question(q.qid, q.category, q.gold, s.chunks, S[cat_index[q.category]],
                                             len(s.contract.text), ks))
    return summarize(results, ks)


def single(name):
    return lambda ms: ms[name]


def hybrid(weights: dict[str, float], k: int = 60):
    h = HybridRetriever({n: None for n in weights}, weights, k)
    return h.fuse


def presence_scores(categories, scored, member="learned"):
    """Max chunk score per (contract, category) as a clause-presence signal.
    Returns (labels, scores, category) over ALL questions, including absent ones."""
    cat_index = {c: i for i, c in enumerate(categories)}
    y, s, cats = [], [], []
    for sc in scored:
        S = sc.member_scores[member]
        for q in sc.contract.questions:
            y.append(int(q.has_answer))
            s.append(float(S[cat_index[q.category]].max()))
            cats.append(q.category)
    return np.array(y), np.array(s), cats


def presence_eval(y, s, threshold):
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
    pred = (s >= threshold).astype(int)
    return {"auroc": float(roc_auc_score(y, s)), "threshold": float(threshold),
            "precision": float(precision_score(y, pred)), "recall": float(recall_score(y, pred)),
            "f1": float(f1_score(y, pred)), "n": int(len(y)), "positives": int(y.sum())}


def best_threshold(y, s):
    from sklearn.metrics import f1_score
    cands = np.quantile(s, np.linspace(0.3, 0.95, 131))
    return float(max(cands, key=lambda t: f1_score(y, (s >= t).astype(int))))
