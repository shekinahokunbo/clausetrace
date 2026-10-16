"""Contract-scoped clause retrieval.

Task: given one contract and a clause category ("Governing Law: which state's
law governs the contract?"), rank that contract's chunks so the lawyer-labeled
clause appears near the top. Every retriever exposes
    fit(train_chunks, train_labels) and score(chunks, categories) -> [n_cat, n_chunk].

Retrievers
  BM25Retriever      lexical baseline, implemented directly on sparse matrices
  DenseRetriever     embedding similarity; default embedder is LSA trained on the
                     training contracts (no external model), or any callable
                     text -> vectors (e.g. a sentence-transformers model)
  LearnedRetriever   per-category logistic-regression clause scorer trained on
                     lawyer labels from the training split only
  HybridRetriever    weighted reciprocal-rank fusion of any of the above
"""
from __future__ import annotations

import re
from typing import Callable, Sequence

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

from .chunking import Chunk

_STOP = set("""a an and are as at be by for from has have in is it its of on or that the this to was were will with
shall such any all may other which been not no than then there these those into upon under""".split())
_TOKEN = re.compile(r"[a-z0-9]+")


def _stem(tok: str) -> str:
    for suf in ("ations", "ation", "ings", "ing", "ies", "ied", "es", "ed", "s"):
        if len(tok) > len(suf) + 3 and tok.endswith(suf):
            return tok[: -len(suf)]
    return tok


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class BM25Retriever:
    name = "bm25"

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b

    def fit(self, train_chunks: Sequence[Chunk], train_labels=None, categories=None):
        self.vec = CountVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
        tf = self.vec.fit_transform([c.text for c in train_chunks])
        df = np.asarray((tf > 0).sum(axis=0)).ravel()
        n = tf.shape[0]
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
        self.avgdl = float(tf.sum(axis=1).mean())
        return self

    def _weights(self, texts: Sequence[str]) -> sparse.csr_matrix:
        tf = self.vec.transform(texts).tocsr().astype(np.float64)
        dl = np.asarray(tf.sum(axis=1)).ravel()
        norm = self.k1 * (1 - self.b + self.b * dl / self.avgdl)
        rows = np.repeat(np.arange(tf.shape[0]), np.diff(tf.indptr))
        tf.data = tf.data * (self.k1 + 1) / (tf.data + norm[rows])
        return tf.multiply(self.idf).tocsr()

    def score(self, chunks: Sequence[Chunk], queries: Sequence[str]) -> np.ndarray:
        W = self._weights([c.text for c in chunks])
        Q = (self.vec.transform(queries) > 0).astype(np.float64)
        return np.asarray((Q @ W.T).todense())


class LSAEmbedder:
    """TF-IDF -> truncated SVD, fit on training contracts only. A corpus-trained
    dense representation that runs anywhere with no model download."""

    def __init__(self, dim: int = 256, seed: int = 0):
        self.dim, self.seed = dim, seed

    def fit(self, texts: Sequence[str]):
        self.tfidf = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None,
                                     ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
        X = self.tfidf.fit_transform(texts)
        self.svd = TruncatedSVD(self.dim, random_state=self.seed).fit(X)
        return self

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        return self.svd.transform(self.tfidf.transform(texts))


class DenseRetriever:
    name = "dense"

    def __init__(self, embedder: Callable[[Sequence[str]], np.ndarray] | None = None, dim: int = 256):
        self.embedder, self.dim = embedder, dim

    def fit(self, train_chunks: Sequence[Chunk], train_labels=None, categories=None):
        if self.embedder is None:
            self.embedder = LSAEmbedder(self.dim).fit([c.text for c in train_chunks])
            self.name = "dense-lsa"
        return self

    def score(self, chunks: Sequence[Chunk], queries: Sequence[str]) -> np.ndarray:
        E = normalize(np.asarray(self.embedder([c.text for c in chunks])))
        Q = normalize(np.asarray(self.embedder(list(queries))))
        return Q @ E.T


class LearnedRetriever:
    """One logistic-regression scorer per clause category over TF-IDF n-grams.
    Positives are training chunks that overlap a lawyer-labeled span."""
    name = "learned"

    def __init__(self, C: float = 4.0):
        self.C = C

    def fit(self, train_chunks: Sequence[Chunk], train_labels: np.ndarray, categories: Sequence[str]):
        self.categories = list(categories)
        self.tfidf = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None,
                                     ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=300_000)
        X = self.tfidf.fit_transform([c.text for c in train_chunks])
        self.models: dict[str, LogisticRegression | None] = {}
        for j, cat in enumerate(self.categories):
            y = train_labels[:, j]
            if y.sum() < 2:
                self.models[cat] = None  # too few lawyer labels to learn from
                continue
            self.models[cat] = LogisticRegression(C=self.C, class_weight="balanced", max_iter=2000).fit(X, y)
        return self

    def score(self, chunks: Sequence[Chunk], queries: Sequence[str], categories: Sequence[str]) -> np.ndarray:
        X = self.tfidf.transform([c.text for c in chunks])
        out = np.zeros((len(categories), len(chunks)))
        for i, cat in enumerate(categories):
            m = self.models.get(cat)
            if m is not None:
                out[i] = m.decision_function(X)
        return out


def ranks_from_scores(scores: np.ndarray) -> np.ndarray:
    """rank[i, j] = 0-based rank of chunk j for query i (stable ties by position)."""
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.empty_like(order)
    rows = np.arange(scores.shape[0])[:, None]
    ranks[rows, order] = np.arange(scores.shape[1])[None, :]
    return ranks


class HybridRetriever:
    """Weighted reciprocal-rank fusion: score = sum_r w_r / (k + rank_r)."""

    def __init__(self, members: dict[str, object], weights: dict[str, float] | None = None, k: int = 60):
        self.members = members
        self.weights = weights or {n: 1.0 for n in members}
        self.k = k
        self.name = "hybrid(" + "+".join(f"{n}:{self.weights[n]:g}" for n in members) + ")"

    def fuse(self, member_scores: dict[str, np.ndarray]) -> np.ndarray:
        total = None
        for name, s in member_scores.items():
            w = self.weights.get(name, 0.0)
            if w == 0:
                continue
            contrib = w / (self.k + ranks_from_scores(s) + 1)
            total = contrib if total is None else total + contrib
        return total
