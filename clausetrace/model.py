"""ClauseTrace: the production retriever — fit once, persist, retrieve per request."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from .chunking import Chunk, chunk_contract
from .data import Contract
from .pipeline import build_training_set, category_queries
from .retrieval import BM25Retriever, LearnedRetriever, ranks_from_scores


class ClauseTrace:
    """Learned clause scorer with a BM25 fallback for categories that have too
    few lawyer labels to train on (the scorer returns all zeros there)."""

    def __init__(self, chunk_size: int = 1200, overlap: int = 300):
        self.chunk_size, self.overlap = chunk_size, overlap

    def fit(self, contracts: list[Contract]) -> "ClauseTrace":
        self.queries = category_queries(contracts)
        self.categories = list(self.queries)
        chunks, y = build_training_set(contracts, self.categories, self.chunk_size, self.overlap)
        self.learned = LearnedRetriever().fit(chunks, y, self.categories)
        self.bm25 = BM25Retriever().fit(chunks)
        return self

    def chunks(self, contract: Contract) -> list[Chunk]:
        return chunk_contract(contract.contract_id, contract.text, self.chunk_size, self.overlap)

    def rank(self, chunks: list[Chunk], category: str) -> list[Chunk]:
        s = self.learned.score(chunks, [self.queries[category]], [category])[0]
        if not np.any(s):
            s = self.bm25.score(chunks, [self.queries[category]])[0]
        order = np.argsort(-s, kind="stable")
        return [chunks[i] for i in order]

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "ClauseTrace":
        with open(path, "rb") as f:
            return pickle.load(f)
