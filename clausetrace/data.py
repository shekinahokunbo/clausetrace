"""Load CUAD (Contract Understanding Atticus Dataset) into typed records.

CUAD v1: 510 commercial contracts, 41 clause categories, spans labeled by
lawyers. Each (contract, category) pair is one question; most pairs have no
answer because the clause is absent, which the evaluation must respect.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass(frozen=True)
class Span:
    start: int
    end: int  # exclusive

    def __len__(self) -> int:
        return self.end - self.start


@dataclass
class Question:
    qid: str
    contract_id: str
    category: str
    description: str
    gold: list[Span] = field(default_factory=list)

    @property
    def has_answer(self) -> bool:
        return bool(self.gold)

    @property
    def query(self) -> str:
        return f"{self.category}. {self.description}"


@dataclass
class Contract:
    contract_id: str
    text: str
    questions: list[Question]


def _parse(entry: dict) -> Contract:
    para = entry["paragraphs"][0]
    cid = entry["title"]
    text = para["context"]
    questions = []
    for qa in para["qas"]:
        category = qa["id"].split("__")[-1]
        q = qa["question"]
        description = q.split("Details:", 1)[1].strip() if "Details:" in q else category
        spans = sorted({Span(a["answer_start"], a["answer_start"] + len(a["text"])) for a in qa["answers"]},
                       key=lambda s: (s.start, s.end))
        for s in spans:  # guard against offset drift in the source file
            assert text[s.start:s.end].strip(), f"empty gold span in {qa['id']}"
        questions.append(Question(qa["id"], cid, category, description, list(spans)))
    return Contract(cid, text, questions)


def load_all(data_dir: Path = DATA_DIR) -> list[Contract]:
    return [_parse(e) for e in json.load(open(data_dir / "CUADv1.json"))["data"]]


def _is_dev(contract_id: str, dev_fraction: float) -> bool:
    h = int(hashlib.sha256(contract_id.encode()).hexdigest(), 16)
    return (h % 10_000) / 10_000 < dev_fraction


def load_splits(data_dir: Path = DATA_DIR, dev_fraction: float = 0.2) -> dict[str, list[Contract]]:
    """Official CUAD test split (102 contracts) held out; train split further
    divided into train/dev by a deterministic hash so tuning never sees test."""
    test_ids = {e["title"] for e in json.load(open(data_dir / "test.json"))["data"]}
    contracts = load_all(data_dir)
    test = [c for c in contracts if c.contract_id in test_ids]
    train_full = [c for c in contracts if c.contract_id not in test_ids]
    dev = [c for c in train_full if _is_dev(c.contract_id, dev_fraction)]
    train = [c for c in train_full if not _is_dev(c.contract_id, dev_fraction)]
    return {"train": train, "dev": dev, "train_full": train_full, "test": test}
