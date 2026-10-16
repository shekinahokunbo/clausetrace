"""Retrieval evaluation on CUAD.

  python scripts/run_retrieval_eval.py --split dev    # tune: fit on train, score dev
  python scripts/run_retrieval_eval.py --split test   # report: fit on train+dev, score held-out test

Writes results/retrieval_<split>.json and prints a markdown table.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clausetrace.data import load_splits  # noqa: E402
from clausetrace.pipeline import (best_threshold, evaluate, fit_and_score, hybrid, presence_eval,  # noqa: E402
                                  presence_scores, single)

KS = (1, 3, 5, 10)


def table(rows):
    head = "| Retriever | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Hit@5 | MRR | Context sent @5 |"
    out = [head, "|" + "---|" * 8]
    for name, s in rows:
        out.append(f"| {name} | " + " | ".join(f"{100 * s.span_recall[k]:.1f}%" for k in KS)
                   + f" | {100 * s.hit_rate[5]:.1f}% | {s.mrr:.3f} | {100 * s.median_context_fraction[5]:.1f}% |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--chunk-size", type=int, default=1200)
    ap.add_argument("--overlap", type=int, default=300)
    ap.add_argument("--embedder", choices=["lsa", "st"], default="lsa",
                    help="st = sentence-transformers all-MiniLM-L6-v2 (pip install sentence-transformers)")
    ap.add_argument("--weights", default=None, help="fusion weights as bm25,dense,learned; tuned on dev if omitted")
    ap.add_argument("--presence-threshold", type=float, default=None, help="tuned on dev if omitted")
    args = ap.parse_args()

    splits = load_splits()
    fit = splits["train"] if args.split == "dev" else splits["train_full"]
    ev = splits[args.split]
    print(f"fit on {len(fit)} contracts, evaluate on {len(ev)} {args.split} contracts")

    embedder = None
    if args.embedder == "st":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embedder = lambda texts: model.encode(list(texts), batch_size=64, show_progress_bar=False)

    cats, scored, timing = fit_and_score(fit, ev, args.chunk_size, args.overlap, dense_embedder=embedder)
    dense_name = "dense-lsa" if args.embedder == "lsa" else "dense-minilm"
    rows = [("BM25", evaluate(cats, scored, single("bm25"), KS)),
            (f"Dense ({dense_name})", evaluate(cats, scored, single("dense"), KS)),
            ("Hybrid BM25 + dense (RRF)", evaluate(cats, scored, hybrid({"bm25": 1, "dense": 1}), KS)),
            ("Learned clause scorer", evaluate(cats, scored, single("learned"), KS))]

    if args.weights:
        w = dict(zip(["bm25", "dense", "learned"], map(float, args.weights.split(","))))
        best = (w, evaluate(cats, scored, hybrid(w), KS))
    else:  # grid search fusion weights on this split; only ever run with --split dev
        if args.split == "test":
            sys.exit("pass --weights tuned on dev; refusing to tune on test")
        best = None
        for wb, wd, wl in itertools.product([0, 0.5, 1], [0, 0.5, 1], [1, 2, 3, 4]):
            w = {"bm25": wb, "dense": wd, "learned": wl}
            s = evaluate(cats, scored, hybrid(w), KS)
            if best is None or s.span_recall[5] > best[1].span_recall[5]:
                best = (w, s)
        print(f"best dev fusion weights: {best[0]}")
    rows.append((f"Hybrid all (RRF, weights {best[0]['bm25']:g}/{best[0]['dense']:g}/{best[0]['learned']:g})", best[1]))

    y, ps, _ = presence_scores(cats, scored, "learned")
    if args.presence_threshold is None:
        if args.split == "test":
            sys.exit("pass --presence-threshold tuned on dev; refusing to tune on test")
        thr = best_threshold(y, ps)
    else:
        thr = args.presence_threshold
    presence = presence_eval(y, ps, thr)

    print()
    print(table(rows))
    print(f"\nClause presence (all {presence['n']} questions, {presence['positives']} present): "
          f"AUROC {presence['auroc']:.3f}, F1 {presence['f1']:.3f} (P {presence['precision']:.3f} / "
          f"R {presence['recall']:.3f}) at threshold {thr:.3f}")
    out = {"split": args.split, "n_contracts": len(ev), "chunk_size": args.chunk_size, "overlap": args.overlap,
           "embedder": args.embedder, "fusion_weights": best[0], "timing": timing, "presence": presence,
           "rows": [{"retriever": n, **asdict(s)} for n, s in rows]}
    path = ROOT / "results" / f"retrieval_{args.split}{'' if args.embedder == 'lsa' else '_' + args.embedder}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
