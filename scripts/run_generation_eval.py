"""End-to-end grounded extraction eval: retrieval -> LLM -> citation verifier.

Compares models on the same sampled questions from the held-out CUAD test split.

  export ANTHROPIC_API_KEY=...   GEMINI_API_KEY=...
  python scripts/run_generation_eval.py --models anthropic:claude-haiku-5-5 google:<gemini-model> --n 100
  python scripts/run_generation_eval.py --models mock --n 20     # offline dry run, no keys

Metrics (per model)
  presence F1         did the model correctly say the clause is present / absent
  span precision      share of quoted characters that fall inside lawyer-labeled spans
  span recall         share of lawyer-labeled characters covered by verified quotes
  rejected quotes     share of model quotes that were NOT verbatim in the sources (caught, never shown)
  correction rate     share of requests that triggered the re-retrieve / retry loop
  latency, tokens
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clausetrace.data import load_splits  # noqa: E402
from clausetrace.generate import AnthropicProvider, GeminiProvider, extract  # noqa: E402
from clausetrace.model import ClauseTrace  # noqa: E402

CACHE = ROOT / ".cache" / "clausetrace.pkl"


class MockProvider:
    """Offline stand-in: quotes the first sentence of the top excerpt. Exercises
    the full pipeline (prompt, parse, verify, retry) without network or keys."""
    name = "mock/first-sentence"

    def complete(self, system, user):
        import re
        m = re.search(r"<excerpt (\d+)>\n(.*?)\n</excerpt", user, re.S)
        idx, body = int(m.group(1)), m.group(2)
        sent = re.split(r"(?<=[.;])\s", body.strip(), maxsplit=1)[0][:300]
        return json.dumps({"found": True, "quotes": [{"excerpt": idx, "text": sent}], "summary": ""}), \
            {"input_tokens": len(user) // 4, "output_tokens": len(sent) // 4}


def make_provider(spec: str):
    if spec == "mock":
        return MockProvider()
    vendor, model = spec.split(":", 1)
    return {"anthropic": AnthropicProvider, "google": GeminiProvider}[vendor](model)


def sample(test, n, seed):
    rng = random.Random(seed)
    pos = [(c, q) for c in test for q in c.questions if q.has_answer]
    neg = [(c, q) for c in test for q in c.questions if not q.has_answer]
    return rng.sample(pos, n // 2) + rng.sample(neg, n - n // 2)


def char_set(spans):
    s = set()
    for a, b in spans:
        s.update(range(a, b))
    return s


def run(provider, ct, items, k, k_retry, retry_on_not_found):
    rows = []
    for i, (c, q) in enumerate(items):
        chunks = ct.chunks(c)
        ranked = ct.rank(chunks, q.category)
        ex = extract(provider, lambda kk: ranked[:kk], q.category, q.description, k, k_retry, retry_on_not_found)
        pred = char_set((v.start, v.end) for v in ex.quotes)
        gold = char_set((s.start, s.end) for s in q.gold)
        rows.append({"qid": q.qid, "category": q.category, "gold_present": q.has_answer, "pred_present": ex.found,
                     "pred_chars": len(pred), "gold_chars": len(gold), "overlap_chars": len(pred & gold),
                     "quotes": [v.text for v in ex.quotes], "proposed": ex.total_quotes, "rejected": ex.rejected_quotes,
                     "parse_failures": ex.parse_failures, "corrected": ex.corrected, "attempts": ex.attempts,
                     "input_tokens": ex.input_tokens, "output_tokens": ex.output_tokens, "latency_s": ex.latency_s,
                     "log": ex.log})
        if (i + 1) % 10 == 0:
            print(f"  {provider.name}: {i + 1}/{len(items)}", flush=True)
    return rows


def summarize(rows, full_context_tokens):
    tp = sum(r["gold_present"] and r["pred_present"] for r in rows)
    fp = sum(not r["gold_present"] and r["pred_present"] for r in rows)
    fn = sum(r["gold_present"] and not r["pred_present"] for r in rows)
    tn = sum(not r["gold_present"] and not r["pred_present"] for r in rows)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    pos = [r for r in rows if r["gold_present"]]
    pred_chars = sum(r["pred_chars"] for r in rows)
    lat = sorted(r["latency_s"] for r in rows)
    proposed = sum(r["proposed"] for r in rows)
    tok_in = statistics.mean(r["input_tokens"] for r in rows)
    return {
        "n": len(rows), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "presence_precision": prec, "presence_recall": rec,
        "presence_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "span_precision": sum(r["overlap_chars"] for r in rows) / pred_chars if pred_chars else 0.0,
        "span_recall": sum(r["overlap_chars"] for r in pos) / sum(r["gold_chars"] for r in pos),
        "quotes_proposed": proposed,
        "rejected_quote_rate": sum(r["rejected"] for r in rows) / proposed if proposed else 0.0,
        "parse_failure_rate": sum(r["parse_failures"] for r in rows) / sum(r["attempts"] for r in rows),
        "correction_rate": statistics.mean(r["corrected"] for r in rows),
        "latency_p50_s": lat[len(lat) // 2], "latency_p95_s": lat[min(len(lat) - 1, int(0.95 * len(lat)))],
        "mean_input_tokens": tok_in, "mean_output_tokens": statistics.mean(r["output_tokens"] for r in rows),
        "mean_full_contract_tokens": full_context_tokens,
        "context_reduction": 1 - tok_in / full_context_tokens,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=100, help="questions, half with the clause present, half absent")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--k-retry", type=int, default=10)
    ap.add_argument("--no-retry-on-not-found", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    splits = load_splits()
    if CACHE.exists():
        ct = ClauseTrace.load(CACHE)
    else:
        print("fitting ClauseTrace on train+dev (one-time, a few minutes)...")
        ct = ClauseTrace().fit(splits["train_full"])
        ct.save(CACHE)
    items = sample(splits["test"], args.n, args.seed)
    full_tokens = statistics.mean(len(c.text) / 4 for c, _ in items)  # ~4 chars/token estimate

    out = {"n": args.n, "k": args.k, "k_retry": args.k_retry, "seed": args.seed,
           "retry_on_not_found": not args.no_retry_on_not_found, "models": {}}
    for spec in args.models:
        prov = make_provider(spec)
        print(f"running {prov.name} on {len(items)} questions")
        rows = run(prov, ct, items, args.k, args.k_retry, not args.no_retry_on_not_found)
        out["models"][prov.name] = {"summary": summarize(rows, full_tokens), "rows": rows}

    print("\n| Model | Presence F1 | Precision | Recall | Span precision | Span recall | Quotes rejected | Retry rate | p50 latency | Input tokens vs full contract |")
    print("|" + "---|" * 10)
    for name, m in out["models"].items():
        s = m["summary"]
        print(f"| {name} | {s['presence_f1']:.3f} | {100 * s['presence_precision']:.1f}% | {100 * s['presence_recall']:.1f}% | "
              f"{100 * s['span_precision']:.1f}% | {100 * s['span_recall']:.1f}% | {100 * s['rejected_quote_rate']:.1f}% | "
              f"{100 * s['correction_rate']:.1f}% | {s['latency_p50_s']:.2f}s | -{100 * s['context_reduction']:.0f}% |")
    tag = "mock" if args.models == ["mock"] else "models"
    path = ROOT / "results" / f"generation_{tag}_n{args.n}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
