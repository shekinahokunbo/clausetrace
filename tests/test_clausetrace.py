import json
import unittest

import numpy as np

from clausetrace.chunking import Chunk, chunk_contract
from clausetrace.data import Span
from clausetrace.generate import extract, locate, parse_response, verify
from clausetrace.metrics import chunk_labels, evaluate_question, span_coverage, summarize
from clausetrace.retrieval import BM25Retriever, HybridRetriever, ranks_from_scores, tokenize

TEXT = ("This Agreement is governed by the laws of the State of Delaware. "
        "Either party may terminate this Agreement upon thirty days written notice. "
        "The Supplier shall indemnify the Buyer against all third-party claims. ") * 3


class TestChunking(unittest.TestCase):
    def test_chunks_tile_text_and_keep_offsets(self):
        chunks = chunk_contract("c", TEXT, size=120, overlap=30)
        covered = set()
        for c in chunks:
            self.assertEqual(TEXT[c.start:c.end], c.text)
            covered.update(range(c.start, c.end))
        self.assertEqual(covered, set(range(len(TEXT))))

    def test_rejects_overlap_larger_than_size(self):
        with self.assertRaises(ValueError):
            chunk_contract("c", TEXT, size=100, overlap=100)

    def test_short_text_is_one_chunk(self):
        self.assertEqual(len(chunk_contract("c", "short contract", 800, 200)), 1)


class TestRetrieval(unittest.TestCase):
    def setUp(self):
        self.chunks = [Chunk("c", i, 0, 1, t) for i, t in enumerate(
            ["governed by the laws of Delaware", "terminate upon thirty days notice", "indemnify against claims"])]

    def test_bm25_ranks_matching_chunk_first(self):
        r = BM25Retriever().fit(self.chunks)
        s = r.score(self.chunks, ["termination notice", "governing law"])
        self.assertEqual(int(np.argmax(s[0])), 1)
        self.assertEqual(int(np.argmax(s[1])), 0)

    def test_tokenize_stems_and_drops_stopwords(self):
        self.assertEqual(tokenize("The terminating of agreements"), ["terminat", "agreement"])

    def test_ranks_from_scores(self):
        r = ranks_from_scores(np.array([[0.1, 0.9, 0.5]]))
        self.assertEqual(r.tolist(), [[2, 0, 1]])

    def test_rrf_respects_weights(self):
        a = np.array([[3.0, 2.0, 1.0]])   # prefers chunk 0
        b = np.array([[1.0, 2.0, 3.0]])   # prefers chunk 2
        fused = HybridRetriever({"a": None, "b": None}, {"a": 1.0, "b": 0.0}).fuse({"a": a, "b": b})
        self.assertEqual(int(np.argmax(fused[0])), 0)


class TestMetrics(unittest.TestCase):
    def test_span_coverage_union_of_chunks(self):
        span = Span(10, 30)
        chunks = [Chunk("c", 0, 0, 20, ""), Chunk("c", 1, 25, 40, "")]
        self.assertAlmostEqual(span_coverage(span, chunks), 15 / 20)

    def test_chunk_labels_need_meaningful_overlap(self):
        chunks = [Chunk("c", 0, 0, 100, ""), Chunk("c", 1, 95, 200, "")]
        y = chunk_labels(chunks, [Span(96, 150)])
        self.assertEqual(y.tolist(), [0, 1])  # 4 chars of overlap is not enough

    def test_recall_and_mrr(self):
        chunks = [Chunk("c", 0, 0, 50, ""), Chunk("c", 1, 50, 100, ""), Chunk("c", 2, 100, 150, "")]
        gold = [Span(60, 90)]
        r = evaluate_question("q", "cat", gold, chunks, np.array([0.9, 0.5, 0.1]), 150, ks=(1, 3))
        s = summarize([r], ks=(1, 3))
        self.assertEqual(s.span_recall[1], 0.0)
        self.assertEqual(s.span_recall[3], 1.0)
        self.assertAlmostEqual(s.mrr, 0.5)


class TestVerification(unittest.TestCase):
    def setUp(self):
        self.chunk = Chunk("c", 4, 1000, 1000 + 60, "The laws of the\n  State of Delaware govern this Agreement.")

    def test_locate_tolerates_whitespace_only(self):
        loc = locate("laws of the State of Delaware", self.chunk)
        self.assertIsNotNone(loc)
        a, b = loc
        self.assertEqual(" ".join(self.chunk.text[a - 1000:b - 1000].split()), "laws of the State of Delaware")

    def test_locate_rejects_paraphrase(self):
        self.assertIsNone(locate("Delaware law governs the Agreement", self.chunk))
        self.assertIsNone(locate("laws of the State of New York", self.chunk))

    def test_verify_reattributes_wrong_excerpt_number(self):
        other = Chunk("c", 9, 0, 10, "unrelated")
        ok, bad = verify({"quotes": [{"excerpt": 9, "text": "State of Delaware"}]}, [other, self.chunk])
        self.assertEqual((len(ok), len(bad)), (1, 0))
        self.assertEqual(ok[0].excerpt, 4)

    def test_parse_response_handles_wrapped_json(self):
        d = parse_response('Sure.\n```json\n{"found": false, "quotes": []}\n```')
        self.assertEqual(d["found"], False)
        self.assertIsNone(parse_response("no json here"))


class ScriptedProvider:
    name = "scripted"

    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def complete(self, system, user):
        self.calls.append(user)
        return self.replies.pop(0), {"input_tokens": 10, "output_tokens": 5}


class TestSelfCorrection(unittest.TestCase):
    def setUp(self):
        self.chunks = chunk_contract("c", TEXT, size=120, overlap=30)
        self.retrieve = lambda k: self.chunks[:k]

    def test_hallucinated_quote_triggers_retry_and_is_never_returned(self):
        bad = json.dumps({"found": True, "quotes": [{"excerpt": 0, "text": "governed by the laws of New York"}]})
        good = json.dumps({"found": True, "quotes": [{"excerpt": 0, "text": "the laws of the State of Delaware"}]})
        p = ScriptedProvider([bad, good])
        ex = extract(p, self.retrieve, "Governing Law", "which law governs", k=1, k_retry=3)
        self.assertTrue(ex.corrected)
        self.assertEqual(ex.rejected_quotes, 1)
        self.assertEqual([q.text for q in ex.quotes], ["the laws of the State of Delaware"])
        self.assertIn("not verbatim", p.calls[1])          # feedback reached the model
        self.assertGreater(p.calls[1].count("<excerpt"), p.calls[0].count("<excerpt"))  # wider k

    def test_clean_answer_makes_one_call(self):
        good = json.dumps({"found": True, "quotes": [{"excerpt": 0, "text": "State of Delaware"}]})
        p = ScriptedProvider([good])
        ex = extract(p, self.retrieve, "Governing Law", "which law governs", k=2)
        self.assertEqual((ex.attempts, ex.corrected, ex.found), (1, False, True))

    def test_not_found_rechecked_with_wider_k_unless_disabled(self):
        nf = json.dumps({"found": False, "quotes": []})
        p = ScriptedProvider([nf, nf])
        ex = extract(p, self.retrieve, "Non-Compete", "restrictions", k=1, k_retry=4)
        self.assertEqual((ex.attempts, ex.found, ex.k_used), (2, False, 4))
        p2 = ScriptedProvider([nf])
        ex2 = extract(p2, self.retrieve, "Non-Compete", "restrictions", k=1, retry_on_not_found=False)
        self.assertEqual(ex2.attempts, 1)

    def test_found_with_only_fake_quotes_is_reported_not_found(self):
        bad = json.dumps({"found": True, "quotes": [{"excerpt": 0, "text": "invented clause text here"}]})
        ex = extract(ScriptedProvider([bad, bad]), self.retrieve, "Cap On Liability", "cap", k=1)
        self.assertFalse(ex.found)
        self.assertEqual(ex.quotes, [])


if __name__ == "__main__":
    unittest.main()
