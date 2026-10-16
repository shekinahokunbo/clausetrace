"""Grounded clause extraction with verbatim-citation verification.

Flow for one (contract, clause category) request:
  1. retrieve top-k chunks for the category
  2. ask the model for exact quotes, each tagged with the chunk it came from,
     or an explicit NOT FOUND
  3. verify every quote is a verbatim substring of the chunk it cites, and map
     it back to absolute character offsets in the contract
  4. if a quote fails verification, or the model reports NOT FOUND, re-retrieve
     with a wider k and retry once, telling the model which quotes failed
  5. return only verified quotes; unverifiable text never reaches the user

Providers call the REST APIs directly (no SDK dependency).
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol

from .chunking import Chunk

SYSTEM = """You review commercial contracts for a lawyer. You will receive numbered excerpts from ONE contract and a clause category.
Rules:
- Use only the excerpts. Never rely on outside knowledge of the contract.
- If the category is present, copy the relevant text EXACTLY as written (verbatim, no paraphrase, no ellipses) and give the excerpt number it came from.
- If the excerpts do not contain this clause, say it is not found. Not found is a correct, expected answer for many contracts.
Respond with JSON only:
{"found": true|false, "quotes": [{"excerpt": <int>, "text": "<verbatim text>"}], "summary": "<one sentence or empty>"}"""


def build_prompt(category: str, description: str, chunks: list[Chunk], feedback: str = "") -> str:
    parts = [f"Clause category: {category}\nWhat to look for: {description}\n"]
    for c in chunks:
        parts.append(f"<excerpt {c.index}>\n{c.text}\n</excerpt {c.index}>")
    if feedback:
        parts.append(f"\nA previous attempt failed verification: {feedback}\nQuote only text that appears verbatim.")
    return "\n".join(parts)


class Provider(Protocol):
    name: str

    def complete(self, system: str, user: str) -> tuple[str, dict]: ...


def _post(url: str, headers: dict, payload: dict, timeout: int = 120, retries: int = 4) -> dict:
    """POST with exponential backoff on rate limits and transient server errors."""
    import urllib.error
    body = json.dumps(payload).encode()
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 529) or attempt == retries:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


class AnthropicProvider:
    def __init__(self, model: str, api_key: str | None = None, max_tokens: int = 1024):
        self.model, self.max_tokens = model, max_tokens
        self.api_key = api_key or os.environ["ANTHROPIC_API_KEY"]
        self.name = f"anthropic/{model}"

    def complete(self, system, user):
        d = _post("https://api.anthropic.com/v1/messages",
                  {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                  {"model": self.model, "max_tokens": self.max_tokens, "temperature": 0, "system": system,
                   "messages": [{"role": "user", "content": user}]})
        text = "".join(b.get("text", "") for b in d.get("content", []))
        u = d.get("usage", {})
        return text, {"input_tokens": u.get("input_tokens", 0), "output_tokens": u.get("output_tokens", 0)}


class GeminiProvider:
    def __init__(self, model: str, api_key: str | None = None, max_tokens: int = 1024):
        self.model, self.max_tokens = model, max_tokens
        self.api_key = api_key or os.environ["GEMINI_API_KEY"]
        self.name = f"google/{model}"

    def complete(self, system, user):
        d = _post(f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
                  {"x-goog-api-key": self.api_key, "content-type": "application/json"},
                  {"systemInstruction": {"parts": [{"text": system}]},
                   "contents": [{"role": "user", "parts": [{"text": user}]}],
                   "generationConfig": {"temperature": 0, "maxOutputTokens": self.max_tokens,
                                        "responseMimeType": "application/json"}})
        cand = (d.get("candidates") or [{}])[0]
        text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []))
        u = d.get("usageMetadata", {})
        return text, {"input_tokens": u.get("promptTokenCount", 0), "output_tokens": u.get("candidatesTokenCount", 0)}


_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub(" ", s).strip()


def locate(quote: str, chunk: Chunk) -> tuple[int, int] | None:
    """Absolute [start, end) of `quote` inside `chunk`, tolerant only of
    whitespace differences. Anything else (paraphrase, ellipsis, invented
    words) fails."""
    q = _norm(quote)
    if len(q) < 3:
        return None
    # build normalized text with a map back to original offsets
    norm_chars, mapping, prev_space = [], [], False
    for i, ch in enumerate(chunk.text):
        if ch.isspace():
            if prev_space or not norm_chars:
                continue
            norm_chars.append(" ")
            prev_space = True
        else:
            norm_chars.append(ch)
            prev_space = False
        mapping.append(i)
    hay = "".join(norm_chars)
    pos = hay.find(q)
    if pos < 0:
        return None
    return chunk.start + mapping[pos], chunk.start + mapping[pos + len(q) - 1] + 1


def parse_response(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(d, dict) or "found" not in d:
        return None
    d.setdefault("quotes", [])
    return d


@dataclass
class VerifiedQuote:
    excerpt: int
    text: str
    start: int
    end: int


@dataclass
class Extraction:
    found: bool
    quotes: list[VerifiedQuote]
    summary: str
    attempts: int
    corrected: bool                 # second attempt was triggered
    rejected_quotes: int            # quotes dropped by the verifier, all attempts
    total_quotes: int               # quotes proposed by the model, all attempts
    parse_failures: int
    input_tokens: int
    output_tokens: int
    latency_s: float
    k_used: int
    log: list[str] = field(default_factory=list)


def verify(parsed: dict, chunks: list[Chunk]) -> tuple[list[VerifiedQuote], list[str]]:
    by_idx = {c.index: c for c in chunks}
    ok, bad = [], []
    for q in parsed.get("quotes", []):
        try:
            idx, text = int(q["excerpt"]), str(q["text"])
        except (KeyError, TypeError, ValueError):
            bad.append("malformed quote object")
            continue
        loc = None
        if idx in by_idx:
            loc = locate(text, by_idx[idx])
        if loc is None:  # wrong excerpt number but text exists elsewhere in the context: accept, re-attribute
            for c in chunks:
                loc = locate(text, c)
                if loc:
                    idx = c.index
                    break
        if loc is None:
            bad.append(f'"{text[:80]}" is not verbatim in any excerpt')
        else:
            ok.append(VerifiedQuote(idx, text, *loc))
    return ok, bad


def extract(provider: Provider, retrieve, category: str, description: str, k: int = 5, k_retry: int = 10,
            retry_on_not_found: bool = True) -> Extraction:
    """retrieve(k) -> list[Chunk] ranked for this contract and category.

    retry_on_not_found trades cost for recall: a missed clause is the expensive
    error in contract review, so by default NOT FOUND is re-checked with wider k."""
    t0 = time.time()
    tok_in = tok_out = rejected = proposed = parse_fail = 0
    log: list[str] = []
    feedback = ""
    result_quotes: list[VerifiedQuote] = []
    found, summary, attempts, corrected, k_now = False, "", 0, False, k
    for attempt in range(2):
        attempts += 1
        chunks = retrieve(k_now)
        text, usage = provider.complete(SYSTEM, build_prompt(category, description, chunks, feedback))
        tok_in += usage.get("input_tokens", 0)
        tok_out += usage.get("output_tokens", 0)
        parsed = parse_response(text)
        if parsed is None:
            parse_fail += 1
            log.append(f"attempt {attempts}: unparseable response")
            feedback = "your response was not valid JSON in the required format."
        else:
            good, bad = verify(parsed, chunks)
            proposed += len(parsed.get("quotes", []))
            rejected += len(bad)
            found = bool(parsed["found"]) and bool(good)
            summary = str(parsed.get("summary", ""))
            result_quotes = good
            log.append(f"attempt {attempts}: k={k_now} found={parsed['found']} verified={len(good)} rejected={len(bad)}")
            needs_retry = bool(bad) or (retry_on_not_found and not parsed["found"])
            if not needs_retry:
                break
            feedback = "; ".join(bad) if bad else "nothing found in the top excerpts; more excerpts are now included."
        if attempt == 0:
            corrected, k_now = True, k_retry
    return Extraction(found, result_quotes, summary, attempts, corrected, rejected, proposed, parse_fail,
                      tok_in, tok_out, time.time() - t0, k_now, log)
