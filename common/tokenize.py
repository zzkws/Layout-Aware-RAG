"""Part-number-aware tokenizer, used by the BM25 / FTS path.

A standard tokenizer splits 7M-26.000MAAJ-T into 7/M/26/000/MAAJ/T, which
destroys part-number search -- the dominant query type for this corpus.

Part-like tokens (long, contain a digit, contain separators) are indexed
through three channels:
  1. the intact token          -- exact match
  2. alphanumeric subfields    -- partial recall, user remembers part of it
  3. character 3-grams         -- substring match, e.g. 26.000M alone
Ordinary words take the normal lowercase path.
"""
import re

_WORD_RE = re.compile(r"[a-z0-9]+")
_PARTLIKE_RE = re.compile(r"^(?=.*\d)[a-z0-9][a-z0-9.\-_/±]{4,}$")


def _char_ngrams(token: str, n: int = 3):
    s = token.replace("-", "").replace("_", "")
    return {s[i:i + n] for i in range(len(s) - n + 1)} if len(s) >= n else set()


def tokenize(text: str) -> list[str]:
    tokens = []
    for raw in text.lower().split():
        raw = raw.strip(".,;:()[]{}\"'")
        if not raw:
            continue
        if _PARTLIKE_RE.match(raw):
            tokens.append(raw)                       # intact part number
            tokens.extend(_WORD_RE.findall(raw))     # subfields
            tokens.extend(_char_ngrams(raw))         # 3-grams
        else:
            tokens.extend(_WORD_RE.findall(raw))
    return tokens
