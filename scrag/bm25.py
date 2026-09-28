"""Pure-python BM25 — the keyword route of hybrid retrieval (technique 2/8)."""

from __future__ import annotations

import math
import re


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.2, b: float = 0.75):
        self.docs = [tokenize(d) for d in docs]
        self.n = len(docs)
        self.k1, self.b = k1, b
        self.avgdl = sum(len(d) for d in self.docs) / max(1, self.n)
        self.df: dict[str, int] = {}
        for d in self.docs:
            for term in set(d):
                self.df[term] = self.df.get(term, 0) + 1

    def scores(self, query: str) -> list[float]:
        out = []
        for d in self.docs:
            s, dl = 0.0, len(d)
            for term in tokenize(query):
                if term not in self.df:
                    continue
                f = d.count(term)
                idf = math.log(1 + (self.n - self.df[term] + 0.5) / (self.df[term] + 0.5))
                denom = f + self.k1 * (1 - self.b + self.b * dl / max(1e-9, self.avgdl))
                s += idf * (f * (self.k1 + 1)) / denom
            out.append(s)
        return out

    def rank(self, query: str) -> list[int]:
        scores = self.scores(query)
        return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
