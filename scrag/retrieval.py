"""Hybrid retrieval: BM25 + dense vectors, RRF fusion, LLM rerank, parent expansion.

Techniques 2/8 (hybrid keyword+vector), 3/8 (RRF fusion), 4/8 (LLM rerank).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .bm25 import BM25
from .chunking import ChunkedCorpus
from .embeddings import Embedder, cosine
from .llm import LLM


@dataclass
class Retrieved:
    child_id: int
    parent_id: int
    child_text: str
    parent_text: str
    source: str
    rrf_score: float
    route_ranks: dict[str, int] = field(default_factory=dict)
    rerank_score: float | None = None


def rrf_fuse(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: score = Σ 1/(k + rank + 1)."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


class HybridRetriever:
    def __init__(self, corpus: ChunkedCorpus, embedder: Embedder):
        self.corpus = corpus
        self.embedder = embedder
        child_texts = [c.text for c in corpus.children]
        self.bm25 = BM25(child_texts)
        self.child_vecs = embedder.embed(child_texts) if child_texts else []

    def search(self, query: str, top_k: int = 8) -> list[Retrieved]:
        children = self.corpus.children
        if not children:
            return []
        q_vec = self.embedder.embed([query])[0]
        dense_rank = sorted(range(len(children)),
                            key=lambda i: cosine(q_vec, self.child_vecs[i]),
                            reverse=True)
        bm25_rank = self.bm25.rank(query)
        fused = rrf_fuse([dense_rank, bm25_rank])[:top_k]
        out = []
        for idx, score in fused:
            ch = children[idx]
            out.append(Retrieved(
                child_id=ch.child_id, parent_id=ch.parent_id,
                child_text=ch.text,
                parent_text=self.corpus.parents[ch.parent_id],
                source=ch.source, rrf_score=score,
                route_ranks={"dense": dense_rank.index(idx),
                             "bm25": bm25_rank.index(idx)},
            ))
        return out

    def rerank(self, query: str, candidates: list[Retrieved],
               llm: LLM, top_n: int = 4) -> list[Retrieved]:
        """Technique 4/8: ask the LLM to score each candidate's relevance 0-10."""
        scored = []
        for cand in candidates:
            prompt = (
                "TASK: rerank\n"
                f"Question: {query}\n\nPassage [{cand.child_id}]:\n{cand.child_text}\n\n"
                "Rate this passage's relevance to the question as an integer 0-10. "
                "Reply with ONLY the number."
            )
            try:
                score = float(llm.chat("You are a relevance judge.", prompt).strip().split()[0])
            except Exception:
                score = 5.0
            cand.rerank_score = score
            scored.append(cand)
        scored.sort(key=lambda c: (c.rerank_score or 0), reverse=True)
        return scored[:top_n]
