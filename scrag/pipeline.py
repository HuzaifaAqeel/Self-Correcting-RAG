"""The self-correcting pipeline (technique 8/8).

    retrieve → LLM rerank → generate → critic → (refine → critic)* → answer

The critic's per-claim verdicts gate a repair loop: while any claim scores
below the confidence threshold (and iterations remain), a refiner rewrites the
answer dropping or fixing the failed claims, and the critic re-checks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .critic import ClaimVerdict, HallucinationCritic
from .llm import LLM
from .retrieval import HybridRetriever, Retrieved


@dataclass
class PipelineResult:
    answer: str
    confidence: float
    claim_report: list[ClaimVerdict] = field(default_factory=list)
    correction_iterations: int = 0
    citations: list[dict] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)


GENERATE_SYSTEM = (
    "You answer questions using ONLY the provided sources. "
    "Every factual statement must carry a citation like [1]. "
    "If the sources don't contain the answer, say so — never invent facts."
)

REFINE_SYSTEM = (
    "You repair answers. Keep only claims supported by the sources; "
    "drop or rewrite anything the critic flagged. Preserve [n] citations."
)


class SelfCorrectingRAG:
    def __init__(
        self,
        retriever: HybridRetriever,
        llm: LLM,
        critic: HallucinationCritic,
        max_corrections: int = 2,
        retrieve_k: int = 8,
        rerank_n: int = 4,
    ):
        self.retriever = retriever
        self.llm = llm
        self.critic = critic
        self.max_corrections = max_corrections
        self.retrieve_k = retrieve_k
        self.rerank_n = rerank_n

    def ask(self, question: str) -> PipelineResult:
        timings: dict[str, float] = {}
        t0 = time.perf_counter()

        candidates = self.retriever.search(question, top_k=self.retrieve_k)
        timings["retrieval_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        sources = self.retriever.rerank(question, candidates, self.llm, top_n=self.rerank_n)
        timings["rerank_ms"] = (time.perf_counter() - t0) * 1000

        context = "\n\n".join(f"[{i + 1}] {s.parent_text}" for i, s in enumerate(sources))

        t0 = time.perf_counter()
        answer = self._generate(question, context)
        timings["generate_ms"] = (time.perf_counter() - t0) * 1000

        iterations = 0
        t0 = time.perf_counter()
        verdicts = self.critic.check(answer, sources)
        while (
            iterations < self.max_corrections
            and verdicts
            and any(not v.passed for v in verdicts)
        ):
            iterations += 1
            answer = self._refine(question, context, answer, verdicts)
            verdicts = self.critic.check(answer, sources)
        timings["critique_ms"] = (time.perf_counter() - t0) * 1000

        confidence = (
            round(sum(v.confidence for v in verdicts) / len(verdicts), 3) if verdicts else 0.0
        )
        citations = [
            {"n": i + 1, "source": s.source, "parent_id": s.parent_id,
             "excerpt": s.parent_text[:160] + "…"}
            for i, s in enumerate(sources)
        ]
        return PipelineResult(
            answer=answer, confidence=confidence, claim_report=verdicts,
            correction_iterations=iterations, citations=citations, timings_ms=timings,
        )

    def _generate(self, question: str, context: str) -> str:
        return self.llm.chat(
            GENERATE_SYSTEM,
            f"TASK: generate\nQuestion: {question}\n\nSources:\n{context}\n\n"
            "Write a concise answer with [n] citations.",
        )

    def _refine(self, question: str, context: str, answer: str,
                verdicts: list[ClaimVerdict]) -> str:
        failed = "\n".join(
            f"- {v.claim} (confidence {v.confidence}: {v.layer_scores})"
            for v in verdicts if not v.passed
        )
        return self.llm.chat(
            REFINE_SYSTEM,
            f"TASK: refine\nQuestion: {question}\n\nSources:\n{context}\n\n"
            f"Draft answer:\n{answer}\n\nFailed claims:\n{failed}\n\n"
            "Rewrite the answer keeping only source-supported claims.",
        )
