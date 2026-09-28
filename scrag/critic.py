"""The 3-layer hallucination critic (techniques 5-7/8).

Every generated answer is split into atomic claims; each claim is checked by
three independent layers and given a confidence score in [0, 1]:

  Layer 1 — embedding similarity: is the claim close in vector space to any
             retrieved source? (catches off-topic fabrication)
  Layer 2 — NLI entailment: does the best source *support* the claim, stay
             *neutral*, or *contradict* it? (LLM judge; catches subtle twists)
  Layer 3 — keyword/entity overlap: do the claim's numbers, dates, and named
             entities actually appear in the sources? (catches invented facts)

confidence = 0.35 * L1 + 0.40 * L2 + 0.25 * L3
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .chunking import split_sentences
from .embeddings import Embedder, cosine
from .llm import LLM
from .retrieval import Retrieved


@dataclass
class ClaimVerdict:
    claim: str
    layer_scores: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    passed: bool = False
    evidence_id: int | None = None  # child_id of best supporting chunk


_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?%?\b")
_ENTITY_RE = re.compile(r"\b[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+){0,2}\b")
_CITATION_RE = re.compile(r"\[\d+\]")


def extract_checkables(claim: str) -> set[str]:
    """Numbers, dates, and capitalized entities a claim asserts."""
    items = set(_NUMBER_RE.findall(claim))
    items |= {e.strip() for e in _ENTITY_RE.findall(claim)}
    return {i.lower() for i in items if i.strip()}


class HallucinationCritic:
    def __init__(self, embedder: Embedder, llm: LLM, threshold: float = 0.70):
        self.embedder = embedder
        self.llm = llm
        self.threshold = threshold

    def split_claims(self, answer: str) -> list[str]:
        claims = []
        for sent in split_sentences(_CITATION_RE.sub("", answer)):
            sent = sent.strip()
            if len(sent.split()) >= 4:
                claims.append(sent)
        return claims

    def check(self, answer: str, sources: list[Retrieved]) -> list[ClaimVerdict]:
        claims = self.split_claims(answer)
        if not claims:
            return []
        claim_vecs = self.embedder.embed(claims)
        source_vecs = self.embedder.embed([s.child_text for s in sources])
        source_texts = [s.child_text.lower() for s in sources]
        verdicts = []
        for claim, c_vec in zip(claims, claim_vecs):
            sims = [cosine(c_vec, s_vec) for s_vec in source_vecs]
            best = max(range(len(sims)), key=lambda i: sims[i]) if sims else 0
            l1 = min(1.0, max(0.0, (sims[best] if sims else 0.0)))

            l2 = self._nli(claim, sources[best].parent_text if sources else "")

            checkables = extract_checkables(claim)
            joined = " ".join(source_texts)
            hits = sum(1 for item in checkables if item in joined)
            l3 = (hits / len(checkables)) if checkables else 1.0

            conf = 0.35 * l1 + 0.40 * l2 + 0.25 * l3
            verdicts.append(ClaimVerdict(
                claim=claim,
                layer_scores={"embedding": round(l1, 3), "nli": round(l2, 3),
                              "keyword": round(l3, 3)},
                confidence=round(conf, 3),
                passed=conf >= self.threshold,
                evidence_id=sources[best].child_id if sources else None,
            ))
        return verdicts

    def _nli(self, claim: str, source: str) -> float:
        """LLM as NLI judge: entailment / neutral / contradiction."""
        prompt = (
            "TASK: nli\n"
            f"SOURCE: {source}\n\nCLAIM: {claim}\n\n"
            "Does the SOURCE support the CLAIM? Reply with exactly one word: "
            "ENTAIL, NEUTRAL, or CONTRADICT."
        )
        try:
            verdict = self.llm.chat("You are a strict NLI judge.", prompt).strip().upper()
        except Exception:
            return 0.5
        if "CONTRADICT" in verdict:
            return 0.0
        if "ENTAIL" in verdict:
            return 1.0
        return 0.5
