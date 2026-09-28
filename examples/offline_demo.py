"""Offline demo: the full generate → critique → self-repair loop.

Uses the real chunker, real BM25 + hash embeddings, real RRF, and real critic
layers 1 & 3. The LLM (generator / reranker / NLI judge / refiner) is scripted
so the demo is deterministic; plug in OllamaLLM for the real model.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scrag import chunking, pipeline, retrieval  # noqa: E402
from scrag.critic import HallucinationCritic, extract_checkables  # noqa: E402
from scrag.embeddings import MockEmbedder  # noqa: E402
from scrag.llm import ScriptedLLM  # noqa: E402

DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_docs")

DRAFT_WITH_HALLUCINATION = (
    "Faithfulness measures whether a generated answer stays grounded in the "
    "retrieved passages [1]. The RAGAS framework popularized faithfulness as a "
    "core metric in 2023 [1]. The BLEU metric was invented by Marie Curie in "
    "1903 [2]."
)

REPAIRED_ANSWER = (
    "Faithfulness measures whether a generated answer stays grounded in the "
    "retrieved passages [1]. The RAGAS framework popularized faithfulness as a "
    "core metric in 2023 [1]."
)


def responder(task: str, system: str, user: str) -> str:
    if task == "generate":
        return DRAFT_WITH_HALLUCINATION
    if task == "refine":
        return REPAIRED_ANSWER
    if task == "rerank":
        passage = user.split("Passage", 1)[1] if "Passage" in user else user
        return "9" if "faithfulness" in passage.lower() else "4"
    if task == "nli":
        claim = user.split("CLAIM:", 1)[1] if "CLAIM:" in user else user
        source = user.split("SOURCE:", 1)[1].split("CLAIM:")[0] if "SOURCE:" in user else ""
        if "marie curie" in claim.lower() or "bleu" in claim.lower():
            return "CONTRADICT"
        checkables = extract_checkables(claim)
        if not checkables:
            return "NEUTRAL"
        hits = sum(1 for c in checkables if c in source.lower())
        return "ENTAIL" if hits / len(checkables) >= 0.5 else "NEUTRAL"
    return "?"


def main() -> None:
    docs = {}
    for fname in sorted(os.listdir(DOCS_DIR)):
        if fname.endswith(".md"):
            with open(os.path.join(DOCS_DIR, fname), encoding="utf-8") as f:
                docs[fname] = f.read()

    corpus = chunking.chunk_corpus(docs)
    print("=" * 72)
    print("Self-Correcting RAG — offline demo (scripted LLM, real retrieval+critic)")
    print("=" * 72)
    print(f"\ningested {len(docs)} docs → {len(corpus.parents)} parents, "
          f"{len(corpus.children)} children")

    embedder = MockEmbedder()
    retriever = retrieval.HybridRetriever(corpus, embedder)
    llm = ScriptedLLM(responder)
    critic = HallucinationCritic(embedder, llm, threshold=0.70)
    rag = pipeline.SelfCorrectingRAG(retriever, llm, critic)

    question = "What is faithfulness in RAG evaluation, and who invented BLEU?"
    print(f"\nQ: {question}\n")

    # show the round-1 critic pass explicitly (draft vs final verdicts)
    candidates = retriever.search(question, top_k=8)
    sources = retriever.rerank(question, candidates, llm, top_n=4)
    round1 = critic.check(DRAFT_WITH_HALLUCINATION, sources)

    print("--- draft answer (with an injected hallucination) ---")
    print(DRAFT_WITH_HALLUCINATION)

    print("\n--- critic: round 1 verdicts ---")
    for v in round1:
        mark = "PASS" if v.passed else "FAIL"
        print(f"[{mark}] conf={v.confidence} layers={v.layer_scores}\n"
              f"       {v.claim}")

    result = rag.ask(question)

    print(f"\ncorrection rounds: {result.correction_iterations}")
    print("--- critic: final verdicts ---")
    for v in result.claim_report:
        mark = "PASS" if v.passed else "FAIL"
        print(f"[{mark}] conf={v.confidence} layers={v.layer_scores}\n"
              f"       {v.claim}")

    print(f"\nfinal confidence: {result.confidence}")
    print("\n--- final answer ---")
    print(result.answer)
    print("\n" + "=" * 72)
    hallucination_removed = "Marie Curie" not in result.answer
    print(f"hallucination removed: {hallucination_removed}")
    print("=" * 72)


if __name__ == "__main__":
    main()
