"""Tests: the critic must catch an injected hallucination and the loop must remove it."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from scrag import chunking, pipeline, retrieval  # noqa: E402
from scrag.critic import HallucinationCritic  # noqa: E402
from scrag.embeddings import MockEmbedder  # noqa: E402
from scrag.llm import ScriptedLLM  # noqa: E402

from examples.offline_demo import DOCS_DIR, responder  # noqa: E402


def make_rag():
    docs = {}
    for fname in sorted(os.listdir(DOCS_DIR)):
        if fname.endswith(".md"):
            with open(os.path.join(DOCS_DIR, fname), encoding="utf-8") as f:
                docs[fname] = f.read()
    corpus = chunking.chunk_corpus(docs)
    embedder = MockEmbedder()
    retriever = retrieval.HybridRetriever(corpus, embedder)
    llm = ScriptedLLM(responder)
    critic = HallucinationCritic(embedder, llm)
    return pipeline.SelfCorrectingRAG(retriever, llm, critic), docs


def test_parent_child_chunking():
    _, docs = make_rag()
    corpus = chunking.chunk_corpus(docs)
    assert len(corpus.parents) > 0 and len(corpus.children) > len(corpus.parents)
    for ch in corpus.children:
        assert ch.parent_id in corpus.children_of_parent
        assert ch.child_id in corpus.children_of_parent[ch.parent_id]
    print("test_parent_child_chunking: OK")


def test_rrf_fusion_prefers_consensus():
    fused = retrieval.rrf_fuse([[0, 1, 2], [1, 0, 2]])
    order = [idx for idx, _ in fused]
    assert order[0] in (0, 1)  # top-2 in both routes beats rank-3 everywhere
    assert order[-1] == 2
    print("test_rrf_fusion_prefers_consensus: OK")


def test_critic_catches_injected_hallucination():
    rag, _ = make_rag()
    draft = (
        "Faithfulness measures whether a generated answer stays grounded in the "
        "retrieved passages [1]. The BLEU metric was invented by Marie Curie in "
        "1903 [2]."
    )
    candidates = rag.retriever.search("faithfulness BLEU", top_k=8)
    verdicts = rag.critic.check(draft, candidates)
    assert len(verdicts) == 2, verdicts
    assert verdicts[0].passed, verdicts[0]
    assert not verdicts[1].passed, verdicts[1]
    assert "marie curie" in verdicts[1].claim.lower()
    assert verdicts[1].layer_scores["nli"] == 0.0  # contradiction caught
    print("test_critic_catches_injected_hallucination: OK")


def test_self_repair_loop_removes_hallucination():
    rag, _ = make_rag()
    result = rag.ask("What is faithfulness, and who invented BLEU?")
    assert "Marie Curie" not in result.answer
    assert result.correction_iterations >= 1
    assert all(v.passed for v in result.claim_report)
    assert result.confidence >= 0.7
    print("test_self_repair_loop_removes_hallucination: OK")


if __name__ == "__main__":
    test_parent_child_chunking()
    test_rrf_fusion_prefers_consensus()
    test_critic_catches_injected_hallucination()
    test_self_repair_loop_removes_hallucination()
    print("\nall tests passed")
