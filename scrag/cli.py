"""Command-line interface: ingest documents, ask questions.

    python -m scrag.cli ingest examples/sample_docs/
    python -m scrag.cli ask "What is faithfulness in RAG evaluation?"
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scrag import chunking, pipeline, retrieval  # noqa: E402
from scrag.critic import HallucinationCritic  # noqa: E402
from scrag.embeddings import MockEmbedder, OllamaEmbedder  # noqa: E402
from scrag.llm import OllamaLLM  # noqa: E402

INDEX_PATH = os.environ.get("SCRAG_INDEX", ".scrag_index.json")


def build_backends(use_mock: bool):
    if use_mock or not os.environ.get("OLLAMA_BASE_URL"):
        return MockEmbedder(), None
    return OllamaEmbedder(), OllamaLLM()


def cmd_ingest(args):
    docs = {}
    for fname in sorted(os.listdir(args.docs)):
        if fname.endswith((".md", ".txt")):
            with open(os.path.join(args.docs, fname), encoding="utf-8") as f:
                docs[fname] = f.read()
    corpus = chunking.chunk_corpus(docs)
    index = {
        "docs": docs,
        "parents": corpus.parents,
        "parent_sources": corpus.parent_sources,
        "children": [{"text": c.text, "child_id": c.child_id,
                      "parent_id": c.parent_id, "source": c.source}
                     for c in corpus.children],
    }
    with open(args.index, "w", encoding="utf-8") as f:
        json.dump(index, f)
    print(f"indexed {len(docs)} docs → {len(corpus.parents)} parents, "
          f"{len(corpus.children)} children → {args.index}")


def load_retriever(index_path: str, embedder):
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)
    corpus = chunking.ChunkedCorpus(
        parents=index["parents"], parent_sources=index["parent_sources"],
        children=[chunking.ChildChunk(**c) for c in index["children"]])
    return retrieval.HybridRetriever(corpus, embedder)


def cmd_ask(args):
    embedder, llm = build_backends(args.mock)
    if llm is None:
        print("error: no OLLAMA_BASE_URL set — rerun with --mock for the offline backend",
              file=sys.stderr)
        sys.exit(2)
    retriever = load_retriever(args.index, embedder)
    critic = HallucinationCritic(embedder, llm)
    rag = pipeline.SelfCorrectingRAG(retriever, llm, critic)
    result = rag.ask(args.question)
    print(f"\nQ: {args.question}\n")
    print(f"A: {result.answer}\n")
    print(f"confidence: {result.confidence} "
          f"(after {result.correction_iterations} correction round(s))")
    for v in result.claim_report:
        mark = "✓" if v.passed else "✗"
        print(f"  {mark} [{v.confidence}] {v.claim}")
    print("\nsources:")
    for c in result.citations:
        print(f"  [{c['n']}] {c['source']} — {c['excerpt']}")


def main():
    parser = argparse.ArgumentParser(prog="scrag")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_ingest = sub.add_parser("ingest")
    p_ingest.add_argument("docs")
    p_ingest.add_argument("--index", default=INDEX_PATH)
    p_ingest.set_defaults(func=cmd_ingest)
    p_ask = sub.add_parser("ask")
    p_ask.add_argument("question")
    p_ask.add_argument("--index", default=INDEX_PATH)
    p_ask.add_argument("--mock", action="store_true")
    p_ask.set_defaults(func=cmd_ask)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
