"""Self-Correcting RAG — interactive Streamlit demo (Ollama-powered)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st  # noqa: E402

from scrag import chunking, pipeline, retrieval  # noqa: E402
from scrag.critic import HallucinationCritic  # noqa: E402
from scrag.embeddings import OllamaEmbedder  # noqa: E402
from scrag.llm import OllamaLLM  # noqa: E402

st.set_page_config(page_title="Self-Correcting RAG", page_icon="🛡️", layout="wide")
st.title("🛡️ Self-Correcting RAG")
st.caption("Generate → 3-layer critic → repair loop. Fully local via Ollama.")

with st.sidebar:
    st.header("Settings")
    docs_dir = st.text_input("Docs folder", "examples/sample_docs")
    threshold = st.slider("Confidence threshold", 0.5, 0.95, 0.70, 0.05)
    max_rounds = st.slider("Max correction rounds", 1, 4, 2)
    if st.button("Build index"):
        with st.spinner("Chunking + embedding…"):
            docs = {}
            for fname in sorted(os.listdir(docs_dir)):
                if fname.endswith((".md", ".txt")):
                    with open(os.path.join(docs_dir, fname), encoding="utf-8") as f:
                        docs[fname] = f.read()
            corpus = chunking.chunk_corpus(docs)
            embedder = OllamaEmbedder()
            st.session_state["retriever"] = retrieval.HybridRetriever(corpus, embedder)
            st.session_state["embedder"] = embedder
            st.success(f"{len(corpus.parents)} parents · {len(corpus.children)} children")

question = st.text_input("Ask a question", "What is faithfulness in RAG evaluation?")

if st.button("Ask", type="primary"):
    if "retriever" not in st.session_state:
        st.warning("Build the index first (sidebar).")
        st.stop()
    with st.spinner("Retrieving → generating → critiquing…"):
        embedder = st.session_state["embedder"]
        llm = OllamaLLM()
        critic = HallucinationCritic(embedder, llm, threshold=threshold)
        rag = pipeline.SelfCorrectingRAG(
            st.session_state["retriever"], llm, critic,
            max_corrections=max_rounds)
        result = rag.ask(question)

    st.subheader("Answer")
    st.write(result.answer)
    st.metric("Confidence", f"{result.confidence:.3f}",
              delta=f"{result.correction_iterations} correction round(s)")

    st.subheader("Claim-by-claim critic report")
    for v in result.claim_report:
        icon = "✅" if v.passed else "❌"
        with st.expander(f"{icon} [{v.confidence:.3f}] {v.claim[:80]}…"):
            st.write(v.claim)
            st.json(v.layer_scores)

    st.subheader("Sources")
    for c in result.citations:
        st.markdown(f"**[{c['n']}] {c['source']}** — {c['excerpt']}")
