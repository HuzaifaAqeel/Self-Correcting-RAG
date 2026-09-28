<div align="center">

# 🛡️ Self-Correcting RAG

**A RAG pipeline that checks its own work — then fixes it.**

Generate → 3-layer hallucination critic → confidence-gated repair loop.
Fully local via Ollama. No cloud APIs, no keys.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)

</div>

---

## The 8 techniques

| # | Technique | Where |
|---|-----------|-------|
| 1 | **Parent-child chunking** — small children for precise retrieval, expanded to parents for context | `scrag/chunking.py` |
| 2 | **Hybrid retrieval** — BM25 keyword + dense vectors in parallel | `scrag/bm25.py`, `scrag/retrieval.py` |
| 3 | **RRF fusion** — `Σ 1/(60 + rank + 1)` merges both routes | `scrag/retrieval.py` |
| 4 | **LLM rerank** — cross-check top candidates before generation | `scrag/retrieval.py` |
| 5 | **Critic L1** — embedding similarity of each claim vs sources | `scrag/critic.py` |
| 6 | **Critic L2** — NLI judge: entail / neutral / contradict per claim | `scrag/critic.py` |
| 7 | **Critic L3** — numbers / dates / named entities must appear in sources | `scrag/critic.py` |
| 8 | **Self-repair loop** — claims below the confidence threshold trigger a rewrite + re-check | `scrag/pipeline.py` |

Confidence per claim: `0.35·L1 + 0.40·L2 + 0.25·L3`.

---

## Quick start

```bash
# 1. local models (once)
ollama pull llama3.2 && ollama pull nomic-embed-text

# 2. index your docs
python -m scrag.cli ingest examples/sample_docs/

# 3a. ask from the CLI
python -m scrag.cli ask "What is faithfulness in RAG evaluation?"

# 3b. or launch the Streamlit UI
pip install -r requirements.txt
streamlit run app.py
```

Configure via environment (see `.env.example`):

```bash
OLLAMA_BASE_URL=http://localhost:11434
TEXT_MODEL=llama3.2
EMBED_MODEL=nomic-embed-text
```

### Offline demo (no Ollama, no downloads)

```bash
python examples/offline_demo.py
```

Runs the real chunker, BM25, hash embeddings, RRF, and critic layers 1 & 3,
with a scripted LLM standing in for generation / rerank / NLI / refine — so you
can watch an injected hallucination get caught and removed, deterministically.

---

## How the critic works

Each generated answer is split into atomic claims. Every claim is scored by
three independent layers:

- **L1 embedding** — cosine similarity between the claim and the best source
  chunk. Catches off-topic fabrication.
- **L2 NLI** — the LLM judges whether the parent context *entails*, stays
  *neutral* on, or *contradicts* the claim. Catches subtle twists of real facts.
- **L3 keyword** — the claim's numbers, dates, and capitalized entities are
  checked for literal presence in the sources. Catches invented figures.

Claims below the threshold (default 0.70) are handed to the refiner with their
layer scores, the answer is rewritten keeping only supported claims, and the
critic re-checks — up to 2 rounds.

## Tests

```bash
python tests/test_pipeline.py
```

Covers parent-child mapping, RRF consensus, hallucination injection (the critic
must flag the invented claim with an NLI contradiction), and the full repair
loop (the final answer must drop it and reach confidence ≥ 0.70).

## Project structure

```
scrag/
├── chunking.py     # parent-child chunker
├── bm25.py         # pure-python BM25
├── embeddings.py   # OllamaEmbedder + deterministic MockEmbedder
├── llm.py          # OllamaLLM + ScriptedLLM
├── retrieval.py    # hybrid search + RRF + LLM rerank + parent expansion
├── critic.py       # 3-layer hallucination critic
├── pipeline.py     # generate → critique → repair loop
└── cli.py          # ingest / ask commands
app.py              # Streamlit demo
examples/           # offline_demo.py + sample docs
tests/              # test_pipeline.py
```

## License

MIT — see [LICENSE](LICENSE).
