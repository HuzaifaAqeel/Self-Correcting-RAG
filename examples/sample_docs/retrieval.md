# Retrieval strategies for RAG

Modern RAG systems rarely rely on a single retrieval route. The most common
pattern is hybrid retrieval: a dense vector search for semantic similarity
runs alongside BM25 keyword search, and their rankings are fused with
Reciprocal Rank Fusion (RRF). The RRF formula is score(d) = sum over routes of
1 / (k + rank), with k conventionally set to 60. This fusion was introduced by
Cormack, Clarke, and Buettcher in their 2009 SIGIR paper.

After fusion, a reranker re-scores the top candidates. Cross-encoder rerankers
such as Cohere Rerank 3 or the open-source bge-reranker read the query and
passage together and output a relevance score. This second pass is expensive
but typically lifts recall@5 by 5 to 10 points on the BEIR benchmark suite.

Chunking strategy interacts with retrieval quality. Parent-child chunking
indexes small child chunks of 200 to 300 tokens for precise matching, then
expands the winners to their 800 to 1000 token parent chunks before passing
context to the generator. The 2024 study by Chen and colleagues on the
TREC-RAG track showed parent-child expansion improving answer faithfulness by
12 percent over fixed-size chunking.
