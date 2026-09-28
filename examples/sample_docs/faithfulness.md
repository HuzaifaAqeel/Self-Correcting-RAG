# Faithfulness in RAG evaluation

Faithfulness measures whether the generated answer stays grounded in the
retrieved passages. A claim is faithful if a reader can verify it against the
cited source without needing outside knowledge. The RAGAS framework, released
in 2023 by Shahul Es and collaborators, popularized faithfulness as a core
metric alongside answer relevancy and context precision.

Faithfulness is computed claim by claim. Each sentence of the answer becomes
an atomic claim, and a judge model decides whether the claim is entailed by
the retrieved context. The faithfulness score is the fraction of supported
claims. A score of 1.0 means every claim is grounded; anything below 0.8
usually triggers a regeneration or a human review in production systems.

A related concept is answer correctness, which compares the answer to a
reference gold answer instead of the retrieved context. Faithfulness cares
about grounding, while correctness cares about matching the expected output.
Both are reported in the 2024 RAGAS benchmark study on the WikiEval dataset.
