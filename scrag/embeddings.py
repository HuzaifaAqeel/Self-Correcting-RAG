"""Embedding backends: Ollama (nomic-embed-text) or a deterministic mock.

The mock is a hash-bag vector — good enough for tests, demos, and CI with no
model downloads. Swap to OllamaEmbedder for real quality.
"""

from __future__ import annotations

import hashlib
import math
import os
from typing import Protocol


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class MockEmbedder:
    """Deterministic hash-bag embeddings. No downloads, no API calls."""

    def __init__(self, dim: int = 128):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for tok in set(re_tokenize(text)):
                vec[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            out.append([x / norm for x in vec])
        return out


def re_tokenize(text: str) -> list[str]:
    import re

    return re.findall(r"[a-z0-9]+", text.lower())


class OllamaEmbedder:
    """Embeddings via a local Ollama daemon (default model: nomic-embed-text)."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
    ):
        self.model = model or os.environ.get("EMBED_MODEL", "nomic-embed-text")
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL",
                                                    "http://localhost:11434")).rstrip("/")

    def embed(self, texts: list[str]) -> list[list[float]]:
        import json
        import urllib.request

        out = []
        for text in texts:
            req = urllib.request.Request(
                f"{self.base_url}/api/embed",
                data=json.dumps({"model": self.model, "input": text}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = json.loads(resp.read().decode())
            out.append(payload["embeddings"][0])
        return out
