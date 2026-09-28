"""LLM backends: Ollama chat or a scripted mock.

The pipeline talks to the `LLM` protocol only. `ScriptedLLM` routes on a
`TASK:` marker at the start of the user prompt, which makes demos and tests
deterministic without any model.
"""

from __future__ import annotations

import os
from typing import Callable, Protocol


class LLM(Protocol):
    def chat(self, system: str, user: str) -> str: ...


class OllamaLLM:
    """Chat via a local Ollama daemon (default model: llama3.2)."""

    def __init__(self, model: str | None = None, base_url: str | None = None):
        self.model = model or os.environ.get("TEXT_MODEL", "llama3.2")
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL",
                                                    "http://localhost:11434")).rstrip("/")

    def chat(self, system: str, user: str) -> str:
        import json
        import urllib.request

        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps({
                "model": self.model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            payload = json.loads(resp.read().decode())
        return payload["message"]["content"]


class ScriptedLLM:
    """Deterministic stand-in: dispatches on the TASK: marker in the prompt."""

    def __init__(self, responder: Callable[[str, str, str], str]):
        self.responder = responder

    def chat(self, system: str, user: str) -> str:
        task = "generate"
        for line in user.splitlines():
            if line.startswith("TASK:"):
                task = line.split("TASK:", 1)[1].strip()
                break
        return self.responder(task, system, user)
