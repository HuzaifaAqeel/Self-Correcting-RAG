"""Parent-child chunking.

Technique 1/8: documents are split into small *child* chunks for precise
retrieval, grouped under larger *parent* chunks that are returned as answer
context. Precision from the children, context from the parents.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ChildChunk:
    text: str
    child_id: int
    parent_id: int
    source: str


@dataclass
class ChunkedCorpus:
    parents: list[str]                     # parent_id -> text
    parent_sources: list[str]               # parent_id -> doc name
    children: list[ChildChunk]
    children_of_parent: dict[int, list[int]] = field(default_factory=dict)


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(])")


def split_sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def chunk_text(
    text: str,
    source: str,
    parent_size: int = 900,
    child_size: int = 220,
    overlap: int = 40,
) -> ChunkedCorpus:
    """Split one document into parent/child chunks (character budgets)."""
    sentences = split_sentences(text)
    corpus = ChunkedCorpus(parents=[], parent_sources=[], children=[])

    child_id = 0

    def add_parent(sent_group: list[str]) -> None:
        nonlocal child_id
        parent_text = " ".join(sent_group).strip()
        if not parent_text:
            return
        pid = len(corpus.parents)
        corpus.parents.append(parent_text)
        corpus.parent_sources.append(source)
        # subdivide the parent into overlapping child chunks
        child_ids: list[int] = []
        buf, buf_len = [], 0
        for sent in sent_group:
            buf.append(sent)
            buf_len += len(sent) + 1
            if buf_len >= child_size:
                corpus.children.append(ChildChunk(
                    text=" ".join(buf).strip(), child_id=child_id,
                    parent_id=pid, source=source))
                child_ids.append(child_id)
                child_id += 1
                # overlap: keep trailing sentences up to `overlap` chars
                keep, keep_len = [], 0
                for s in reversed(buf):
                    keep.insert(0, s)
                    keep_len += len(s) + 1
                    if keep_len >= overlap:
                        break
                buf, buf_len = keep, keep_len
        if buf and (not child_ids or " ".join(buf).strip() != corpus.children[-1].text):
            corpus.children.append(ChildChunk(
                text=" ".join(buf).strip(), child_id=child_id,
                parent_id=pid, source=source))
            child_ids.append(child_id)
            child_id += 1
        corpus.children_of_parent[pid] = child_ids

    # group sentences into parents
    group, group_len = [], 0
    for sent in sentences:
        group.append(sent)
        group_len += len(sent) + 1
        if group_len >= parent_size:
            add_parent(group)
            group, group_len = [], 0
    if group:
        add_parent(group)
    return corpus


def chunk_corpus(docs: dict[str, str], **kwargs) -> ChunkedCorpus:
    """Chunk a {doc_name: text} mapping into one merged corpus."""
    merged = ChunkedCorpus(parents=[], parent_sources=[], children=[])
    for name, text in docs.items():
        part = chunk_text(text, source=name, **kwargs)
        pid_off = len(merged.parents)
        cid_off = len(merged.children)
        merged.parents.extend(part.parents)
        merged.parent_sources.extend(part.parent_sources)
        for ch in part.children:
            merged.children.append(ChildChunk(
                text=ch.text, child_id=ch.child_id + cid_off,
                parent_id=ch.parent_id + pid_off, source=ch.source))
        for pid, cids in part.children_of_parent.items():
            merged.children_of_parent[pid + pid_off] = [c + cid_off for c in cids]
    return merged
