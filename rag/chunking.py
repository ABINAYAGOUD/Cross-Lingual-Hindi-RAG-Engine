"""Sentence-aware chunking with overlap; tables become row-preserving chunks.

Chunks never cross a page boundary so the `page` citation is always exact.
"""
from __future__ import annotations

from dataclasses import dataclass

from .extract import Page
from .text_utils import split_sentences

DEFAULT_SECTION = "प्रारंभ / Introduction"


@dataclass
class Chunk:
    chunk_id: int
    text: str
    page: int
    section: str
    kind: str          # "text" | "table" | "questions"

    @property
    def embed_text(self) -> str:
        """Section title is prepended for embedding only (gives the model topical context)."""
        return f"{self.section}: {self.text}" if self.section else self.text


def _hard_split(s: str, max_chars: int) -> list[str]:
    """Fallback for a single over-long 'sentence' (e.g. no danda in the source)."""
    out, cur = [], ""
    for w in s.split(" "):
        if cur and len(cur) + len(w) + 1 > max_chars:
            out.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        out.append(cur)
    return out


def pack_sentences(sentences: list[str], max_chars: int, overlap_chars: int) -> list[str]:
    sents: list[str] = []
    for s in sentences:
        sents.extend(_hard_split(s, max_chars) if len(s) > max_chars else [s])

    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for s in sents:
        if cur and cur_len + len(s) + 1 > max_chars:
            chunks.append(" ".join(cur))
            tail, n = [], 0
            for prev in reversed(cur):              # carry trailing sentences as overlap
                tail.insert(0, prev)
                n += len(prev) + 1
                if n >= overlap_chars:
                    break
            if len(tail) == len(cur):               # avoid repeating the whole chunk
                tail = tail[1:]
            while tail and sum(len(t) + 1 for t in tail) + len(s) > max_chars:
                tail.pop(0)
            cur, cur_len = tail, sum(len(t) + 1 for t in tail)
        cur.append(s)
        cur_len += len(s) + 1
    if cur and (not chunks or " ".join(cur) != chunks[-1]):
        chunks.append(" ".join(cur))
    return chunks


def table_to_chunks(rows: list[list[str]], max_chars: int, per_row: bool = True) -> list[str]:
    """Header-aware linearisation: 'वर्ष: 1997 | सम्मान: भारत रत्न'.

    per_row=True (default): one chunk per table row. Rows are self-contained facts, and packing
    several rows into one chunk dilutes the embedding (a query about one award would be
    matched against five). Rows are never split.
    """
    if len(rows) < 2:
        return [" | ".join(c for c in rows[0] if c)] if rows else []
    header, body = rows[0], rows[1:]
    lines = []
    for r in body:
        cells = [f"{h}: {c}" if h else c for h, c in zip(header, r) if c]
        if cells:
            lines.append(" | ".join(cells))
    if per_row:
        return lines
    chunks, cur = [], ""
    for ln in lines:
        if cur and len(cur) + len(ln) + 1 > max_chars:
            chunks.append(cur)
            cur = ln
        else:
            cur = f"{cur}\n{ln}" if cur else ln
    if cur:
        chunks.append(cur)
    return chunks


def is_question_list(text: str) -> bool:
    """Study-question lists ('1. ...? 2. ...?') match user queries almost verbatim but carry no
    answers, so they are tagged and excluded from retrieval by default."""
    return text.count("?") >= 3


def build_chunks(pages: list[Page], max_chars: int = 600, overlap_chars: int = 120) -> list[Chunk]:
    chunks: list[Chunk] = []
    section = DEFAULT_SECTION

    def add(text: str, page: int, kind: str):
        chunks.append(Chunk(len(chunks), text, page, section, kind))

    for page in pages:
        buf: list[str] = []

        def flush():
            for t in pack_sentences(buf, max_chars, overlap_chars):
                add(t, page.number, "questions" if is_question_list(t) else "text")
            buf.clear()

        for item in page.items:
            if item.kind == "heading":
                flush()
                section = item.text
            elif item.kind == "para":
                buf.extend(split_sentences(item.text))
            else:  # table
                flush()
                for t in table_to_chunks(item.rows, max_chars):
                    add(t, page.number, "table")
        flush()
    return chunks
