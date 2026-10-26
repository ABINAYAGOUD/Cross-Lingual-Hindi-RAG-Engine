"""Glue: query -> retrieve -> generate -> formatted answer with citations."""
from __future__ import annotations

from dataclasses import dataclass

from . import config
from .embedding import Embedder
from .generate import generate
from .store import Hit, VectorStore
from .text_utils import normalize


@dataclass
class Result:
    query: str
    answer: str
    cited: list[Hit]
    retrieved: list[Hit]


class RAG:
    def __init__(self):
        self.embedder = Embedder(config.EMBED_MODEL)
        self.store = VectorStore(config.DB_DIR, config.COLLECTION)

    def retrieve(self, query: str, k: int = config.TOP_K, hybrid: bool = False,
                 page: int | None = None, include_questions: bool = False) -> list[Hit]:
        q = normalize(query)
        qvec = self.embedder.embed_query(q)
        if hybrid and page is None:
            return self.store.hybrid_search(q, qvec, k, include_questions=include_questions)
        return self.store.search(qvec, k, page, include_questions)

    def ask(self, query: str, k: int = config.TOP_K, hybrid: bool = False,
            page: int | None = None, provider: str | None = None,
            include_questions: bool = False) -> Result:
        hits = self.retrieve(query, k, hybrid, page, include_questions)
        answer, cited = generate(query, hits, provider)
        return Result(query, answer, cited, hits)


def format_source(h: Hit) -> str:
    return (f'page: {h.page} · section: "{h.section}" · chunk_id: {h.chunk_id} · '
            f'type: {h.kind} · score: {h.score:.2f}')


def format_result(r: Result) -> str:
    srcs = "\n".join(f"        {format_source(h)}" for h in r.cited)
    return f"Query : {r.query}\nAnswer: {r.answer}\nSource:\n{srcs}\n"
