"""ChromaDB persistence + retrieval (vector, and optional hybrid BM25+vector via RRF)."""
from __future__ import annotations

from dataclasses import dataclass

import chromadb
import numpy as np

from .chunking import Chunk
from .text_utils import tokenize


@dataclass
class Hit:
    chunk_id: int
    text: str
    page: int
    section: str
    kind: str
    score: float          # cosine similarity (0..1, higher is better)


class VectorStore:
    def __init__(self, path: str, collection: str):
        self.client = chromadb.PersistentClient(path=path)
        self.name = collection
        self._cache = None
        self._bm25 = None

    # ---------- write ----------
    def rebuild(self, chunks: list[Chunk], embeddings: np.ndarray, source: str) -> None:
        try:
            self.client.delete_collection(self.name)
        except Exception:
            pass
        col = self.client.create_collection(self.name, metadata={"hnsw:space": "cosine"})
        col.add(
            ids=[str(c.chunk_id) for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings.tolist(),
            metadatas=[
                {"chunk_id": c.chunk_id, "page": c.page, "section": c.section,
                 "kind": c.kind, "source": source}
                for c in chunks
            ],
        )

    # ---------- read ----------
    @property
    def col(self):
        return self.client.get_collection(self.name)

    @staticmethod
    def _hit(doc, meta, score) -> Hit:
        return Hit(int(meta["chunk_id"]), doc, int(meta["page"]), meta["section"],
                   meta["kind"], float(score))

    def search(self, qvec: np.ndarray, k: int, page: int | None = None,
               include_questions: bool = False) -> list[Hit]:
        conds = []
        if not include_questions:
            conds.append({"kind": {"$ne": "questions"}})   # skip answer-less study-question lists
        if page:
            conds.append({"page": page})                    # metadata filtering (bonus)
        where = None if not conds else conds[0] if len(conds) == 1 else {"$and": conds}
        res = self.col.query(
            query_embeddings=[qvec.tolist()], n_results=k, where=where,
            include=["documents", "metadatas", "distances"],
        )
        return [self._hit(d, m, 1 - dist) for d, m, dist in
                zip(res["documents"][0], res["metadatas"][0], res["distances"][0])]

    def _load_all(self):
        if self._cache is None:
            r = self.col.get(include=["documents", "metadatas", "embeddings"])
            order = np.argsort([int(m["chunk_id"]) for m in r["metadatas"]])
            docs = [r["documents"][i] for i in order]
            metas = [r["metadatas"][i] for i in order]
            embs = np.array([r["embeddings"][i] for i in order])
            self._cache = (docs, metas, embs)
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi([tokenize(d) or ["_"] for d in docs])
        return self._cache

    def hybrid_search(self, query: str, qvec: np.ndarray, k: int, pool: int = 20,
                      include_questions: bool = False) -> list[Hit]:
        """Reciprocal-rank fusion of dense and BM25 rankings (BM25 helps with names/numbers)."""
        docs, metas, embs = self._load_all()
        cos = embs @ qvec                                   # vectors are L2-normalised
        bm = self._bm25.get_scores(tokenize(query))
        if not include_questions:                           # mask out question-list chunks
            mask = np.array([m["kind"] == "questions" for m in metas])
            cos = np.where(mask, -1.0, cos)
            bm = np.where(mask, 0.0, bm)
        dense_rank = list(np.argsort(-cos)[:pool])
        sparse_rank = [i for i in np.argsort(-bm)[:pool] if bm[i] > 0]
        fused: dict[int, float] = {}
        for ranking in (dense_rank, sparse_rank):
            for r, i in enumerate(ranking):
                fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (60 + r + 1)
        top = sorted(fused, key=fused.get, reverse=True)[:k]
        return [self._hit(docs[i], metas[i], cos[i]) for i in top]
