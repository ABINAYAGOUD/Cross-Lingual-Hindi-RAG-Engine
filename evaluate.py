"""Retrieval evaluation (bonus): is a chunk containing the expected answer string in the top-k?
Compares pure vector search vs hybrid (BM25 + vector). Edit `expect_any` in queries.json to
match the real wording of YOUR PDF after reading results/chunks.txt."""
import json

from rag.pipeline import RAG
from rag.text_utils import normalize

KS = (1, 3, 5)


def first_hit_rank(hits, needles):
    for i, h in enumerate(hits, 1):
        if any(normalize(n).lower() in h.text.lower() for n in needles):
            return i
    return None


def main():
    queries = json.load(open("queries.json", encoding="utf-8"))
    rag = RAG()
    for hybrid in (False, True):
        ranks = []
        print(f"\n=== {'hybrid' if hybrid else 'vector'} ===")
        for q in queries:
            r = first_hit_rank(rag.retrieve(q["query"], 10, hybrid), q["expect_any"])
            ranks.append(r)
            print(f"  rank={r!s:>4}  [{q['lang']}] {q['query']}")
        for k in KS:
            print(f"  hit@{k}: {sum(r is not None and r <= k for r in ranks)}/{len(ranks)}")
        mrr = sum(1 / r for r in ranks if r) / len(ranks)
        print(f"  MRR: {mrr:.3f}")


if __name__ == "__main__":
    main()
