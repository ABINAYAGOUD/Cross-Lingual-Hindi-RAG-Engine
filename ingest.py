"""Build the vector index: extract -> normalise -> chunk -> embed -> store in ChromaDB."""
import argparse
import os
import statistics
import sys

from rag import config
from rag.chunking import build_chunks
from rag.embedding import Embedder
from rag.extract import extract_pages
from rag.store import VectorStore
from rag.text_utils import devanagari_ratio


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdf", default=config.PDF_PATH)
    ap.add_argument("--max-chars", type=int, default=config.MAX_CHARS)
    ap.add_argument("--overlap", type=int, default=config.OVERLAP_CHARS)
    ap.add_argument("--dump", default="results/chunks.txt", help="write all chunks here for inspection")
    a = ap.parse_args()

    if not os.path.exists(a.pdf):
        sys.exit(f"PDF not found: {a.pdf}\nPut the Hindi PDF in data/ (or pass --pdf PATH).")

    pages = extract_pages(a.pdf)
    full = " ".join(i.text for p in pages for i in p.items if i.kind != "table")
    if len(full) < 200:
        sys.exit("Almost no text extracted - the PDF may be scanned. OCR it first (e.g. ocrmypdf -l hin).")
    ratio = devanagari_ratio(full)
    print(f"Pages: {len(pages)} | Devanagari share of letters: {ratio:.0%}")
    if ratio < 0.3 or "\ufffd" in full:
        print("WARNING: low Devanagari share or replacement chars -> the PDF may use a legacy font "
              "(garbled extraction). Inspect results/chunks.txt before trusting retrieval.")

    chunks = build_chunks(pages, a.max_chars, a.overlap)
    print(f"Chunks: {len(chunks)} (tables: {sum(c.kind == 'table' for c in chunks)})")

    os.makedirs(os.path.dirname(a.dump) or ".", exist_ok=True)
    with open(a.dump, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(f"--- chunk {c.chunk_id} | page {c.page} | {c.kind} | section: {c.section}\n{c.text}\n\n")

    emb = Embedder(config.EMBED_MODEL)
    toks = emb.token_lengths([c.embed_text for c in chunks])
    print(f"Tokens/chunk: mean {statistics.mean(toks):.0f}, max {max(toks)} (model limit 512)")

    vecs = emb.embed_passages([c.embed_text for c in chunks])
    VectorStore(config.DB_DIR, config.COLLECTION).rebuild(chunks, vecs, os.path.basename(a.pdf))
    print(f"Stored in ChromaDB at '{config.DB_DIR}' (collection '{config.COLLECTION}'). Done.")


if __name__ == "__main__":
    main()
