# RAG over a Hindi Document — MegaMindz Internship Screening, Level 1

A small, end-to-end Retrieval-Augmented Generation pipeline over
`Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf` (a 22-page Hindi study guide on A.P.J. Abdul Kalam).

**Flow:** PDF → extract text + tables per page → normalise Devanagari → sentence-aware chunks →
multilingual embeddings → ChromaDB (with page / section / chunk_id metadata) → top-k retrieval →
grounded answer from an LLM → **citation built automatically from the stored metadata**.

| | |
|---|---|
| Embeddings | `intfloat/multilingual-e5-base` (multilingual, 768-d, run locally) |
| Vector DB | ChromaDB (persistent, cosine) |
| LLM used for the submitted results | `qwen2.5:7b` via Ollama (local, no API key). Also supported: Claude, OpenAI |
| Results log | [`results/test_results.md`](results/test_results.md) — the 6 required queries with answers + citations |

## 1. Chunking strategy

**Sentence-aware packing: max 600 characters, ~120 characters of overlap (the last 1–2 sentences).**

- The test questions are factoid ("which year", "who", "where"). Small, focused chunks give sharper
  embeddings than page-sized ones, while whole sentences stay together. 600 characters is roughly
  150–200 tokens; `ingest.py` prints the real token statistics (final index: mean 89, max 192 tokens
  per chunk — far below the model's 512 limit, so nothing is truncated).
- Overlap protects facts that straddle a boundary. It visibly helped: the sentence naming
  "Madras Institute of Technology" is carried into the neighbouring chunk as well.
- Chunks **never cross a page**, so the cited page number is always exact (cost: a sentence that
  spans two pages is split).
- The section heading (detected from font size / bold) is stored as metadata and also prepended to
  the text *for embedding only*, giving the model topical context.
- **Tables: one chunk per row**, linearised with its header (`वर्ष: 1997 | सम्मान: भारत रत्न | …`).
  My first version packed several rows per chunk; inspecting results showed the "Bharat Ratna" row
  was diluted by the other awards. Row-level chunks are self-contained facts and rank much better.
- **Study-question lists are excluded from retrieval by default** (chunks with ≥3 `?` are tagged
  `kind="questions"`; `--include-questions` overrides). Section 19 of the PDF repeats the test
  queries almost word for word but contains no answers, so it ranked highly as a pure distractor.
- Final index: 22 pages → **100 chunks** (49 text, 49 table rows, 2 question-list chunks).

## 2. Why ChromaDB

It runs embedded with a plain `pip install` (no Docker, no server), persists to a local folder,
stores metadata next to each vector and supports metadata filters (`where`), which I use for the
`kind` exclusion and the optional `--page` filter. For one ~22-page document that is the right size
of tool. I would choose Qdrant for a multi-user production service (payload indexing, filtering at
scale, a real server) — not needed here.

## 3. How Hindi text is handled

- **Embeddings:** a multilingual model (`multilingual-e5-base`) puts Hindi and English in one vector
  space, so the English queries 4–6 retrieve the Hindi chunks. The E5 `query:` / `passage:` prefixes
  are applied (the model was trained with them; omitting them hurts retrieval).
- **Normalisation** (`rag/text_utils.py`): Unicode **NFC**; removal of zero-width / soft-hyphen / BOM
  characters; NBSP → space; **Devanagari digits → ASCII** (`१९९७` ≡ `1997`) for documents *and*
  queries; line-break reflow. UTF-8 end to end. `ingest.py` warns when the Devanagari share is low
  or U+FFFD appears (a sign of a legacy-font PDF).
- **Segmentation:** splits on the **danda `।` and double danda `॥`**, plus `?` and `!`; an English
  full stop splits only after ≥2 letters/digits, so `A.P.J.` and `Dr.` do not. Over-long "sentences"
  fall back to word-boundary splits.
- **Table extraction:** PyMuPDF's built-in `table.extract()` scrambles Devanagari conjuncts
  (`वर्ष` came out as `वरष ्`). I read each table cell with normal text extraction clipped to the
  cell instead — this produced clean table text.
- **Doubled vowel signs:** this PDF's text layer sometimes emits a vowel sign twice (`सम्माान`).
  A vowel sign can never repeat in valid Devanagari, so repeated signs are collapsed.
- **BM25 tokenizer** (used by the optional hybrid mode) uses explicit Devanagari Unicode ranges,
  because Python's `\w` splits Hindi words at vowel signs.

## 4. Results

- **Retrieval check** (`python evaluate.py`): for all 6 required queries a chunk containing the
  expected answer string was ranked **#1** with both vector search and hybrid (BM25 + vector)
  search — hit@1 = hit@3 = hit@5 = 6/6, MRR = 1.0. This includes the three English queries over
  Hindi text, so cross-lingual retrieval works.
- **Answers and citations:** see `results/test_results.md`. Every answer carries
  `page · section · chunk_id · type · score`, produced from the vector-DB metadata (the LLM only
  reports *which* numbered passages it used).
- **Caveat on that evaluation — please read:** it is small (6 queries) and *not independent*: the
  expected strings were chosen after reading the PDF, and a hit only checks that the string appears
  in the chunk. With a perfect score it cannot distinguish vector from hybrid search, so I make no
  claim that either is better.

## 5. Limitations (honest)

- **Source-text defects I cannot fix by normalisation.** The PDF's font maps some conjunct glyphs to
  nothing, so some words arrive with missing letters (`स्थित` → `स त`, `दृष्टि` → `दृष`). The
  embedding model tolerates this, but it affects exact-match search and the text shown to the LLM.
  OCR (or another extractor) would be needed to recover them; I did not do this.
- **Generation quality is bounded by a 7B local model.** Retrieval ranked the right chunk first for
  all six queries, but in the submitted run the answers show small generation errors: Q4 correctly
  names the Madras Institute of Technology in Chennai but adds a wrong expansion "(MADRAS IIT)" (the
  document says MIT); Q5 gets the co-author and year right but mis-spells the book title
  ("Agni Ki Udhaan"); Q6 gives the date and the lecture setting but mis-spells the place ("Shilang"
  for Shillong) and omits that he collapsed at the Indian Institute of Management Shillong, a detail
  that is in the top-ranked chunk (chunk 49, page 15). So these are generation faults, not retrieval
  faults. Answer language is chosen in code (Hindi or English per question) because the model
  otherwise answered English questions in Hindi. A stronger LLM (the code supports Claude and
  OpenAI) would likely fix these; I did not tune further because the output is deterministic
  (temperature 0) and each run is slow on my CPU-only laptop (3 to 14 minutes per question).
- **Citations depend on the model's self-report.** The cited chunks are real and come from metadata,
  but the model can under-cite or cite a less informative chunk (in Q6 it cited two table rows rather
  than the narrative chunk that contains the collapse). Falls back to the top-1 chunk if no valid number is returned.
- **No "not found" threshold.** Retrieval always returns top-k; E5 cosine scores sit in a narrow
  band (~0.75–0.87 even for weak matches), so refusing off-topic questions relies on the prompt only.
- **Heuristics:** heading detection (font size/bold), header/footer removal, doubled-vowel collapse
  and the "≥3 question marks" rule are tuned to this document; they may misfire on others.
- **Chunks never cross pages**, and the table-of-contents chunk still appears occasionally as noise.
- Chunk size / overlap / model were chosen by reasoning and by inspecting outputs, **not by a
  systematic ablation** — six queries are too few to separate the options.

**What I would do with more time:** build a larger labelled set (≈50 questions with the gold *page*
rather than a substring); ablate chunk size, overlap and `bge-m3` vs E5; add a cross-encoder
re-ranker (`bge-reranker-v2-m3`) and a calibrated similarity threshold; evaluate answer faithfulness
(LLM-as-judge or span-level checks); try OCR / an alternative extractor against the font defect;
allow cross-page windows with multi-page citations.

## 6. How to run (exact commands)

Tested on Windows with Python 3.13. Needs ~3 GB of disk for packages and the embedding model.

```bash
# 1. environment
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# 2. the supplied PDF must be at data/Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf

# 3. build the index (downloads the ~1 GB embedding model on first run)
python ingest.py                  # also writes results/chunks.txt for inspection
```

Choose **one** LLM (retrieval and citations work without any; with none you get the best passage):

```bash
# A) Local, free, no key — install Ollama from https://ollama.com first
ollama pull qwen2.5:7b            # ~4.7 GB; use qwen2.5:3b on low-RAM machines
set LLM_PROVIDER=ollama           # macOS/Linux: export LLM_PROVIDER=ollama
set OLLAMA_MODEL=qwen2.5:7b

# B) OpenAI                       # pip install openai
set OPENAI_API_KEY=...            # read from the environment only; never stored in the repo

# C) Claude                       # pip install anthropic
set ANTHROPIC_API_KEY=...
```

```bash
python run_queries.py             # the 6 required queries -> results/test_results.md
python ask.py "कलाम को भारत रत्न किस वर्ष प्राप्त हुआ?" --show-chunks
python ask.py "Who co-wrote the autobiography?" --hybrid
python ask.py "..." --page 16     # metadata filter: search one page only
python evaluate.py                # hit@k / MRR, vector vs hybrid
python -m pytest -q               # unit tests (no model download needed)
```

On a CPU-only laptop a 7B model can take minutes per question. If `run_queries.py` reports a timeout,
raise `OLLAMA_TIMEOUT` (seconds, default 1200), pass `-k 3` to send fewer passages to the LLM
(retrieval ranked the answer first in every check), or use `qwen2.5:3b`.

Settings can be overridden with environment variables: `MAX_CHARS`, `OVERLAP_CHARS`, `TOP_K`,
`EMBED_MODEL` (e.g. `BAAI/bge-m3`), `LLM_PROVIDER`, `OLLAMA_MODEL`, `PDF_PATH`. If you change the
chunking settings or the embedding model, re-run `python ingest.py`.

## Layout

```
rag/extract.py     PDF -> headings, paragraphs, tables (PyMuPDF); drops page numbers/running headers
rag/text_utils.py  Devanagari normalisation, danda-aware sentence split, BM25 tokenizer
rag/chunking.py    sentence packing with overlap; table-row chunks; question-list tagging
rag/embedding.py   multilingual-e5 wrapper (query:/passage: prefixes)
rag/store.py       ChromaDB storage, filtered vector search, hybrid BM25+vector (RRF)
rag/generate.py    grounded prompt, LLM providers, robust "SOURCES:" parsing
rag/pipeline.py    retrieve -> generate -> cited answer
ingest.py  ask.py  run_queries.py  evaluate.py     command-line entry points
queries.json       the 6 required queries + expected strings for evaluate.py
tests/test_core.py unit tests
```

*AI assistants were used while building this (the brief permits it). I inspected real outputs
throughout, which is how the table, distractor and prompt issues above were found and fixed.*
# Cross-Lingual-Hindi-RAG-Engine
