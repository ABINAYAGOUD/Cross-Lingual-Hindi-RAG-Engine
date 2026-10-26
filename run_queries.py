"""Run the 6 required test queries and write results/test_results.md (the deliverable log)."""
import argparse
import json
import os
import time

from rag import config
from rag.pipeline import RAG, format_source


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="queries.json")
    ap.add_argument("--out", default="results/test_results.md")
    ap.add_argument("--hybrid", action="store_true")
    ap.add_argument("-k", type=int, default=config.TOP_K)
    a = ap.parse_args()

    queries = json.load(open(a.queries, encoding="utf-8"))
    rag = RAG()
    mode = "hybrid (BM25+vector)" if a.hybrid else "vector"
    lines = [f"# Test query results\n",
             f"- Embedding model: `{config.EMBED_MODEL}`  \n- Retrieval: {mode}, top-{a.k}  \n"
             f"- LLM provider: `{config.llm_provider()}`\n"]
    failed = []
    for n, q in enumerate(queries, 1):
        t0 = time.time()
        try:
            r = rag.ask(q["query"], a.k, a.hybrid)
        except Exception as e:                      # e.g. LLM timeout: keep the other results
            failed.append(n)
            print(f"[{n}] {q['query']}\n    FAILED after {time.time() - t0:.0f}s: {e}\n")
            lines.append(f"## {n}. {q['query']}  _({q['lang']})_\n\n**ERROR:** {e}\n")
            continue
        print(f"[{n}] {q['query']}\n    -> {r.answer}   ({time.time() - t0:.0f}s)\n")
        lines.append(f"## {n}. {q['query']}  _({q['lang']})_\n")
        lines.append(f"**Answer:** {r.answer}\n")
        lines.append("**Source (citation):**\n")
        lines += [f"- {format_source(h)}" for h in r.cited]
        lines.append("\n<details><summary>All retrieved chunks</summary>\n")
        for h in r.retrieved:
            lines.append(f"- `{format_source(h)}`  \n  {h.text[:300].replace(chr(10), ' ')}…")
        lines.append("\n</details>\n")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {a.out}")
    if failed:
        print(f"WARNING: queries {failed} failed. Re-run (try: set OLLAMA_TIMEOUT=2400, or -k 3) "
              "before submitting; the results file contains an ERROR for them.")


if __name__ == "__main__":
    main()
