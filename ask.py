"""Ask a question (Hindi or English). Omit the question for an interactive prompt."""
import argparse

from rag import config
from rag.pipeline import RAG, format_result, format_source


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("question", nargs="?")
    ap.add_argument("-k", type=int, default=config.TOP_K)
    ap.add_argument("--hybrid", action="store_true", help="BM25 + vector fusion")
    ap.add_argument("--page", type=int, help="restrict retrieval to one page (metadata filter)")
    ap.add_argument("--include-questions", action="store_true",
                    help="also search study-question lists (excluded by default: no answers in them)")
    ap.add_argument("--show-chunks", action="store_true", help="print all retrieved chunks")
    a = ap.parse_args()

    rag = RAG()
    qs = [a.question] if a.question else iter(lambda: input("\nप्रश्न / question (Ctrl-C to quit): ").strip(), None)
    for q in qs:
        if not q:
            continue
        r = rag.ask(q, a.k, a.hybrid, a.page, include_questions=a.include_questions)
        print(format_result(r))
        if a.show_chunks:
            for h in r.retrieved:
                print(f"  - {format_source(h)}\n    {h.text[:200]}...")
        if a.question:
            break


if __name__ == "__main__":
    main()
