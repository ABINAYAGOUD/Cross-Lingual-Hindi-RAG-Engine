"""PDF -> structured items per page: headings, paragraphs, and tables (PyMuPDF)."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import pymupdf as fitz  # PyMuPDF

from .text_utils import normalize

_PAGE_NUM = re.compile(r"^(पृष्ठ|page)?\s*[-–]?\s*\d{1,4}\s*[-–]?$", re.I)


@dataclass
class Item:
    kind: str                      # "heading" | "para" | "table"
    text: str = ""                 # for heading / para
    rows: list = field(default_factory=list)   # for table: list[list[str]]
    edge: bool = False             # sits in the top/bottom page margin (header/footer candidate)


@dataclass
class Page:
    number: int                    # 1-based, matches the PDF viewer
    items: list[Item]


def _body_font_size(doc) -> float:
    sizes: Counter = Counter()
    for page in doc:
        for b in page.get_text("dict")["blocks"]:
            for line in b.get("lines", []):
                for s in line["spans"]:
                    sizes[round(s["size"], 1)] += len(s["text"].strip())
    return sizes.most_common(1)[0][0] if sizes else 11.0


def _block_info(block) -> tuple[str, float, float]:
    parts, max_size, bold, total = [], 0.0, 0, 0
    for line in block["lines"]:
        parts.append("".join(s["text"] for s in line["spans"]).strip())
        for s in line["spans"]:
            n = len(s["text"].strip())
            total += n
            max_size = max(max_size, s["size"])
            if s["flags"] & 16:
                bold += n
    return " ".join(p for p in parts if p), max_size, (bold / total if total else 0.0)


def _table_rows(page, table) -> list[list[str]]:
    """Read each cell with normal text extraction (clipped to the cell). PyMuPDF's own
    table.extract() reorders Devanagari conjuncts/matras (e.g. 'वर्ष' -> 'वरष ्')."""
    try:
        raw = []
        for row in table.rows:
            raw.append([
                page.get_text("text", clip=fitz.Rect(c) + (1, 1, -1, -1)) if c else ""
                for c in row.cells
            ])
        if any(any(c.strip() for c in r) for r in raw):
            return _clean_table(raw)
    except Exception:
        pass
    return _clean_table(table.extract())


def _clean_table(raw) -> list[list[str]]:
    rows = []
    for r in raw:
        cells = [normalize(c or "") for c in r]
        if any(cells):
            rows.append(cells)
    return rows


def extract_pages(pdf_path: str) -> list[Page]:
    doc = fitz.open(pdf_path)
    body = _body_font_size(doc)
    pages: list[Page] = []

    for pno, page in enumerate(doc, start=1):
        try:
            tables = page.find_tables().tables
        except Exception:        # old PyMuPDF / odd page: fall back to text only
            tables = []
        table_rects = [fitz.Rect(t.bbox) for t in tables]

        placed: list[tuple[float, Item]] = []
        for b in page.get_text("dict", sort=True)["blocks"]:
            if b["type"] != 0 or not b.get("lines"):
                continue
            cx, cy = (b["bbox"][0] + b["bbox"][2]) / 2, (b["bbox"][1] + b["bbox"][3]) / 2
            if any(r.contains(fitz.Point(cx, cy)) for r in table_rects):
                continue         # this text belongs to a table; handled separately
            text, size, bold_ratio = _block_info(b)
            text = normalize(text)
            if not text or _PAGE_NUM.match(text):
                continue
            is_heading = (
                len(text) <= 100
                and not text.endswith(("।", "॥"))
                and (size >= body * 1.15 or bold_ratio > 0.8)
            )
            h = page.rect.height
            edge = b["bbox"][3] < 0.09 * h or b["bbox"][1] > 0.91 * h
            placed.append((b["bbox"][1], Item("heading" if is_heading else "para", text=text, edge=edge)))

        for t in tables:
            rows = _table_rows(page, t)
            if rows:
                placed.append((t.bbox[1], Item("table", rows=rows)))

        placed.sort(key=lambda x: x[0])
        pages.append(Page(pno, [it for _, it in placed]))

    return _drop_running_headers(pages)


def _drop_running_headers(pages: list[Page]) -> list[Page]:
    """Remove margin lines repeated on many pages (running headers/footers)."""
    if len(pages) < 4:
        return pages
    counts: Counter = Counter()
    for p in pages:
        for t in {re.sub(r"\d+", "", i.text) for i in p.items if i.edge}:
            counts[t] += 1
    limit = max(3, int(0.4 * len(pages)))
    noisy = {t for t, c in counts.items() if c >= limit and len(t) < 120}
    for p in pages:
        p.items = [i for i in p.items if not (i.edge and re.sub(r"\d+", "", i.text) in noisy)]
    return pages
