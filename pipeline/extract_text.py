"""Stage 3: lossless text recovery from the PDF text layer, per element bbox.

Text is written back into the native_text field of every element in
layout/{doc}.json, where the stage-4 grouping pass (vlm_blocks.py) reads it.

On a born-digital PDF, clipping the text layer is exact. Regions backed by an
embedded image or vector curves (for example the pin table on page 4 of
7n_10pad) yield an empty native_text, and their semantics rest on the VLM
description instead.

Usage:
    python pipeline/extract_text.py [--docs 6u 7m ...]
"""
import argparse
import json
import sys
from pathlib import Path

import fitz

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def _clean(text: str) -> str:
    text = " ".join(text.split())
    # Drop private-use-area bullet glyphs
    return "".join(ch for ch in text if not 0xE000 <= ord(ch) <= 0xF8FF).strip()


def extract_elements(doc_id: str):
    layout_path = config.LAYOUT_DIR / f"{doc_id}.json"
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    pages_meta = json.loads(
        (config.PAGES_DIR / doc_id / "pages.json").read_text(encoding="utf-8"))
    pmeta_by_no = {p["page"]: p for p in pages_meta["pages"]}
    pdf = fitz.open(pages_meta["source_pdf"])

    n_chars = 0
    for lpage in layout["pages"]:
        fpage = pdf[lpage["page"] - 1]
        scale = 72.0 / pmeta_by_no[lpage["page"]]["render_dpi"]
        for e in lpage["elements"]:
            rect = fitz.Rect(*[v * scale for v in e["bbox_px"]])
            e["native_text"] = _clean(fpage.get_text("text", clip=rect, sort=True))
            n_chars += len(e["native_text"])
    pdf.close()
    layout_path.write_text(
        json.dumps(layout, ensure_ascii=False, indent=2), encoding="utf-8")
    n_el = sum(len(p["elements"]) for p in layout["pages"])
    print(f"[extract/elements] {doc_id}: {n_el} elements, {n_chars} chars")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", nargs="*")
    args = ap.parse_args()
    for doc_id in args.docs or config.DEMO_DOCS:
        extract_elements(doc_id)


if __name__ == "__main__":
    main()
