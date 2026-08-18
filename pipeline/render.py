"""Stage 1: PDF rendering and document pre-check.

Each PDF is rendered to 200 dpi page images, with page-level metadata recorded:
- size in PDF points (the authoritative coordinate system) and in pixels
- text-layer character count -> born-digital decision, which determines whether
  stage 3 can read a text layer or the page would need OCR

Usage:
    python pipeline/render.py            # process config.DEMO_DOCS
    python pipeline/render.py --all      # process every PDF in the directory
"""
import argparse
import json
import sys
from pathlib import Path

import fitz  # PyMuPDF

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def render_doc(doc_id: str) -> dict:
    pdf_path = config.PDF_SOURCE_DIR / f"{doc_id}.pdf"
    out_dir = config.PAGES_DIR / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    zoom = config.RENDER_DPI / 72.0
    pages = []
    for page in doc:
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        img_path = out_dir / f"p{page.number + 1}.png"
        pix.save(img_path)
        native_chars = len(page.get_text().strip())
        pages.append({
            "page": page.number + 1,
            "width_pt": round(page.rect.width, 2),
            "height_pt": round(page.rect.height, 2),
            "width_px": pix.width,
            "height_px": pix.height,
            "render_dpi": config.RENDER_DPI,
            "image": img_path.name,
            "native_text_chars": native_chars,
            "text_source": "pdf_layer" if native_chars > 50 else "needs_ocr",
        })
    meta = {
        "doc_id": doc_id,
        "source_pdf": str(pdf_path),
        "page_count": len(pages),
        "pages": pages,
    }
    (out_dir / "pages.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    doc.close()
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="process every PDF in the directory")
    ap.add_argument("--docs", nargs="*", help="explicit list of doc_ids")
    args = ap.parse_args()

    if args.all:
        doc_ids = sorted(p.stem for p in config.PDF_SOURCE_DIR.glob("*.pdf"))
    else:
        doc_ids = args.docs or config.DEMO_DOCS

    for doc_id in doc_ids:
        meta = render_doc(doc_id)
        low_text = sum(1 for p in meta["pages"] if p["text_source"] == "needs_ocr")
        print(f"[render] {doc_id}: {meta['page_count']} pages, "
              f"{low_text} low-text page(s)")


if __name__ == "__main__":
    main()
