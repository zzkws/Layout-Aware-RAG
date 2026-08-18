"""Whole-corpus orchestration: run every PDF in the corpus through the pipeline.

Properties:
- Resumable. Each stage skips documents whose artifacts already exist, so the
  script can be re-run as often as needed.
- Quota-aware. Sustained Gemini failures (429 / quota / RESOURCE_EXHAUSTED) stop
  the VLM stage gracefully; documents that did complete are still indexed, and
  re-running the script picks up from that point.
- Stable log protocol, for monitoring by grep:
  [stage] / [vlm-ok] / [QUOTA-STOP] / [FATAL] / [ALL-DONE]

Usage:
    python -X utf8 -u run_full_corpus.py
"""
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config
from pipeline.extract_text import extract_elements
from pipeline.layout import detect_doc, get_model_weights
from pipeline.render import render_doc
from pipeline.vlm_blocks import process_doc

QUOTA_MARKERS = ("429", "quota", "resource_exhausted", "rate limit")
VLM_WORKERS = 5   # Gemini concurrency, at document level; pages stay serial per document


def log(msg):
    print(msg, flush=True)


def main():
    all_docs = sorted(p.stem for p in config.PDF_SOURCE_DIR.glob("*.pdf"))
    log(f"[stage] corpus: {len(all_docs)} PDFs")

    # 1. render (skipped per pages.json)
    todo = [d for d in all_docs
            if not (config.PAGES_DIR / d / "pages.json").exists()]
    log(f"[stage] render start: {len(todo)} to do, "
        f"{len(all_docs) - len(todo)} skipped")
    for d in todo:
        render_doc(d)
    log("[stage] render done")

    # 2. layout (skipped per layout/{doc}.json; the model loads once)
    todo = [d for d in all_docs
            if not (config.LAYOUT_DIR / f"{d}.json").exists()]
    log(f"[stage] layout start: {len(todo)} to do")
    if todo:
        weights = get_model_weights()
        from doclayout_yolo import YOLOv10
        model = YOLOv10(str(weights))
        for i, d in enumerate(todo, 1):
            detect_doc(model, d)
            if i % 10 == 0:
                log(f"[layout] {i}/{len(todo)}")
    log("[stage] layout done")

    # 3. extract (run for documents without VLM artifacts; idempotent)
    todo = [d for d in all_docs
            if not (config.DESC_DIR / f"{d}.json").exists()]
    log(f"[stage] extract start: {len(todo)} to do")
    for d in todo:
        extract_elements(d)
    log("[stage] extract done")

    # 4. VLM (skipped per descriptions/{doc}.json; concurrent per document;
    #    stops as soon as the quota is exhausted)
    todo = [d for d in all_docs
            if not (config.DESC_DIR / f"{d}.json").exists()]
    log(f"[stage] vlm start: {len(todo)} to do, "
        f"{len(all_docs) - len(todo)} already done, "
        f"{VLM_WORKERS} workers")
    from concurrent.futures import ThreadPoolExecutor, as_completed
    ex = ThreadPoolExecutor(max_workers=VLM_WORKERS)
    futs = {ex.submit(process_doc, d): d for d in todo}
    done_n, fail_n, stopped = 0, 0, False
    for fut in as_completed(futs):
        d = futs[fut]
        try:
            fut.result()
            done_n += 1
            log(f"[vlm-ok] {d} ({done_n}/{len(todo)})")
        except Exception as e:
            msg = str(e).lower()
            if any(m in msg for m in QUOTA_MARKERS):
                log(f"[QUOTA-STOP] doc={d} done={done_n}/{len(todo)} err={e}")
                ex.shutdown(wait=False, cancel_futures=True)
                stopped = True
                break
            fail_n += 1
            log(f"[vlm-err] {d}: {e}")
            if fail_n >= 3:  # several documents failing in a row: quota or network
                log(f"[QUOTA-STOP] {fail_n} failures, done={done_n}/{len(todo)}")
                ex.shutdown(wait=False, cancel_futures=True)
                stopped = True
                break
    if not stopped:
        ex.shutdown(wait=True)
    log("[stage] vlm done")

    # 5. merge member crops into one evidence image per chunk
    from pipeline.merge_crops import merge_chunk_crops
    done_docs = sorted(p.stem for p in config.DESC_DIR.glob("*.json"))
    n_merged = sum(merge_chunk_crops(d) for d in done_docs)
    log(f"[stage] merge_crops done: {n_merged} merged images")

    # 6. index every completed document
    log(f"[stage] index start: {len(done_docs)} docs")
    import numpy as np

    from pipeline.index_build import build_bm25, build_chunks, build_dense
    chunks = build_chunks(done_docs)
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.INDEX_DIR / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    np.save(config.INDEX_DIR / "dense.npy", build_dense(chunks))
    (config.INDEX_DIR / "bm25.json").write_text(
        json.dumps(build_bm25(chunks)), encoding="utf-8")
    log(f"[ALL-DONE] docs={len(done_docs)}/{len(all_docs)} "
        f"chunks={len(chunks)}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        log(f"[FATAL] {sys.exc_info()[1]}")
        sys.exit(1)
