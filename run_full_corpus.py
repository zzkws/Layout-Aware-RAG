"""全量语料编排：把 TXC datasheet 目录下全部 PDF 跑完整条管线。

特性：
- 断点续传：每阶段按产物存在性跳过已完成文档，可反复重跑
- 配额保护：Gemini 持续失败（429/quota/RESOURCE_EXHAUSTED）时优雅停止
  VLM 阶段，已完成的文档照常进索引；之后重跑本脚本即从断点继续
- 日志协议（供监控 grep）：[stage] / [vlm-ok] / [QUOTA-STOP] / [FATAL] / [ALL-DONE]

用法:
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
VLM_WORKERS = 5   # Gemini 请求并发度（文档级并发，每文档内部仍按页串行）


def log(msg):
    print(msg, flush=True)


def main():
    all_docs = sorted(p.stem for p in config.PDF_SOURCE_DIR.glob("*.pdf"))
    log(f"[stage] corpus: {len(all_docs)} PDFs")

    # ① render（按 pages.json 跳过）
    todo = [d for d in all_docs
            if not (config.PAGES_DIR / d / "pages.json").exists()]
    log(f"[stage] render start: {len(todo)} to do, "
        f"{len(all_docs) - len(todo)} skipped")
    for d in todo:
        render_doc(d)
    log("[stage] render done")

    # ② layout（按 layout/{doc}.json 跳过；模型只加载一次）
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

    # ③ extract（对尚无 VLM 产物的文档跑，幂等）
    todo = [d for d in all_docs
            if not (config.DESC_DIR / f"{d}.json").exists()]
    log(f"[stage] extract start: {len(todo)} to do")
    for d in todo:
        extract_elements(d)
    log("[stage] extract done")

    # ④ VLM（按 descriptions/{doc}.json 跳过；文档级并发；配额耗尽即停）
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
            if fail_n >= 3:  # 多文档持续失败，大概率配额/网络
                log(f"[QUOTA-STOP] {fail_n} failures, done={done_n}/{len(todo)}")
                ex.shutdown(wait=False, cancel_futures=True)
                stopped = True
                break
    if not stopped:
        ex.shutdown(wait=True)
    log("[stage] vlm done")

    # ④.5 chunk 成员裁剪图合并（每 chunk 一张，证据分析用）
    from pipeline.merge_crops import merge_chunk_crops
    done_docs = sorted(p.stem for p in config.DESC_DIR.glob("*.json"))
    n_merged = sum(merge_chunk_crops(d) for d in done_docs)
    log(f"[stage] merge_crops done: {n_merged} merged images")

    # ⑤ 索引全部已完成文档
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
