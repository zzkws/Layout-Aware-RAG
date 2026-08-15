"""Stage 6: 双路索引构建。

dense 路: config.EMBED_MODEL（默认 sentence-transformers 后端；fastembed 为备选）
          embed_text = toc_path + section_title + description + native_text
          （按此顺序拼接后截断至 1500 字符：先"在哪/是什么"，后正文）
sparse 路: 自研 BM25 + 型号感知分词器（common/tokenize.py）
          fts_text = section_title + toc_path + native_text + description
                     + keywords + 文档卡词

产物: index/chunks.jsonl（chunk 元数据）、index/dense.npy、index/bm25.json

用法:
    python pipeline/index_build.py [--docs ...]
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from common.tokenize import tokenize


def build_chunks(doc_ids) -> list[dict]:
    chunks = []
    for doc_id in doc_ids:
        blocks = json.loads(
            (config.BLOCKS_DIR / f"{doc_id}.json").read_text(encoding="utf-8"))
        desc_file = config.DESC_DIR / f"{doc_id}.json"
        descs = (json.loads(desc_file.read_text(encoding="utf-8"))
                 if desc_file.exists() else {"doc_card": {}, "blocks": {}})
        card = descs["doc_card"]
        card_words = " ".join(str(v) for k, v in card.items()
                              if k != "toc")  # toc 是结构化列表，不进词袋

        for b in blocks["blocks"]:
            if not b["indexable"]:
                continue
            d = descs["blocks"].get(b["block_id"], {})
            description = d.get("description", "")
            keywords = d.get("keywords", [])
            native = b.get("native_text", "")
            if not (native or description):
                continue
            toc_path = b.get("toc_path", "")
            # 目录管"在哪"(toc_path)，标题管"是什么"(section_title)，
            # 然后才是语义概述和原文
            embed_text = "\n".join(filter(None, [
                toc_path, b["section_title"], description,
                native]))[:1500].strip()
            fts_text = " ".join(filter(None, [
                b["section_title"], toc_path, native, description,
                " ".join(keywords), card_words]))
            chunks.append({
                "chunk_id": b["block_id"],
                "doc_id": doc_id,
                "page": b["page"],
                "bboxes_pdf": b["bboxes_pdf"],
                "block_type": b["block_type"],
                "section_title": b["section_title"],
                "toc_path": toc_path,
                "native_text": native,
                "description": description,
                "keywords": keywords,
                "doc_card": card,
                "crop_images": b["crop_images"],
                "page_image": b["page_image"],
                "source_pdf": str(config.PDF_SOURCE_DIR / f"{doc_id}.pdf"),
                "embed_text": embed_text,
                "fts_text": fts_text,
            })
    return chunks


def build_dense(chunks) -> np.ndarray:
    from common.embedder import embed_docs
    return embed_docs([c["embed_text"] for c in chunks])


def build_bm25(chunks) -> dict:
    """倒排索引 + 文档长度，BM25 打分在 search 时进行。"""
    postings: dict[str, dict[int, int]] = {}
    doc_lens = []
    for i, c in enumerate(chunks):
        toks = tokenize(c["fts_text"])
        doc_lens.append(len(toks))
        for tok, tf in Counter(toks).items():
            postings.setdefault(tok, {})[i] = tf
    return {
        "n_docs": len(chunks),
        "avg_len": sum(doc_lens) / max(1, len(doc_lens)),
        "doc_lens": doc_lens,
        "postings": {t: list(p.items()) for t, p in postings.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", nargs="*")
    ap.add_argument("--all", action="store_true",
                    help="索引所有已完成 VLM 描述的文档")
    args = ap.parse_args()
    if args.all:
        doc_ids = sorted(p.stem for p in config.DESC_DIR.glob("*.json"))
    else:
        doc_ids = args.docs or config.DEMO_DOCS

    chunks = build_chunks(doc_ids)
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.INDEX_DIR / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    vecs = build_dense(chunks)
    np.save(config.INDEX_DIR / "dense.npy", vecs)

    bm25 = build_bm25(chunks)
    (config.INDEX_DIR / "bm25.json").write_text(
        json.dumps(bm25), encoding="utf-8")

    print(f"[index] {len(chunks)} chunks | dense {vecs.shape} | "
          f"bm25 vocab {len(bm25['postings'])}")


if __name__ == "__main__":
    main()
