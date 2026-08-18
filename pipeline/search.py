"""Stage 7: dual-path recall (dense + BM25) fused by RRF.

Each result carries native_text, description, chunk crops, and the source
file / page / bbox, so the evidence links straight back to a region of the
original PDF page.

Usage:
    python pipeline/search.py "load capacitance 12pF crystal"
    python pipeline/search.py --json out.json "query1" "query2" ...
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from common.tokenize import tokenize

K1, B = 1.5, 0.75


class Index:
    def __init__(self):
        self.chunks = [json.loads(line) for line in
                       (config.INDEX_DIR / "chunks.jsonl").read_text(
                           encoding="utf-8").splitlines()]
        self.dense = np.load(config.INDEX_DIR / "dense.npy")
        bm = json.loads((config.INDEX_DIR / "bm25.json").read_text(
            encoding="utf-8"))
        self.n_docs = bm["n_docs"]
        self.avg_len = bm["avg_len"]
        self.doc_lens = bm["doc_lens"]
        self.postings = {t: dict((int(i), tf) for i, tf in p)
                         for t, p in bm["postings"].items()}

    def embed_query(self, query: str) -> np.ndarray:
        from common.embedder import embed_query
        return embed_query(query)

    def dense_rank(self, query: str) -> list[int]:
        scores = self.dense @ self.embed_query(query)
        return list(np.argsort(-scores)), scores

    def bm25_rank(self, query: str):
        scores = np.zeros(self.n_docs, dtype=np.float32)
        for tok in set(tokenize(query)):
            plist = self.postings.get(tok)
            if not plist:
                continue
            idf = math.log(1 + (self.n_docs - len(plist) + 0.5)
                           / (len(plist) + 0.5))
            for i, tf in plist.items():
                norm = tf * (K1 + 1) / (
                    tf + K1 * (1 - B + B * self.doc_lens[i] / self.avg_len))
                scores[i] += idf * norm
        return list(np.argsort(-scores)), scores

    def search(self, query: str, top_k: int = config.TOP_K):
        d_rank, d_scores = self.dense_rank(query)
        b_rank, b_scores = self.bm25_rank(query)
        rrf = {}
        for rank_list, tag in ((d_rank, "dense"), (b_rank, "bm25")):
            for r, idx in enumerate(rank_list):
                rrf.setdefault(idx, {"score": 0.0, "dense_rank": None,
                                     "bm25_rank": None})
                rrf[idx]["score"] += 1.0 / (config.RRF_K + r + 1)
                rrf[idx][f"{tag}_rank"] = r + 1
        ranked = sorted(rrf.items(), key=lambda kv: -kv[1]["score"])[:top_k]

        results = []
        for idx, info in ranked:
            c = self.chunks[idx]
            results.append({
                "rrf_score": round(info["score"], 4),
                "dense_rank": info["dense_rank"],
                "bm25_rank": info["bm25_rank"],
                "dense_score": round(float(d_scores[idx]), 4),
                "bm25_score": round(float(b_scores[idx]), 2),
                "chunk_id": c["chunk_id"],
                "block_type": c["block_type"],
                "section_title": c["section_title"],
                "toc_path": c.get("toc_path", ""),
                "doc_id": c["doc_id"],
                "source_pdf": c["source_pdf"],
                "page": c["page"],
                "bboxes_pdf": c["bboxes_pdf"],
                "crop_images": c["crop_images"],
                "description": c["description"],
                "native_text": c["native_text"][:300],
            })
        return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("queries", nargs="+")
    ap.add_argument("--top-k", type=int, default=config.TOP_K)
    ap.add_argument("--json", help="write results to a JSON file, for reporting")
    args = ap.parse_args()

    index = Index()
    all_results = {}
    for q in args.queries:
        results = index.search(q, args.top_k)
        all_results[q] = results
        print(f"\n=== Query: {q}")
        for r in results:
            print(f"  #{r['rrf_score']:.4f} [{r['chunk_id']}] "
                  f"{r['block_type']} | {r['section_title'] or '-'} "
                  f"| p{r['page']} | dense#{r['dense_rank']} bm25#{r['bm25_rank']}")
            print(f"     {r['description'][:110]}")
            print(f"     evidence: {r['source_pdf']} p{r['page']} "
                  f"{len(r['bboxes_pdf'])} bbox(es), "
                  f"{len(r['crop_images'])} crop(s)")

    if args.json:
        Path(args.json).write_text(
            json.dumps(all_results, ensure_ascii=False, indent=2),
            encoding="utf-8")


if __name__ == "__main__":
    main()
