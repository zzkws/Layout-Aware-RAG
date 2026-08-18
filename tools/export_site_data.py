"""Export a compact, path-safe corpus bundle for the public Sites demos."""
from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from common.tokenize import tokenize
from tools.public_data import public_chunk

DEFAULT_QUERIES = {
    "txc": [
        "7M 26.000MHz load capacitance ESR",
        "VCXO 3.3V phase jitter package",
        "pin connection tri-state output",
    ],
    "tkd": [
        "32.768 kHz crystal load capacitance",
        "temperature compensated oscillator frequency stability",
        "package dimensions land pattern",
    ],
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def bm25_results(query: str, bm25: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    n_docs = int(bm25["n_docs"])
    avg_len = float(bm25["avg_len"] or 1)
    doc_lens = bm25["doc_lens"]
    postings = bm25["postings"]
    scores = [0.0] * n_docs
    for token in set(tokenize(query)):
        entries = postings.get(token) or []
        if not entries:
            continue
        idf = math.log(1 + (n_docs - len(entries) + 0.5) / (len(entries) + 0.5))
        for index, tf in entries:
            norm = tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * doc_lens[index] / avg_len))
            scores[index] += idf * norm
    ranked = sorted(range(n_docs), key=lambda index: scores[index], reverse=True)
    return [
        {"chunk_index": index, "bm25_score": round(scores[index], 4)}
        for index in ranked[:limit]
        if scores[index] > 0
    ]


def export_bundle(corpus: str, data_dir: Path, out_dir: Path) -> dict[str, Any]:
    index_dir = data_dir / "index"
    raw_chunks = load_jsonl(index_dir / "chunks.jsonl")
    public_chunks = [public_chunk(chunk) for chunk in raw_chunks]
    bm25 = json.loads((index_dir / "bm25.json").read_text(encoding="utf-8"))

    documents: dict[str, dict[str, Any]] = {}
    block_types: Counter[str] = Counter()
    for chunk, public in zip(raw_chunks, public_chunks, strict=True):
        block_types[public["type"]] += 1
        doc = documents.setdefault(
            public["doc_id"],
            {
                "id": public["doc_id"],
                "title": str((chunk.get("doc_card") or {}).get("title") or public["doc_id"]),
                "series": str((chunk.get("doc_card") or {}).get("series") or ""),
                "product_type": str((chunk.get("doc_card") or {}).get("product_type") or ""),
                "frequency_range": str((chunk.get("doc_card") or {}).get("frequency_range") or ""),
                "package": str((chunk.get("doc_card") or {}).get("package_size_mm") or ""),
                "chunks": 0,
                "pages": set(),
            },
        )
        doc["chunks"] += 1
        doc["pages"].add(public["page"])

    document_rows = []
    for doc in documents.values():
        document_rows.append({**doc, "pages": len(doc["pages"])})
    document_rows.sort(key=lambda item: item["id"])

    examples = []
    image_ids: set[str] = set()
    for query in DEFAULT_QUERIES[corpus]:
        results = bm25_results(query, bm25)
        for result in results:
            image_ids.add(public_chunks[result["chunk_index"]]["id"])
        examples.append({"query": query, "mode": "BM25 browser snapshot", "results": results})

    for chunk in public_chunks:
        chunk["image_available"] = chunk["id"] in image_ids

    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_payload = {
        "id": corpus,
        "title": "TXC Layout-Aware RAG" if corpus == "txc" else "TKD Layout-Aware RAG",
        "manufacturer": corpus.upper(),
        "documents": len(document_rows),
        "chunks": len(public_chunks),
        "pages": sum(item["pages"] for item in document_rows),
        "block_types": block_types.most_common(),
        "document_rows": document_rows,
        "public_mode": "Live BM25 search + static evidence snapshots; no visitor API keys.",
    }
    (out_dir / "corpus.json").write_text(
        json.dumps(corpus_payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    (out_dir / "chunks.json").write_text(
        json.dumps(public_chunks, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    (out_dir / "bm25.json").write_text(
        json.dumps(bm25, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    (out_dir / "examples.json").write_text(
        json.dumps(examples, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )

    evidence_root = out_dir.parent / "evidence"
    copied = 0
    for raw, public in zip(raw_chunks, public_chunks, strict=True):
        if public["id"] not in image_ids:
            continue
        source = data_dir / "blocks" / "merged" / public["doc_id"] / f"{public['id']}.png"
        if not source.is_file():
            continue
        target = evidence_root / public["doc_id"] / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied += 1

    return {
        "corpus": corpus,
        "documents": len(document_rows),
        "chunks": len(public_chunks),
        "images": copied,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=("txc", "tkd"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = export_bundle(args.corpus, args.data_dir.resolve(), args.out.resolve())
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
