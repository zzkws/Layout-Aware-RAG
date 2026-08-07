"""Compute reproducible retrieval-only metrics without calling a generation model."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import mean


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def reciprocal_rank(retrieved: list[str], gold: set[str]) -> float:
    for rank, chunk_id in enumerate(retrieved, 1):
        if chunk_id in gold:
            return 1.0 / rank
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=("txc", "tkd"), default="tkd")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--input", type=Path, default=Path("eval/seed_eval.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("eval/retrieval_baseline.json"))
    args = parser.parse_args()

    os.environ["RAG_CORPUS"] = args.corpus
    if args.data_dir:
        os.environ["RAG_DATA_DIR"] = str(args.data_dir.resolve())

    from pipeline.search import Index

    index = Index()
    rows = []
    for item in read_jsonl(args.input):
        gold = set(item.get("gold_chunks") or [])
        results = index.search(item["question"], top_k=15)
        retrieved = [result["chunk_id"] for result in results]
        rows.append(
            {
                "id": item["id"],
                "answerable": bool(item.get("answerable")),
                "gold_count": len(gold),
                "recall_at_5": len(gold & set(retrieved[:5])) / len(gold) if gold else None,
                "recall_at_15": len(gold & set(retrieved[:15])) / len(gold) if gold else None,
                "reciprocal_rank": reciprocal_rank(retrieved, gold) if gold else None,
                "retrieved": retrieved,
            }
        )

    scored = [row for row in rows if row["gold_count"]]
    summary = {
        "corpus": args.corpus,
        "queries": len(rows),
        "scored_queries": len(scored),
        "recall_at_5": round(mean(row["recall_at_5"] for row in scored), 4) if scored else None,
        "recall_at_15": round(mean(row["recall_at_15"] for row in scored), 4) if scored else None,
        "mrr_at_15": round(mean(row["reciprocal_rank"] for row in scored), 4) if scored else None,
        "rows": rows,
    }
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    public_summary = {key: value for key, value in summary.items() if key != "rows"}
    print(json.dumps(public_summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
