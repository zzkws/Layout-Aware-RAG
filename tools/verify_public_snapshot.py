"""Validate the compact corpus snapshots shipped with the two demo sites."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

EXPECTED = {
    "txc": {"documents": 108, "pages": 226, "chunks": 875},
    "tkd": {"documents": 74, "pages": 112, "chunks": 547},
}


def verify(corpus: str, data_dir: Path) -> None:
    corpus_data = json.loads((data_dir / "corpus.json").read_text(encoding="utf-8"))
    chunks = json.loads((data_dir / "chunks.json").read_text(encoding="utf-8"))
    bm25 = json.loads((data_dir / "bm25.json").read_text(encoding="utf-8"))
    expected = EXPECTED[corpus]
    for key, value in expected.items():
        if corpus_data.get(key) != value:
            raise ValueError(f"{corpus} {key}: expected {value}, got {corpus_data.get(key)}")
    if len(chunks) != expected["chunks"] or bm25.get("n_docs") != expected["chunks"]:
        raise ValueError(f"{corpus}: chunk/BM25 length mismatch")
    ids = [row.get("id") for row in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{corpus}: duplicate chunk ids")
    payload = json.dumps(chunks, ensure_ascii=False)
    if "\\\\" in payload or ":\\" in payload:
        raise ValueError(f"{corpus}: machine-local path leaked into public chunks")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=EXPECTED, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    verify(args.corpus, args.data_dir)
    print(f"{args.corpus} public snapshot passed.")


if __name__ == "__main__":
    main()
