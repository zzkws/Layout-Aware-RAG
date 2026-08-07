from __future__ import annotations

import json
from pathlib import Path

from common.tokenize import tokenize
from evidence_service import EvidenceService
from pipeline.search import Index
from tools.export_site_data import export_bundle
from tools.public_data import public_chunk, public_source_name


def test_model_number_tokenizer_keeps_exact_and_partial_terms():
    tokens = tokenize("7M-26.000MAAJ-T load capacitance")
    assert "7m-26.000maaj-t" in tokens
    assert "26" in tokens
    assert "000maaj" in tokens
    assert "load" in tokens


def test_bm25_ranks_matching_document_first():
    index = Index.__new__(Index)
    index.n_docs = 2
    index.avg_len = 2.0
    index.doc_lens = [2, 2]
    index.postings = {"crystal": {0: 2}, "oscillator": {1: 1}}
    ranked, scores = index.bm25_rank("crystal")
    assert ranked[0] == 0
    assert scores[0] > scores[1]


class FakeIndex:
    def __init__(self):
        self.chunks = [
            {
                "chunk_id": "d1_p1_b01",
                "doc_id": "d1",
                "page": 1,
                "block_type": "table",
                "section_title": "Specifications",
                "toc_path": "D1 > Specifications",
                "native_text": "frequency stability ±10 ppm",
                "description": "Electrical specification table",
                "keywords": ["stability"],
                "source_pdf": "C:" + r"\private\d1.pdf",
                "bboxes_pdf": [[1, 2, 3, 4]],
                "crop_images": [],
            }
        ]

    def search(self, request, top_k):
        return [
            {
                "chunk_id": "d1_p1_b01",
                "rrf_score": 0.03,
                "dense_rank": 1,
                "bm25_rank": 2,
                "dense_score": 0.8,
                "bm25_score": 4.2,
            }
        ]


def test_evidence_service_preserves_public_tool_contract():
    service = EvidenceService(index=FakeIndex())
    package = service.build_evidence_package("frequency stability", filters={"doc_ids": ["d1"]})
    assert package["evidence"][0]["evidence_id"] == "E1"
    assert package["evidence"][0]["rank"]["dense_rank"] == 1
    assert package["groups"][0]["evidence_ids"] == ["E1"]


def test_public_paths_strip_machine_details():
    source = "D:" + r"\Quartz Corpus\pdf\6u.pdf"
    assert public_source_name(source) == "pdf/6u.pdf"
    projected = public_chunk(FakeIndex().chunks[0])
    assert projected["source_pdf"] == "pdf/d1.pdf"
    assert "private" not in json.dumps(projected)


def test_site_export_is_compact_and_path_safe(tmp_path: Path):
    data_dir = tmp_path / "data"
    index_dir = data_dir / "index"
    index_dir.mkdir(parents=True)
    chunk = {**FakeIndex().chunks[0], "doc_card": {"title": "Demo"}}
    (index_dir / "chunks.jsonl").write_text(json.dumps(chunk) + "\n", encoding="utf-8")
    bm25 = {
        "n_docs": 1,
        "avg_len": 2,
        "doc_lens": [2],
        "postings": {"crystal": [[0, 1]]},
    }
    (index_dir / "bm25.json").write_text(json.dumps(bm25), encoding="utf-8")

    out = tmp_path / "site" / "data"
    summary = export_bundle("txc", data_dir, out)
    exported = json.loads((out / "chunks.json").read_text(encoding="utf-8"))
    assert summary["documents"] == 1
    assert exported[0]["source_pdf"] == "pdf/d1.pdf"
    assert not any("C:" in path.read_text(encoding="utf-8") for path in out.glob("*.json"))
