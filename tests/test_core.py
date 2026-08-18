from __future__ import annotations

import json
from pathlib import Path

import pytest

from common.tokenize import tokenize
from evidence_service import EvidenceService
from pipeline.layout import dedup_elements
from pipeline.search import Index
from pipeline.vlm_blocks import split_discontiguous
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


def test_dedup_keeps_the_larger_box_even_at_lower_confidence():
    """The 6u table_footnote case, pinned.

    The complete box scores 0.37 and the truncated one 0.73. Ranking by
    confidence keeps the truncated box and silently cuts off the start of the
    Note line; ranking by area keeps the box that preserves more of the source.
    This is the rule METHOD.md argues hardest for, so it is pinned here.
    """
    complete = {"label": "table_footnote", "conf": 0.37, "bbox_px": [100, 500, 900, 560]}
    truncated = {"label": "table_footnote", "conf": 0.73, "bbox_px": [400, 500, 900, 560]}

    survivors = dedup_elements([truncated, complete])

    assert len(survivors) == 1
    assert survivors[0]["bbox_px"] == complete["bbox_px"]
    assert survivors[0]["conf"] == 0.37


def test_dedup_preserves_cross_class_nesting():
    """A plain_text annotation inside a figure is content, not a duplicate.

    Resolving it is a grouping decision, so detection must leave it alone.
    """
    figure = {"label": "figure", "conf": 0.9, "bbox_px": [0, 0, 800, 800]}
    annotation = {"label": "plain_text", "conf": 0.5, "bbox_px": [100, 100, 300, 160]}

    survivors = dedup_elements([figure, annotation])

    assert len(survivors) == 2
    assert {s["label"] for s in survivors} == {"figure", "plain_text"}


def test_dedup_drops_a_same_class_box_nested_in_a_larger_one():
    outer = {"label": "table", "conf": 0.4, "bbox_px": [0, 0, 600, 400]}
    inner = {"label": "table", "conf": 0.95, "bbox_px": [10, 10, 590, 390]}

    survivors = dedup_elements([inner, outer])

    assert len(survivors) == 1
    assert survivors[0]["bbox_px"] == outer["bbox_px"]


def test_header_and_footer_grouped_together_are_split_apart():
    """The guard that stops one chunk from spanning a whole page."""
    page_h = 1000.0
    elems = [
        {"bbox_px": [0, 10, 600, 40]},     # running header
        {"bbox_px": [0, 950, 600, 985]},   # footer, far below
    ]
    groups = [{"elements": [0, 1], "description": "d", "keywords": ["k"]}]

    out = split_discontiguous(groups, elems, page_h)

    assert len(out) == 2
    assert [g["elements"] for g in out] == [[0], [1]]


def test_a_vertically_contiguous_group_is_left_alone():
    page_h = 1000.0
    elems = [
        {"bbox_px": [0, 100, 600, 300]},   # table
        {"bbox_px": [0, 305, 600, 340]},   # its footnote, directly beneath
    ]
    groups = [{"elements": [0, 1], "description": "d", "keywords": ["k"]}]

    out = split_discontiguous(groups, elems, page_h)

    assert len(out) == 1
    assert out[0]["elements"] == [0, 1]
    assert out[0]["description"] == "d"


def test_missing_corpus_gives_an_actionable_message(tmp_path, monkeypatch):
    """Cloning gives you the code but not the corpus, so this is the first
    thing a new user hits. It must name the release, not just a path."""
    import config
    from pipeline import search as search_mod

    monkeypatch.setattr(config, "INDEX_DIR", tmp_path / "index")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")

    with pytest.raises(SystemExit) as excinfo:
        search_mod._require_index()

    message = str(excinfo.value)
    assert "releases/tag/v0.1.0" in message
    assert "chunks.jsonl" in message
    assert "RAG_CORPUS" in message
