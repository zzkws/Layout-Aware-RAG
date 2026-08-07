"""Helpers for exporting public, machine-independent corpus metadata."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def public_source_name(value: str) -> str:
    """Return a corpus-relative PDF name without leaking local paths."""
    normalized = str(value or "").replace("\\", "/")
    return f"pdf/{Path(normalized).name}" if normalized else ""


def public_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    """Project an internal chunk onto the stable public site schema."""
    chunk_id = str(chunk["chunk_id"])
    doc_id = str(chunk["doc_id"])
    return {
        "id": chunk_id,
        "doc_id": doc_id,
        "page": int(chunk.get("page") or 0),
        "type": str(chunk.get("block_type") or "unknown"),
        "title": str(chunk.get("section_title") or ""),
        "toc": str(chunk.get("toc_path") or ""),
        "description": str(chunk.get("description") or ""),
        "text": str(chunk.get("native_text") or ""),
        "keywords": list(chunk.get("keywords") or []),
        "bboxes": list(chunk.get("bboxes_pdf") or []),
        "source_pdf": public_source_name(str(chunk.get("source_pdf") or "")),
        "evidence_image": f"/evidence/{doc_id}/{chunk_id}.png",
    }
