"""Evidence package builder for agent workflows.

This module wraps the existing dense+BM25 RAG index and turns raw search
results into a stable, claim-friendly evidence package. It deliberately does
not make engineering judgments; reasoning and verification live one layer up.
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import config
from pipeline.search import Index

DEFAULT_TOP_K = 15


class EvidenceService:
    """Read-only access to indexed chunks and evidence images."""

    def __init__(self, index: Index | None = None):
        self.index = index or Index()
        self._chunks_by_id = {c["chunk_id"]: c for c in self.index.chunks}

    def build_evidence_package(
        self,
        request: str,
        top_k: int = DEFAULT_TOP_K,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Retrieve, deduplicate, filter, and group evidence chunks."""
        request = (request or "").strip()
        if not request:
            raise ValueError("request is required")
        top_k = max(1, min(int(top_k or DEFAULT_TOP_K), 50))
        filters = filters or {}

        # Search wider when filters are present so post-filtering does not
        # starve the package.
        search_k = max(top_k, top_k * 4 if filters else top_k)
        raw_results = self.index.search(request, search_k)
        items: list[dict[str, Any]] = []
        seen: set[str] = set()
        for result in raw_results:
            chunk_id = result["chunk_id"]
            if chunk_id in seen:
                continue
            chunk = self._chunks_by_id.get(chunk_id)
            if not chunk or not self._matches_filters(chunk, filters):
                continue
            seen.add(chunk_id)
            items.append(self._evidence_item(len(items) + 1, chunk, result))
            if len(items) >= top_k:
                break

        return {
            "request": request,
            "query": request,
            "top_k": top_k,
            "filters": filters,
            "evidence": items,
            "groups": self._groups(items),
            "warnings": self._package_warnings(items),
        }

    def get_chunk(self, chunk_id: str) -> dict[str, Any]:
        chunk = self._chunks_by_id.get(chunk_id)
        if not chunk:
            raise KeyError(f"unknown chunk_id: {chunk_id}")
        return {**chunk, "image": self._image_info(chunk)}

    def get_evidence_image(self, chunk_id: str, as_base64: bool = False) -> dict[str, Any]:
        chunk = self.get_chunk(chunk_id)
        image = chunk["image"]
        if as_base64 and image["available"]:
            data = Path(image["path"]).read_bytes()
            image = {**image, "base64": base64.b64encode(data).decode("ascii")}
        return image

    def _matches_filters(self, chunk: dict[str, Any], filters: dict[str, Any]) -> bool:
        doc_ids = set(filters.get("doc_ids") or [])
        block_types = set(filters.get("block_types") or [])
        if doc_ids and chunk.get("doc_id") not in doc_ids:
            return False
        if block_types and chunk.get("block_type") not in block_types:
            return False
        return True

    def _evidence_item(
        self,
        evidence_no: int,
        chunk: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "evidence_id": f"E{evidence_no}",
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "page": chunk["page"],
            "block_type": chunk["block_type"],
            "toc_path": chunk.get("toc_path", ""),
            "section_title": chunk.get("section_title", ""),
            "native_text": chunk.get("native_text", ""),
            "description": chunk.get("description", ""),
            "keywords": chunk.get("keywords", []),
            "source_pdf": str(chunk.get("source_pdf", "")).replace("\\", "/"),
            "bboxes_pdf": chunk.get("bboxes_pdf", []),
            "image": self._image_info(chunk),
            "rank": {
                "rrf_score": result.get("rrf_score"),
                "dense_rank": result.get("dense_rank"),
                "bm25_rank": result.get("bm25_rank"),
                "dense_score": result.get("dense_score"),
                "bm25_score": result.get("bm25_score"),
            },
        }

    def _image_info(self, chunk: dict[str, Any]) -> dict[str, Any]:
        merged_rel = f"blocks/merged/{chunk['doc_id']}/{chunk['chunk_id']}.png"
        merged_path = config.DATA_DIR / merged_rel
        if merged_path.is_file():
            return {
                "available": True,
                "kind": "merged",
                "relative_path": merged_rel,
                "url": f"/data/{merged_rel}",
                "path": str(merged_path),
            }

        crops = chunk.get("crop_images") or []
        if crops:
            rel = str(crops[0]).replace("\\", "/")
            path = config.DATA_DIR / rel
            return {
                "available": path.is_file(),
                "kind": "crop",
                "relative_path": rel,
                "url": f"/data/{rel}",
                "path": str(path),
            }

        return {
            "available": False,
            "kind": "none",
            "relative_path": "",
            "url": "",
            "path": "",
        }

    def _groups(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, str, str], list[str]] = {}
        for item in items:
            key = (item["doc_id"], item["toc_path"], item["block_type"])
            grouped.setdefault(key, []).append(item["evidence_id"])
        return [
            {
                "doc_id": doc_id,
                "toc_path": toc_path,
                "block_type": block_type,
                "evidence_ids": evidence_ids,
            }
            for (doc_id, toc_path, block_type), evidence_ids in grouped.items()
        ]

    def _package_warnings(self, items: list[dict[str, Any]]) -> list[str]:
        warnings: list[str] = []
        if not items:
            warnings.append("No evidence chunks were retrieved.")
        missing_images = [i["evidence_id"] for i in items if not i["image"]["available"]]
        if missing_images:
            warnings.append(
                "Missing evidence images for: " + ", ".join(missing_images)
            )
        return warnings
