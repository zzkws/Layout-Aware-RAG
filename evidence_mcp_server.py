"""MCP server exposing datasheet evidence tools.

Run after installing the Python MCP SDK:
    python -X utf8 evidence_mcp_server.py
"""
from __future__ import annotations

from typing import Any

from evidence_service import EvidenceService

SERVICE = EvidenceService()


def _require_mcp():
    try:
        from mcp.server.fastmcp import FastMCP
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Python package 'mcp' is required. Install dependencies with "
            "`pip install -e .[mcp]`."
        ) from exc
    return FastMCP


FastMCP = _require_mcp()
mcp = FastMCP("evidence-rag")


@mcp.tool()
def build_evidence_package(
    request: str,
    top_k: int = 15,
    filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Retrieve and organize evidence chunks for an engineering request."""
    return SERVICE.build_evidence_package(request, top_k, filters)


@mcp.tool()
def get_chunk(chunk_id: str) -> dict[str, Any]:
    """Return full indexed metadata for a chunk."""
    try:
        return SERVICE.get_chunk(chunk_id)
    except KeyError as exc:
        return {"error": str(exc), "chunk_id": chunk_id}


@mcp.tool()
def get_evidence_image(chunk_id: str, as_base64: bool = False) -> dict[str, Any]:
    """Return the merged evidence image for a chunk, optionally as base64."""
    try:
        return SERVICE.get_evidence_image(chunk_id, as_base64=as_base64)
    except KeyError as exc:
        return {"error": str(exc), "chunk_id": chunk_id}


if __name__ == "__main__":
    mcp.run()
