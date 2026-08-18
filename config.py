"""Global configuration.

All paths are rooted at this file's directory. Coordinate contract:
- bbox_pdf: PDF points (pt, 72 dpi basis). The authoritative chunk coordinate,
            independent of render DPI.
- bbox_px:  rendered-page pixels. Cropping and visualization only.
            bbox_px = bbox_pdf * dpi / 72
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def _path_from_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default.resolve()

# ── Corpus profiles: one pipeline / webapp over several independent corpora ──
# RAG_CORPUS=txc (default) / tkd. Paths are overridable by environment variable
# so that no personal workspace, host name, or internal address is committed.
CORPUS = os.environ.get("RAG_CORPUS", "txc").lower()
if CORPUS not in {"txc", "tkd"}:
    raise ValueError("RAG_CORPUS must be 'txc' or 'tkd'")

if CORPUS == "tkd":
    MANUFACTURER = "TKD"
    CORPUS_TITLE = "TKD Datasheet"
    PDF_SOURCE_DIR = _path_from_env(
        "RAG_SOURCE_DIR", PROJECT_ROOT / "corpora" / "tkd" / "pdf"
    )
    DATA_DIR = _path_from_env(
        "RAG_DATA_DIR", PROJECT_ROOT / "corpora" / "tkd" / "data"
    )
    DEMO_DOCS = ["selection_guide"]
else:
    MANUFACTURER = "TXC"
    CORPUS_TITLE = "TXC Datasheet"
    PDF_SOURCE_DIR = _path_from_env(
        "RAG_SOURCE_DIR", PROJECT_ROOT / "corpora" / "txc" / "pdf"
    )
    DATA_DIR = _path_from_env(
        "RAG_DATA_DIR", PROJECT_ROOT / "corpora" / "txc" / "data"
    )
    DEMO_DOCS = ["6u", "7n_10pad", "7m", "8y", "oe_1"]

# Pipeline artifacts (DATA_DIR is chosen by the profile above)
PAGES_DIR = DATA_DIR / "pages"            # {doc}/p{n}.png + {doc}/pages.json
LAYOUT_DIR = DATA_DIR / "layout"          # {doc}.json + overlay png
BLOCKS_DIR = DATA_DIR / "blocks"          # {doc}.json + crops/ + overlay png
DESC_DIR = DATA_DIR / "descriptions"      # {doc}.json (doc card + per-chunk description)
INDEX_DIR = DATA_DIR / "index"            # chunks.jsonl + dense.npy + bm25.json
REPORTS_DIR = PROJECT_ROOT / "reports"

# Rendering
RENDER_DPI = 200

# DocLayout-YOLO
MODELS_DIR = PROJECT_ROOT / "models"
LAYOUT_MODEL_REPO = "juliozhao/DocLayout-YOLO-DocStructBench"
LAYOUT_MODEL_FILE = "doclayout_yolo_docstructbench_imgsz1024.pt"
LAYOUT_IMGSZ = 1024
LAYOUT_CONF = 0.25

# DocStructBench 10-class label set (model output id -> name)
LAYOUT_CLASSES = {
    0: "title",
    1: "plain_text",
    2: "abandon",          # headers, footers, page numbers
    3: "figure",
    4: "figure_caption",
    5: "table",
    6: "table_caption",
    7: "table_footnote",
    8: "isolate_formula",
    9: "formula_caption",
}

# Offline VLM (element grouping decisions + description generation).
# The key is read from the environment only.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# Optional OpenAI-compatible generation and query-rewrite services. Unset means
# retrieval-only, which is the default and the public configuration.
LLM_BASE_URL = os.environ.get("RAG_LLM_BASE_URL", "").rstrip("/")
LLM_MODEL = os.environ.get("RAG_LLM_MODEL", "")
LLM_API_KEY = os.environ.get("RAG_LLM_API_KEY", "")
REWRITE_BASE_URL = os.environ.get("RAG_REWRITE_BASE_URL", LLM_BASE_URL).rstrip("/")
REWRITE_MODEL = os.environ.get("RAG_REWRITE_MODEL", LLM_MODEL)
REWRITE_API_KEY = os.environ.get("RAG_REWRITE_API_KEY", LLM_API_KEY)

# Embedding (backend: "st" = sentence-transformers / "fastembed" = light fallback)
EMBED_BACKEND = os.environ.get("RAG_EMBED_BACKEND", "st")
EMBED_MODEL = "microsoft/harrier-oss-v1-0.6b"      # multilingual, 1024-dim
EMBED_LOCAL_DIR = MODELS_DIR / "harrier-oss-v1-0.6b"
EMBED_QUERY_PROMPT = "web_search_query"            # the model's built-in query prompt
# fastembed fallback route
FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"
FASTEMBED_CACHE = MODELS_DIR / "fastembed_cache"

# Retrieval
RRF_K = 60
TOP_K = 15
