"""evidence_rag_pilot 全局配置。

所有路径以本文件所在目录为根。坐标约定：
- bbox_pdf: PDF 点坐标 (pt, 72dpi 基准)，是 chunk 的权威坐标，与渲染 DPI 无关
- bbox_px:  渲染图像素坐标，仅用于裁图和可视化，换算 = bbox_pdf * dpi / 72
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def _path_from_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default.resolve()

# ── 语料 profile：同一套管线/webapp 跑多套互相独立的语料 ──────────────
# RAG_CORPUS=txc（默认）/ tkd。路径可用环境变量覆盖，公开配置中不保存
# 个人工作区、服务器名或内网地址。
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

# 管线产物（DATA_DIR 由上方 profile 决定）
PAGES_DIR = DATA_DIR / "pages"            # {doc}/p{n}.png + {doc}/pages.json
LAYOUT_DIR = DATA_DIR / "layout"          # {doc}.json + overlay png
BLOCKS_DIR = DATA_DIR / "blocks"          # {doc}.json + crops/ + overlay png
DESC_DIR = DATA_DIR / "descriptions"      # {doc}.json (文档摘要卡 + 逐块 description)
INDEX_DIR = DATA_DIR / "index"            # chunks.jsonl + dense.npy + bm25.json
REPORTS_DIR = PROJECT_ROOT / "reports"

# 渲染
RENDER_DPI = 200

# DocLayout-YOLO
MODELS_DIR = PROJECT_ROOT / "models"
LAYOUT_MODEL_REPO = "juliozhao/DocLayout-YOLO-DocStructBench"
LAYOUT_MODEL_FILE = "doclayout_yolo_docstructbench_imgsz1024.pt"
LAYOUT_IMGSZ = 1024
LAYOUT_CONF = 0.25

# DocStructBench 10 类标签（模型输出 id -> 名称）
LAYOUT_CLASSES = {
    0: "title",
    1: "plain_text",
    2: "abandon",          # 页眉页脚页码等
    3: "figure",
    4: "figure_caption",
    5: "table",
    6: "table_caption",
    7: "table_footnote",
    8: "isolate_formula",
    9: "formula_caption",
}

# 离线 VLM（板块合并决策 + description 生成）。密钥只从环境变量读取。
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# 可选的 OpenAI-compatible 生成与查询改写服务。留空即检索-only；这也是
# 公开版本的默认行为。
LLM_BASE_URL = os.environ.get("RAG_LLM_BASE_URL", "").rstrip("/")
LLM_MODEL = os.environ.get("RAG_LLM_MODEL", "")
LLM_API_KEY = os.environ.get("RAG_LLM_API_KEY", "")
REWRITE_BASE_URL = os.environ.get("RAG_REWRITE_BASE_URL", LLM_BASE_URL).rstrip("/")
REWRITE_MODEL = os.environ.get("RAG_REWRITE_MODEL", LLM_MODEL)
REWRITE_API_KEY = os.environ.get("RAG_REWRITE_API_KEY", LLM_API_KEY)

# Embedding（backend: "st" = sentence-transformers / "fastembed" = 轻量备选）
EMBED_BACKEND = os.environ.get("RAG_EMBED_BACKEND", "st")
EMBED_MODEL = "microsoft/harrier-oss-v1-0.6b"      # 多语言，1024 维
EMBED_LOCAL_DIR = MODELS_DIR / "harrier-oss-v1-0.6b"
EMBED_QUERY_PROMPT = "web_search_query"            # 模型内置检索 query prompt
# fastembed 备选路线
FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"
FASTEMBED_CACHE = MODELS_DIR / "fastembed_cache"

# 检索
RRF_K = 60
TOP_K = 15
