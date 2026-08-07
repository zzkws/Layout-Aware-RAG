"""统一 embedding 后端（惰性单例）。

st       : sentence-transformers，默认 microsoft/harrier-oss-v1-0.6b
           （Qwen3 架构多语言检索模型；query 侧走模型内置
            web_search_query prompt，文档侧不加 prompt）
fastembed: ONNX bge-small 轻量备选（离线缓存）

index_build / search / webapp 统一经由 embed_docs / embed_query 调用，
切换后端只改 config.EMBED_BACKEND。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

_model = None


def _load():
    global _model
    if _model is not None:
        return _model
    if config.EMBED_BACKEND == "st":
        from sentence_transformers import SentenceTransformer
        path = (config.EMBED_LOCAL_DIR if config.EMBED_LOCAL_DIR.exists()
                else config.EMBED_MODEL)
        _model = SentenceTransformer(str(path))  # 自动选 cuda/cpu
    else:
        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "1")  # 优先本地缓存
        from fastembed import TextEmbedding
        try:
            _model = TextEmbedding(model_name=config.FASTEMBED_MODEL,
                                   cache_dir=str(config.FASTEMBED_CACHE))
        except Exception:
            os.environ["HF_HUB_OFFLINE"] = "0"
            _model = TextEmbedding(model_name=config.FASTEMBED_MODEL,
                                   cache_dir=str(config.FASTEMBED_CACHE))
    return _model


def embed_docs(texts: list[str]) -> np.ndarray:
    m = _load()
    if config.EMBED_BACKEND == "st":
        vecs = m.encode(list(texts), normalize_embeddings=True,
                        batch_size=8, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)
    vecs = np.array(list(m.embed(list(texts))), dtype=np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def embed_query(text: str) -> np.ndarray:
    m = _load()
    if config.EMBED_BACKEND == "st":
        kwargs = {}
        if (getattr(m, "prompts", None)
                and config.EMBED_QUERY_PROMPT in m.prompts):
            kwargs["prompt_name"] = config.EMBED_QUERY_PROMPT
        vec = m.encode([text], normalize_embeddings=True,
                       show_progress_bar=False, **kwargs)[0]
        return np.asarray(vec, dtype=np.float32)
    vec = np.array(list(m.query_embed(text))[0], dtype=np.float32)
    return vec / np.linalg.norm(vec)
