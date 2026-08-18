"""Demonstration web backend: a standard-library HTTP service, no new deps.

  GET  /            single-page frontend (webapp/index.html)
  GET  /chunks      chunk browser (webapp/chunks.html)
  GET  /doc         per-document TOC tree (webapp/doc.html, ?doc=<doc_id>)
  GET  /api/stats   corpus distribution (document / page / chunk / block_type
                    counts, plus the document summary cards)
  GET  /api/chunks  every indexed chunk; the data source for the browser page
  GET  /api/doc     one document's doc_card (with toc tree) and all its chunks
                    (?id=<doc_id>)
  POST /api/search  {"query": "...", "top_k": 5}   retrieve directly
  POST /api/ask     {"request": "...", "top_k": 5} optional query rewrite,
                                                   then retrieve
  POST /api/answer  {"request": "...", "query": "...", "chunk_ids": [...]}
                    An optional OpenAI-compatible multimodal model then reads
                    the evidence (text plus crops) together with the original
                    request and produces an answer carrying [E1] citations.
  GET  /chat        optional model chat page (webapp/chat.html)
  POST /api/chat    {"messages": [...]} -> SSE stream proxied to the configured
                    model service
  GET  /data/...    static pipeline artifacts (chunk crops and so on)

Usage:
    python -X utf8 webapp/server.py [--port 8765]
"""
import argparse
import base64
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from pipeline.search import Index

WEB_DIR = Path(__file__).parent
INDEX = Index()
_lock = threading.Lock()   # fastembed inference is not thread-safe: serialize retrieval


def _b64_image(path, max_w=1200):
    """Load an image and cap its width.

    Width only, never longest edge: a merged chunk image is a tall narrow
    strip, and fitting it to a longest-edge budget crushes table text into
    illegibility."""
    from io import BytesIO

    from PIL import Image
    img = Image.open(path)
    if img.width > max_w:
        img = img.resize((max_w, round(img.height * max_w / img.width)))
    buf = BytesIO()
    img.convert("RGB").save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()

# The prompts below are intentionally written in Chinese: they drive the demo
# interface, which is Chinese. Everything else in this repository -- pipeline,
# evidence contract, documentation -- is English. Both prompt bodies are also
# corpus-aware, substituting the active manufacturer for the default label.
CHAT_SYS = """你是 TXC 石英器件技术助手。
你熟悉石英晶体谐振器、TCXO/VCXO/OCXO/SPXO 等频率器件的参数体系
（频率容差、温度稳定度、负载电容、ESR、相位噪声、老化等）。
回答用中文（除非用户用其他语言），工程师口吻，简洁直接；
不确定的参数值明确说不确定，不要编造具体数值。""".replace("TXC", config.MANUFACTURER)

DEEPSEEK_SYS = """You translate an engineer's request into ONE retrieval query for a RAG
index of TXC quartz-device datasheets (crystal units, TCXO, VCXO, OCXO).
The index contains English chunk descriptions and datasheet text: electrical
specification tables, pin connections, package dimensions, land patterns,
test circuits, product features, applications.
Rules: write one concise English query (8~20 words); keep concrete parameter
names, units, values and series names from the request; no boolean syntax.
Return JSON: {"query": "...", "note": "<一句中文：你如何理解需求并改写>"}
""".replace("TXC", config.MANUFACTURER)


def corpus_stats() -> dict:
    docs, types = {}, {}
    for c in INDEX.chunks:
        d = docs.setdefault(c["doc_id"], {"chunks": 0, "pages": set(),
                                          "card": c.get("doc_card", {})})
        d["chunks"] += 1
        d["pages"].add(c["page"])
        types[c["block_type"]] = types.get(c["block_type"], 0) + 1
    return {
        "total_docs": len(docs),
        "total_pages": sum(len(d["pages"]) for d in docs.values()),
        "total_chunks": len(INDEX.chunks),
        "embed_model": config.EMBED_MODEL,
        "vlm_model": config.GEMINI_MODEL,
        "block_types": sorted(types.items(), key=lambda kv: -kv[1]),
        "docs": [{
            "doc_id": k, "chunks": v["chunks"], "pages": len(v["pages"]),
            "series": v["card"].get("series", ""),
            "product_type": v["card"].get("product_type", ""),
            "title": v["card"].get("title", ""),
            "frequency_range": v["card"].get("frequency_range", ""),
            "package": v["card"].get("package_size_mm", ""),
        } for k, v in sorted(docs.items())],
    }


def _merged_image(c) -> str | None:
    """Conventional path of a chunk's merged image, if it exists."""
    rel = f"blocks/merged/{c['doc_id']}/{c['chunk_id']}.png"
    return rel if (config.DATA_DIR / rel).is_file() else None


def do_search(query: str, top_k: int) -> list:
    with _lock:
        results = INDEX.search(query, top_k)
    for r in results:  # Windows path -> URL
        r["crop_images"] = [p.replace("\\", "/") for p in r["crop_images"]]
        r["source_pdf"] = r["source_pdf"].replace("\\", "/")
        r["merged_image"] = _merged_image(r)
    return results


ANSWER_PROMPT = """你是 TXC 石英器件 datasheet 助手。请基于下方检索到的证据块回答用户需求。

用户原始需求：
{request}

（检索式：{query}）

证据块 [E1]~[E{n}] 的文本如下，对应的版面裁剪图按同样顺序附在消息里：
{evidence}

要求：
- 只依据证据回答，具体数值带单位照抄，不要推算或编造
- 引用出处用 [E1] 这样的标记，紧跟在对应结论后面
- 证据不足以回答的部分，明确说缺什么
- 用中文，150~300 字，工程师口吻，可用短列表
- 纯文本输出：禁止 LaTeX 和 $ 符号，±/~/℃ 直接写字符（如 ±5ppb）；
  仅允许 **加粗** 和以 - 开头的列表行""".replace("TXC", config.MANUFACTURER)


def _auth_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def generate_answer(user_request: str, query: str, chunks: list) -> str:
    """Have the optional OpenAI-compatible multimodal service read the evidence
    and produce an answer with citations."""
    if not (config.LLM_BASE_URL and config.LLM_MODEL):
        raise RuntimeError(
            "answer generation is disabled; set RAG_LLM_BASE_URL and RAG_LLM_MODEL"
        )
    evidence_lines, content = [], []
    n_img = 0
    for i, c in enumerate(chunks, 1):
        evidence_lines.append(
            f"[E{i}] {c['doc_id']}.pdf 第{c['page']}页 | {c['block_type']} | "
            f"{c.get('section_title') or '-'} | "
            f"目录位置: {c.get('toc_path') or '-'}\n"
            f"  description: {c['description']}\n"
            f"  text: {c['native_text'][:600]}")
        # One merged image per chunk, at the conventional merge_crops.py path.
        # Older data without a merged image falls back to the first member crop.
        merged = (config.DATA_DIR / "blocks" / "merged" / c["doc_id"]
                  / f"{c['chunk_id']}.png")
        img_path = (merged if merged.exists() else
                    (config.DATA_DIR / c["crop_images"][0].replace("\\", "/")
                     if c["crop_images"] else None))
        if img_path is not None:
            content.append({"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{_b64_image(img_path)}"}})
            n_img += 1
    prompt = ANSWER_PROMPT.format(request=user_request, query=query,
                                  n=len(chunks),
                                  evidence="\n".join(evidence_lines))
    prompt = prompt.replace(
        "对应的版面裁剪图按同样顺序附在消息里：",
        f"每个证据块对应一张版面合并图，按同样顺序附在消息里（共 {n_img} 张）：")
    content.insert(0, {"type": "text", "text": prompt})
    r = requests.post(
        f"{config.LLM_BASE_URL}/chat/completions",
        headers=_auth_headers(config.LLM_API_KEY),
        json={"model": config.LLM_MODEL,
              "messages": [{"role": "user", "content": content}],
              "temperature": 0.2, "max_tokens": 900},
        timeout=(8, 300))
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def rewrite_query(user_request: str) -> dict:
    if not (config.REWRITE_BASE_URL and config.REWRITE_MODEL):
        return {"query": user_request, "note": "查询改写未启用，直接检索原始输入。"}
    r = requests.post(
        f"{config.REWRITE_BASE_URL}/chat/completions",
        headers=_auth_headers(config.REWRITE_API_KEY),
        json={"model": config.REWRITE_MODEL,
              "messages": [{"role": "system", "content": DEEPSEEK_SYS},
                           {"role": "user", "content": user_request}],
              "response_format": {"type": "json_object"},
              "temperature": 0.2},
        timeout=90)
    r.raise_for_status()
    return json.loads(r.json()["choices"][0]["message"]["content"])


class Handler(BaseHTTPRequestHandler):

    def _json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path, ctype: str):
        body = path.read_bytes()
        # Corpus-aware: on a non-default profile, swap the page title and
        # manufacturer name for the active corpus
        if config.MANUFACTURER != "TXC" and ctype.startswith("text/html"):
            body = body.replace(b"TXC Datasheet", config.CORPUS_TITLE.encode())
            body = body.replace(b"TXC", config.MANUFACTURER.encode())
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        try:
            if self.path in ("/", "/index.html"):
                self._file(WEB_DIR / "index.html", "text/html; charset=utf-8")
            elif self.path in ("/chat", "/chat.html"):
                self._file(WEB_DIR / "chat.html", "text/html; charset=utf-8")
            elif self.path in ("/chunks", "/chunks.html"):
                self._file(WEB_DIR / "chunks.html", "text/html; charset=utf-8")
            elif self.path.startswith("/doc"):
                self._file(WEB_DIR / "doc.html", "text/html; charset=utf-8")
            elif self.path.startswith("/api/doc"):
                from urllib.parse import parse_qs, urlparse
                doc_id = parse_qs(urlparse(self.path).query).get(
                    "id", [""])[0]
                cs = [c for c in INDEX.chunks if c["doc_id"] == doc_id]
                if not cs:
                    self._json({"error": f"unknown doc: {doc_id}"}, 404)
                    return
                self._json({
                    "doc_id": doc_id,
                    "doc_card": cs[0].get("doc_card", {}),
                    "source_pdf": cs[0]["source_pdf"].replace("\\", "/"),
                    "chunks": [{
                        "chunk_id": c["chunk_id"],
                        "page": c["page"],
                        "block_type": c["block_type"],
                        "section_title": c["section_title"],
                        "toc_path": c.get("toc_path", ""),
                        "description": c["description"],
                        "keywords": c.get("keywords", []),
                        "native_text": c["native_text"],
                        "n_elements": len(c["bboxes_pdf"]),
                        "crop_images": [p.replace("\\", "/")
                                        for p in c["crop_images"]],
                        "merged_image": _merged_image(c),
                    } for c in cs]})
            elif self.path == "/api/stats":
                self._json(corpus_stats())
            elif self.path == "/api/chunks":
                self._json({"chunks": [{
                    "chunk_id": c["chunk_id"],
                    "doc_id": c["doc_id"],
                    "page": c["page"],
                    "block_type": c["block_type"],
                    "section_title": c["section_title"],
                    "toc_path": c.get("toc_path", ""),
                    "description": c["description"],
                    "keywords": c.get("keywords", []),
                    "native_text": c["native_text"],
                    "n_elements": len(c["bboxes_pdf"]),
                    "crop_images": [p.replace("\\", "/")
                                    for p in c["crop_images"]],
                    "merged_image": _merged_image(c),
                    "source_pdf": c["source_pdf"].replace("\\", "/"),
                } for c in INDEX.chunks]})
            elif self.path.startswith("/data/"):
                rel = self.path[len("/data/"):].split("?")[0]
                target = (config.DATA_DIR / rel).resolve()
                if not str(target).startswith(str(config.DATA_DIR.resolve())):
                    self._json({"error": "forbidden"}, 403)
                    return
                if not target.is_file():
                    self._json({"error": "not found"}, 404)
                    return
                ctype = ("image/png" if target.suffix == ".png"
                         else "application/octet-stream")
                self._file(target, ctype)
            else:
                self._json({"error": "not found"}, 404)
        except Exception as e:
            self._json({"error": str(e)}, 500)

    def _chat_stream(self, payload):
        """Proxy the configured OpenAI-compatible /chat/completions as an SSE stream."""
        msgs = payload.get("messages") or []
        if not msgs:
            self._json({"error": "empty messages"}, 400)
            return
        if not (config.LLM_BASE_URL and config.LLM_MODEL):
            self._json({"error": "chat generation is disabled"}, 503)
            return
        up = requests.post(
            f"{config.LLM_BASE_URL}/chat/completions",
            headers=_auth_headers(config.LLM_API_KEY),
            json={"model": config.LLM_MODEL,
                  "messages": [{"role": "system", "content": CHAT_SYS}] + msgs,
                  "stream": True, "temperature": 0.6, "max_tokens": 1200},
            stream=True, timeout=(8, 600))
        up.raise_for_status()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            for line in up.iter_lines():
                if line:
                    self.wfile.write(line + b"\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionAbortedError):
            pass   # client disconnected mid-stream
        finally:
            up.close()

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n) or b"{}")
            top_k = int(payload.get("top_k", config.TOP_K))
            if self.path == "/api/chat":
                self._chat_stream(payload)
                return
            if self.path == "/api/search":
                query = (payload.get("query") or "").strip()
                if not query:
                    self._json({"error": "empty query"}, 400)
                    return
                self._json({"query": query, "results": do_search(query, top_k)})
            elif self.path == "/api/ask":
                req = (payload.get("request") or "").strip()
                if not req:
                    self._json({"error": "empty request"}, 400)
                    return
                rw = rewrite_query(req)
                query = rw.get("query", "").strip() or req
                self._json({"request": req, "query": query,
                            "note": rw.get("note", ""),
                            "results": do_search(query, top_k)})
            elif self.path == "/api/answer":
                req = (payload.get("request") or "").strip()
                query = (payload.get("query") or "").strip()
                ids = payload.get("chunk_ids") or []
                by_id = {c["chunk_id"]: c for c in INDEX.chunks}
                chunks = []
                for cid in ids:
                    c = by_id.get(cid)
                    if c:
                        chunks.append({**c, "crop_images": [
                            p.replace("\\", "/") for p in c["crop_images"]]})
                if not (req and chunks):
                    self._json({"error": "need request and chunk_ids"}, 400)
                    return
                self._json({"answer": generate_answer(req, query, chunks),
                            "model": config.LLM_MODEL})
            else:
                self._json({"error": "not found"}, 404)
        except Exception as e:
            self._json({"error": str(e)}, 500)

    def log_message(self, fmt, *args):  # quieter logging
        print(f"[web] {self.address_string()} {fmt % args}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1",
                    help="use 0.0.0.0 to serve on the local network")
    args = ap.parse_args()

    print("[web] warming up the embedding model ...")
    INDEX.embed_query("warm up")   # so the first real query does not stall
    print(f"[web] http://{args.host}:{args.port}  "
          f"({len(INDEX.chunks)} chunks ready)")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
