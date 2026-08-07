"""生成 HTML 演示报告：各阶段产物 + 检索结果。

用法:
    python build_report.py --search-json reports/search_results.json
"""
import argparse
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config

CSS = """
body{font-family:'Segoe UI',system-ui,sans-serif;max-width:1200px;margin:24px auto;
     padding:0 16px;color:#222;line-height:1.55}
h1{border-bottom:3px solid #d62728;padding-bottom:8px}
h2{border-left:5px solid #d62728;padding-left:10px;margin-top:40px}
h3{margin-top:28px}
img{max-width:100%;border:1px solid #ccc;border-radius:4px}
.row{display:flex;gap:12px;flex-wrap:wrap}
.row>div{flex:1;min-width:380px}
.cap{font-size:13px;color:#666;margin:4px 0 16px}
.chunk{border:1px solid #ddd;border-radius:6px;padding:12px 16px;margin:14px 0;
       background:#fafafa}
.chunk img{max-width:560px;display:block;margin:8px 0}
.meta{font-size:12.5px;color:#555;font-family:Consolas,monospace}
.desc{background:#eef6ff;padding:8px 10px;border-radius:4px;margin:8px 0;font-size:14px}
.txt{font-size:12.5px;color:#444;font-family:Consolas,monospace;background:#f4f4f4;
     padding:6px 8px;border-radius:4px;max-height:90px;overflow:auto}
.q{background:#fff8e6;border:1px solid #e8c96a;border-radius:6px;padding:10px 14px;
   margin-top:26px;font-weight:600}
.rank{font-size:13px;color:#888}
table{border-collapse:collapse;font-size:14px}
td,th{border:1px solid #ccc;padding:4px 10px}
"""


def rel(p):  # data/ 相对路径 -> 报告相对路径
    return f"../data/{p}".replace("\\", "/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search-json", required=True)
    ap.add_argument("--out", default=str(config.REPORTS_DIR / "demo_report.html"))
    args = ap.parse_args()

    searches = json.loads(Path(args.search_json).read_text(encoding="utf-8"))
    parts = [f"<html><head><meta charset='utf-8'>"
             f"<title>evidence_rag_pilot 演示报告</title>"
             f"<style>{CSS}</style></head><body>"]
    parts.append("<h1>TXC Datasheet 多模态版面切块 RAG — 演示报告</h1>")
    parts.append(
        "<p>管线: PDF → 页图渲染 → DocLayout-YOLO 元素框 → 板块合并 → "
        "文字层抽取 → VLM description → dense+BM25 双路索引 → RRF 融合检索。"
        "chunk = 页面版面板块，召回自带 file/page/bbox/裁剪图证据。</p>")

    # 每文档统计 + 可视化
    parts.append("<h2>1. 切块结果</h2>")
    parts.append("<table><tr><th>文档</th><th>页数</th><th>板块数</th>"
                 "<th>可索引</th><th>板块类型</th></tr>")
    for doc_id in config.DEMO_DOCS:
        blocks = json.loads((config.BLOCKS_DIR / f"{doc_id}.json")
                            .read_text(encoding="utf-8"))["blocks"]
        pages = json.loads((config.PAGES_DIR / doc_id / "pages.json")
                           .read_text(encoding="utf-8"))["page_count"]
        idx = [b for b in blocks if b["indexable"]]
        types = sorted({b["block_type"] for b in idx})
        parts.append(f"<tr><td>{doc_id}</td><td>{pages}</td>"
                     f"<td>{len(blocks)}</td><td>{len(idx)}</td>"
                     f"<td>{', '.join(types)}</td></tr>")
    parts.append("</table>")

    parts.append("<h2>2. 元素框 vs 合并后板块（gold set 页面）</h2>")
    for doc_id, pno in [("6u", 1), ("6u", 2), ("7n_10pad", 2), ("7n_10pad", 4)]:
        parts.append(f"<h3>{doc_id} 第 {pno} 页</h3><div class='row'>")
        parts.append(f"<div><img src='{rel(f'layout/overlays/{doc_id}/p{pno}_elements.png')}'>"
                     f"<div class='cap'>DocLayout-YOLO 元素级输出</div></div>")
        parts.append(f"<div><img src='{rel(f'blocks/overlays/{doc_id}/p{pno}_blocks.png')}'>"
                     f"<div class='cap'>规则合并后的板块（=chunk）</div></div></div>")

    # chunk 样例
    parts.append("<h2>3. Chunk 样例（图 + 文字层 + description）</h2>")
    shown = 0
    for doc_id in ["6u", "7n_10pad"]:
        blocks = json.loads((config.BLOCKS_DIR / f"{doc_id}.json")
                            .read_text(encoding="utf-8"))["blocks"]
        descs = json.loads((config.DESC_DIR / f"{doc_id}.json")
                           .read_text(encoding="utf-8"))["blocks"]
        for b in blocks:
            if not b["indexable"] or b["block_id"] not in descs or shown >= 6:
                continue
            if b["block_type"] in ("text_section", "notes"):
                continue
            d = descs[b["block_id"]]
            imgs = "".join(f"<img src='{rel(p)}'>" for p in b["crop_images"])
            bb = " ".join(str(x) for x in b["bboxes_pdf"])
            toc = b.get("toc_path", "")
            parts.append(
                f"<div class='chunk'><div class='meta'>{b['block_id']} | "
                f"{b['block_type']} | {html.escape(b['section_title'] or '-')} | "
                f"p{b['page']} | {len(b['crop_images'])} 个元素 | "
                f"bboxes={html.escape(bb[:160])}"
                f"{' | 目录: ' + html.escape(toc) if toc else ''}</div>"
                f"{imgs}"
                f"<div class='desc'><b>description:</b> "
                f"{html.escape(d['description'])}</div>"
                f"<div class='txt'>{html.escape(b['native_text'][:400])}</div></div>")
            shown += 1

    # 检索结果
    parts.append("<h2>4. 检索演示（dense + BM25 → RRF）</h2>")
    for q, results in searches.items():
        parts.append(f"<div class='q'>Query: {html.escape(q)}</div>")
        for r in results[:3]:
            imgs = "".join(f"<img src='{rel(p)}'>" for p in r["crop_images"])
            bb = " ".join(str(x) for x in r["bboxes_pdf"])
            toc = (
                " | 目录: " + html.escape(r.get("toc_path") or "")
                if r.get("toc_path")
                else ""
            )
            parts.append(
                f"<div class='chunk'><div class='rank'>RRF {r['rrf_score']} "
                f"(dense#{r['dense_rank']} / bm25#{r['bm25_rank']}) — "
                f"<b>{r['chunk_id']}</b> {r['block_type']}</div>"
                f"<div class='meta'>evidence: {html.escape(r['source_pdf'])} "
                f"第{r['page']}页 | {len(r['crop_images'])} 个元素 | "
                f"bboxes={html.escape(bb[:160])}{toc}</div>"
                f"{imgs}"
                f"<div class='desc'>{html.escape(r['description'])}</div></div>")

    parts.append("</body></html>")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"[report] {out}")


if __name__ == "__main__":
    main()
