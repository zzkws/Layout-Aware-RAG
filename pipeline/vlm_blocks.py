"""Stage 4: VLM（Gemini）板块切分 + 目录构建 + 描述生成（自底向上）。

三段式（每个 Gemini 调用都要求英文直出，无中文字符）：

  Pass 1（每文档一次）：全部原始页图 -> 文档摘要卡 doc_card + doc_name
                        （doc_name 作为目录根，自动加前缀；本阶段不建 toc）
  Pass 2（每页一次）  ：doc_card + 原始页图 + 元素编号标注图 + 元素清单
                        -> 决定元素分组成板块(chunk)，逐块生成
                           section_title / description / keywords / block_type
                           （此阶段不分配 toc_path）
  Pass 3（每文档一次）：把全部已分好的 chunk 按阅读序汇成清单（纯文本）
                        -> 由 chunk 归纳出全文目录 toc（要求严格按阅读序排），
                           并为每个 chunk 分配目录位置 toc_path

代码侧兜底校验：元素号去重、漏分配元素补成独立板块、按 y 排阅读顺序、
空 toc_path 判索引功能性块（toc_redundant，不进索引；含 figure/table 的块
永不被目录替代，VLM 误判空路径时兜底挂根）。产物：blocks/{doc}.json + 裁剪图
+ overlay + descriptions/{doc}.json，下游 index_build / search / report 无需改动。

用法:
    python pipeline/vlm_blocks.py [--docs selection_guide ...]
环境:
    GEMINI_API_KEY, GEMINI_MODEL 可覆盖默认模型
"""
import argparse
import base64
import json
import os
import shutil
import sys
import time
from pathlib import Path

import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from common.draw import draw_boxes

CROP_PAD = 8

# 板块 overlay 用色环（同一 chunk 成员框同色）
CHUNK_PALETTE = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e",
                 "#17becf", "#e377c2", "#8c564b", "#bcbd22", "#7f7f7f"]

BLOCK_TYPES = [
    "product_features", "application", "test_condition",
    "electrical_specifications_table", "absolute_maximum_ratings_table",
    "environmental_characteristics_table", "frequency_stability_table",
    "test_diagram", "dimensions_footprint", "pin_connection",
    "recommended_land_pattern", "marking_guide", "ordering_information",
    "reflow_profile", "packaging_information", "notes",
    "table_block", "figure_block", "text_section", "header_footer",
]

# ── 三个 prompt：全字段英文直出（无中文字符）─────────────────────────────────

DOC_CARD_PROMPT = """You are reading a complete {mfr} datasheet.
There are {n_pages} page image(s) attached in order.
OUTPUT IN ENGLISH: write every text field (doc_name, title, summary) in English.
The pages may be in Chinese — read them and write the output directly in English,
using no Chinese characters.
Produce a JSON document card (NO table of contents — it is built later):
{{"manufacturer": "{mfr}", "series": "...", "product_type": "SPXO|TCXO|VCXO|OCXO|crystal_unit|...",
"title": "...", "doc_name": "...", "package_size_mm": "...", "frequency_range": "...",
"revision": "...",
"summary": "2-3 sentences describing what this datasheet covers"}}
"doc_name" is a short distinctive name you give the WHOLE document (e.g.
"{mfr} 7N Series Precise TCXO (4/10 Pad)"); it becomes the root of the document's
table of contents, prepended automatically later.
Copy values only from what you see in the images. Output JSON only."""

GROUP_PROMPT = """Document context: {doc_card}

You are looking at page {page} of {n_pages} of this datasheet.
Image 1: the original page. Image 2: the same page with numbered element boxes
from a layout detector (format "index:label").

Element inventory (index | detector label | text extracted from the PDF text
layer; empty text means the content is image-only, read it from Image 1):
{inventory}

OUTPUT IN ENGLISH: write section_title, description and every keyword in English
(the page may be Chinese — express the meaning directly in English, using no
Chinese characters).

TASK
1. Group the element indices into semantic layout blocks (region chunks):
   - a section title belongs together with its body text;
   - a table belongs together with its caption and footnotes ("Note:" lines);
   - a drawing/diagram belongs together with its sub-title and unit labels;
   - page header / footer elements form their own blocks with block_type
     "header_footer"; the top header and the bottom footer are SEPARATE blocks;
   - do NOT merge unrelated sections sitting side by side (e.g. a "Product
     Features" list on the left and an "Application" list on the right are TWO
     separate blocks);
   - a standalone product photo / certification icons strip is its own
     figure_block, not part of a text section.
   Every element index must appear in exactly one block. A block must be a
   spatially contiguous region of the page.
2. For each block write a retrieval description in the document's first-person
   perspective, e.g. "In this {mfr} {series} {ptype} datasheet, this block on
   page {page} defines ...". Spell out the series name, parameter names and
   values explicitly. Copy numbers ONLY from the element text or the images,
   never compute or guess. 2-4 sentences. For image-only blocks (empty text)
   read the content carefully from Image 1.
3. block_type must be one of: {block_types}
4. section_title = a concise ENGLISH title for this block (read the printed
   heading and state it in English; the page may be Chinese — output no Chinese
   characters). If the block has NO printed heading of its own — ESPECIALLY
   figure / drawing / photo / diagram blocks — do NOT leave it empty: give a
   short English descriptive name for what it shows, e.g. "Package Dimensions",
   "Recommended Land Pattern", "Test Circuit", "Pin Connection", "Product Photo",
   "Marking Layout", "Block Diagram". Use "" ONLY for header_footer blocks.

Output JSON only:
{{"blocks": [{{"elements": [0, 1], "block_type": "...", "section_title": "...",
"description": "...", "keywords": ["...", "..."]}}]}}"""

TOC_PROMPT = """You are organizing a {mfr} datasheet named "{doc_name}" into a table of contents.
Below is the FULL list of layout blocks already extracted from the document,
ONE PER LINE, STRICTLY IN READING ORDER — "index" is the reading position
(index 0 is read first; the index increases as you read down each page and then
move on to the next page):
  index | page | block_type | section_title | description (truncated)
{digest}

OUTPUT IN ENGLISH: every toc node title must be in English, using no Chinese
characters.

TASK
A. Build the document's table of contents as a STRUCTURED TREE that organizes
   exactly THESE blocks (do not invent sections that no block supports).
   Granularity:
   - level 1 = top-level sections (Product Features, Application, Electrical
     Specifications, Test Diagram / Test Circuit, Dimensions & Footprint,
     Pin Connection, Ordering Information, Packaging, ...); a continued table
     ("... (continued)") is its own level-1 entry;
   - level 2 = major STRUCTURAL sub-units only: package variants / sub-models /
     output types (e.g. "4 Pad", "10 Pad", "Output : CMOS");
   - STOP at that depth — do NOT list parameter rows inside a table, view names
     inside a drawing, or per-variant repeated items;
   - if the same heading repeats, keep each occurrence as its own node so every
     path is unambiguous;
   - give each node the page it first appears on;
   - OUTPUT THE TOC STRICTLY IN READING ORDER: walk the blocks from index 0
     upward and emit each section the first time one of its blocks appears
     (equivalently, order nodes by the smallest block index assigned to them).
     A section whose earliest block has a smaller index MUST come earlier in the
     toc. NEVER reorder nodes by category, type, or importance.
   Do NOT include the "{doc_name}" root itself — it is prepended automatically.
B. Assign EVERY block (by its index) to the most specific node in the tree you
   just built, as a path joined by " > " (e.g.
   "Dimensions & Footprint > 10 Pad"). Rules:
   - a block that is purely a navigational heading whose whole content becomes a
     toc node (a bare section heading, a lone "Unit : mm" label) gets "" ;
   - a block that contains a table, figure, specification values or any prose
     ALWAYS gets a non-empty path — never "" ;
   - header_footer blocks get "".

Output JSON only:
{{"toc": [{{"level": 1, "title": "Electrical Specifications", "page": 1}}],
"assignments": [{{"index": 0, "toc_path": "Product Features"}}]}}"""


# ---------------------------------------------------------------- Gemini client

def _api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise SystemExit("缺少 GEMINI_API_KEY 环境变量")
    return key


def _img_part(path: Path) -> dict:
    return {"inline_data": {
        "mime_type": "image/png",
        "data": base64.b64encode(path.read_bytes()).decode()}}


def gemini_json(parts: list, retries: int = 3) -> dict:
    model = os.environ.get("GEMINI_MODEL", config.GEMINI_MODEL)
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent")
    body = {"contents": [{"parts": parts}],
            "generationConfig": {"temperature": 0.1,
                                 "response_mime_type": "application/json"}}
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.post(url, headers={"x-goog-api-key": _api_key()},
                              json=body, timeout=300)
            if r.status_code in (429, 500, 503):
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            r.raise_for_status()
            text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(text)
        except Exception as e:  # 瞬态错误指数退避重试
            last_err = e
            if attempt < retries - 1:
                wait = 5 * (2 ** attempt)
                print(f"    [retry] {e} -> {wait}s 后重试")
                time.sleep(wait)
    raise RuntimeError(f"Gemini 调用失败: {last_err}")


# ---------------------------------------------------------------- 三段式调用

def make_doc_card(doc_id: str, pages_meta: dict) -> dict:
    """Pass 1：通读全部页图 -> 文档摘要卡 doc_card + doc_name（不含 toc）。"""
    parts = [{"text": DOC_CARD_PROMPT.format(
        n_pages=pages_meta["page_count"], mfr=config.MANUFACTURER)}]
    for p in pages_meta["pages"]:
        parts.append(_img_part(config.PAGES_DIR / doc_id / p["image"]))
    return gemini_json(parts)


def group_page(doc_id: str, doc_card: dict, lpage: dict, n_pages: int) -> list:
    """Pass 2：单页元素分组成板块 + 逐块描述（不分配 toc_path）。"""
    pno = lpage["page"]
    inventory = "\n".join(
        f"{i} | {e['label']} | {json.dumps(e.get('native_text', '')[:200])}"
        for i, e in enumerate(lpage["elements"]))
    prompt = GROUP_PROMPT.format(
        doc_card=json.dumps(doc_card, ensure_ascii=False),
        page=pno, n_pages=n_pages, inventory=inventory, mfr=config.MANUFACTURER,
        series=doc_card.get("series", ""), ptype=doc_card.get("product_type", ""),
        block_types=", ".join(BLOCK_TYPES))
    parts = [{"text": prompt},
             _img_part(config.PAGES_DIR / doc_id / f"p{pno}.png"),
             _img_part(config.LAYOUT_DIR / "overlays" / doc_id
                       / f"p{pno}_elements.png")]
    return gemini_json(parts).get("blocks", [])


def build_toc(doc_id: str, doc_card: dict, digest: list) -> dict:
    """Pass 3：由全部 chunk（按阅读序的清单）归纳 toc + 给每块分配 toc_path。"""
    prompt = TOC_PROMPT.format(
        mfr=config.MANUFACTURER, doc_name=doc_card.get("doc_name", "") or doc_id,
        digest="\n".join(digest))
    return gemini_json([{"text": prompt}])


# ---------------------------------------------------------------- 校验与落盘

def validate_groups(vlm_blocks: list, n_elements: int) -> list:
    """元素号去重 + 漏分配补单元素板块；返回净化后的分组。"""
    seen = set()
    cleaned = []
    for vb in vlm_blocks:
        ids = [i for i in vb.get("elements", [])
               if isinstance(i, int) and 0 <= i < n_elements and i not in seen]
        if not ids:
            continue
        seen.update(ids)
        btype = vb.get("block_type", "text_section")
        if btype not in BLOCK_TYPES:
            btype = "text_section"
        cleaned.append({"elements": ids, "block_type": btype,
                        "section_title": vb.get("section_title", "") or "",
                        "description": vb.get("description", "") or "",
                        "keywords": vb.get("keywords", []) or []})
    for i in range(n_elements):  # VLM 漏分配的元素兜底成独立板块
        if i not in seen:
            cleaned.append({"elements": [i], "block_type": "text_section",
                            "section_title": "",
                            "description": "", "keywords": [],
                            "_unassigned": True})
    return cleaned


def split_discontiguous(groups: list, elems: list, page_h: float) -> list:
    """几何防线：组内成员若在垂直方向断开超过 18% 页高（如页眉+页脚被并为
    一组），按垂直连续性拆成子组。description 留给最大的子组。"""
    max_gap = page_h * 0.18
    out = []
    for g in groups:
        ids = sorted(g["elements"], key=lambda i: elems[i]["bbox_px"][1])
        runs, cur, cur_bottom = [], [], None
        for i in ids:
            y0, y1 = elems[i]["bbox_px"][1], elems[i]["bbox_px"][3]
            if cur and y0 - cur_bottom > max_gap:
                runs.append(cur)
                cur, cur_bottom = [], None
            cur.append(i)
            cur_bottom = y1 if cur_bottom is None else max(cur_bottom, y1)
        if cur:
            runs.append(cur)
        if len(runs) == 1:
            out.append(g)
            continue
        biggest = max(range(len(runs)),
                      key=lambda k: sum(
                          (elems[i]["bbox_px"][3] - elems[i]["bbox_px"][1])
                          for i in runs[k]))
        for k, run in enumerate(runs):
            sub = dict(g)
            sub["elements"] = run
            if k != biggest:
                sub = {**sub, "description": "", "keywords": []}
            out.append(sub)
    return out


def process_doc(doc_id: str) -> dict:
    """跑完整三段式，落盘 blocks/{doc}.json + descriptions/{doc}.json。"""
    layout = json.loads(
        (config.LAYOUT_DIR / f"{doc_id}.json").read_text(encoding="utf-8"))
    pages_meta = json.loads(
        (config.PAGES_DIR / doc_id / "pages.json").read_text(encoding="utf-8"))
    pmeta_by_no = {p["page"]: p for p in pages_meta["pages"]}

    crops_dir = config.BLOCKS_DIR / "crops" / doc_id
    overlay_dir = config.BLOCKS_DIR / "overlays" / doc_id
    for d in (crops_dir, overlay_dir):  # 清掉上一轮旧产物，避免 stale 混杂
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)

    print(f"[vlm] {doc_id}: pass1 文档摘要卡 ...")
    doc_card = make_doc_card(doc_id, pages_meta)
    doc_name = (doc_card.get("doc_name") or doc_card.get("title")
                or doc_id).strip()

    # Pass2：逐页分组+描述，先不定 toc_path
    all_blocks, metas = [], []   # metas[i] 与 all_blocks[i] 同序：{g, has_graphic}
    for lpage in layout["pages"]:
        pno = lpage["page"]
        pmeta = pmeta_by_no[pno]
        scale = 72.0 / pmeta["render_dpi"]
        print(f"[vlm] {doc_id}: pass2 第 {pno} 页分组+描述 ...")
        groups = validate_groups(
            group_page(doc_id, doc_card, lpage, pages_meta["page_count"]),
            len(lpage["elements"]))
        groups = split_discontiguous(groups, lpage["elements"],
                                     pmeta["height_px"])
        n_un = sum(1 for g in groups if g.get("_unassigned"))
        if n_un:
            print(f"    [warn] {n_un} 个元素未被 VLM 分配，已补为独立板块")

        elems = lpage["elements"]
        groups.sort(key=lambda g: min(elems[i]["bbox_px"][1]    # 按最小 y 排阅读序
                                      for i in g["elements"]))
        img = Image.open(config.PAGES_DIR / doc_id / pmeta["image"])
        page_blocks, overlay_boxes = [], []
        for g in groups:
            # chunk = 成员元素的集合：bbox 不做并集，每个元素保留自己的
            # bbox 和裁剪图；native_text 按元素顺序拼接；description 写整体
            members = [elems[i] for i in g["elements"]]
            bid = f"{doc_id}_p{pno}_b{len(page_blocks):02d}"
            crop_paths, bboxes_px = [], []
            for k, m in enumerate(members):
                bb = m["bbox_px"]
                padded = [max(0, bb[0] - CROP_PAD), max(0, bb[1] - CROP_PAD),
                          min(img.width, bb[2] + CROP_PAD),
                          min(img.height, bb[3] + CROP_PAD)]
                crop_path = crops_dir / f"{bid}_e{k}.png"
                img.crop([int(v) for v in padded]).save(crop_path)
                crop_paths.append(str(crop_path.relative_to(config.DATA_DIR)))
                bboxes_px.append([round(v, 1) for v in bb])
            has_graphic = any(elems[i]["label"] in ("figure", "table")
                              for i in g["elements"])
            block = {
                "block_id": bid, "doc_id": doc_id, "page": pno,
                "reading_order": len(page_blocks),
                "block_type": g["block_type"],
                "section_title": g["section_title"],
                "toc_path": "", "toc_redundant": False,   # Pass3 后回填
                "bboxes_px": bboxes_px,
                "bboxes_pdf": [[round(v * scale, 2) for v in bb]
                               for bb in bboxes_px],
                "render_dpi": pmeta["render_dpi"],
                "indexable": True,                         # Pass3 后回填
                "merge_source": "vlm",
                "native_text": " ".join(
                    m.get("native_text", "") for m in members).strip(),
                "text_source": "pdf_layer",
                "crop_images": crop_paths,
                "page_image": str((config.PAGES_DIR / doc_id /
                                   pmeta["image"]).relative_to(config.DATA_DIR)),
                "elements": [
                    {"element_index": i, "label": elems[i]["label"],
                     "conf": elems[i]["conf"], "bbox_px": elems[i]["bbox_px"]}
                    for i in g["elements"]],
            }
            color = CHUNK_PALETTE[len(page_blocks) % len(CHUNK_PALETTE)]
            for k, bb in enumerate(bboxes_px):  # 同 chunk 成员同色，仅首框贴标签
                overlay_boxes.append({
                    "bbox_px": bb, "color": color, "tag": True,
                    "label": (f"b{block['reading_order']:02d}:{g['block_type']}"
                              if k == 0 else "")})
            page_blocks.append(block)
            metas.append({"g": g, "has_graphic": has_graphic})
        all_blocks += page_blocks
        draw_boxes(config.PAGES_DIR / doc_id / pmeta["image"],
                   overlay_boxes, overlay_dir / f"p{pno}_blocks.png",
                   label_key="label", color_map=None, width=4)

    # Pass3：由全部 chunk（按阅读序）归纳 toc + 给每块分配 toc_path
    print(f"[vlm] {doc_id}: pass3 由 {len(all_blocks)} 个 chunk 生成目录 ...")
    digest = []
    for idx, (b, mt) in enumerate(zip(all_blocks, metas)):
        g = mt["g"]
        digest.append(f"{idx} | p{b['page']} | {b['block_type']} | "
                      f"{(g.get('section_title') or '')[:60]} | "
                      f"{(g.get('description') or '')[:120]}")
    toc_res = build_toc(doc_id, doc_card, digest)
    assign = {}
    for a in toc_res.get("assignments", []) or []:
        try:
            assign[int(a.get("index"))] = (a.get("toc_path") or "").strip()
        except (TypeError, ValueError):
            continue

    # 回填 toc_path / toc_redundant / indexable / descs
    # 空 toc_path（非页眉脚、非兜底块）= 索引功能性块，不进索引；
    # 硬护栏：含 figure/table 的块绝不被目录替代，VLM 误判空路径时兜底挂根
    descs = {}
    for idx, (b, mt) in enumerate(zip(all_blocks, metas)):
        g = mt["g"]
        has_graphic = mt["has_graphic"]
        vlm_path = assign.get(idx, "")
        toc_redundant = (not vlm_path and b["block_type"] != "header_footer"
                         and not g.get("_unassigned") and not has_graphic)
        if vlm_path:
            b["toc_path"] = f"{doc_name} > {vlm_path}"
        elif has_graphic and b["block_type"] != "header_footer":
            b["toc_path"] = doc_name        # 图表块误判空路径：兜底挂根
        else:
            b["toc_path"] = ""
        b["toc_redundant"] = toc_redundant
        b["indexable"] = (b["block_type"] != "header_footer"
                          and not toc_redundant)
        if b["indexable"]:
            descs[b["block_id"]] = {"description": g.get("description", ""),
                                    "keywords": g.get("keywords", []),
                                    "toc_path": b["toc_path"]}

    # toc 用 Pass3(Gemini)按阅读序产出的 toc（digest 已按阅读序、index=阅读位置，
    # prompt 明确要求按阅读序排）；文档名作为根，原各级整体下移一级挂其下
    doc_card["doc_name"] = doc_name
    doc_card["toc"] = ([{"level": 1, "title": doc_name, "page": 1}] +
                       [{**t, "level": int(t.get("level", 1)) + 1}
                        for t in (toc_res.get("toc") or [])])

    (config.BLOCKS_DIR / f"{doc_id}.json").write_text(
        json.dumps({"doc_id": doc_id, "blocks": all_blocks},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    config.DESC_DIR.mkdir(parents=True, exist_ok=True)
    (config.DESC_DIR / f"{doc_id}.json").write_text(
        json.dumps({"doc_card": doc_card, "blocks": descs},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    return {"doc_card": doc_card, "blocks": all_blocks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", nargs="*")
    args = ap.parse_args()
    for doc_id in args.docs or config.DEMO_DOCS:
        result = process_doc(doc_id)
        n_idx = sum(1 for b in result["blocks"] if b["indexable"])
        types = {}
        for b in result["blocks"]:
            if b["indexable"]:
                types[b["block_type"]] = types.get(b["block_type"], 0) + 1
        n_toc = sum(1 for b in result["blocks"] if b.get("toc_redundant"))
        print(f"[vlm] {doc_id}: {len(result['blocks'])} blocks "
              f"({n_idx} indexable, {n_toc} replaced-by-toc) {types}")


if __name__ == "__main__":
    main()
