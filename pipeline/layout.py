"""Stage 2: DocLayout-YOLO 版面元素检测 + 检测框去重。

输入 Stage 1 的页图，输出每页元素框（像素坐标 + PDF 点坐标双坐标系），
并生成元素级 overlay 可视化图。

DocLayout-YOLO 原始输出存在重复框：同位双框（同一个表两个框）、
同类大框包小框（大 figure 里又框一个小 figure）。检测后立即去重，
**保留依据是面积（框住更多内容者胜），面积相同才比置信度**——YOLO 常给
内容更全的大框更低的置信度（如 6u 的 table_footnote 大框 0.37 / 小框 0.73，
按置信度保留会把 Note 行首裁掉）：
  1. 任意类别近似同框（IoU > 0.85）       -> 保留大框
  2. 同类别高重叠（IoU > 0.55）           -> 保留大框
  3. 同类别大框包小框（小框 85% 在大框内）-> 保留大框
跨类别的包含关系（如 figure 内的 plain_text 标注文字）是真实内容，保留，
交给下游 VLM 分组决策。

权重自动从 HuggingFace 下载（网络不通时自动切 hf-mirror.com）。

用法:
    python pipeline/layout.py [--docs 6u 7m ...]
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from common.draw import ELEMENT_COLORS, draw_boxes


def get_model_weights() -> Path:
    local = config.MODELS_DIR / config.LAYOUT_MODEL_FILE
    if local.exists():
        return local
    from huggingface_hub import hf_hub_download
    try:
        path = hf_hub_download(config.LAYOUT_MODEL_REPO, config.LAYOUT_MODEL_FILE,
                               local_dir=config.MODELS_DIR)
    except Exception as e:
        print(f"[layout] HF 直连失败({e})，切换 hf-mirror.com 重试")
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        path = hf_hub_download(config.LAYOUT_MODEL_REPO, config.LAYOUT_MODEL_FILE,
                               local_dir=config.MODELS_DIR)
    return Path(path)


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a, b):
    return _area([max(a[0], b[0]), max(a[1], b[1]),
                  min(a[2], b[2]), min(a[3], b[3])])


def dedup_elements(elements: list) -> list:
    """检测框去重，规则见模块 docstring。

    按 (面积降序, 置信度降序) 贪心：先收大框，后来的小框若与已留框
    高重叠（IoU 超阈值）或同类被包含（85% 面积落在已留框内），即为
    重复，丢弃——天然实现"保留框住更多内容的那个"。"""
    survivors = []
    for e in sorted(elements,
                    key=lambda x: (-_area(x["bbox_px"]), -x["conf"])):
        dup = False
        for k in survivors:
            i = _inter(e["bbox_px"], k["bbox_px"])
            iou = i / (_area(e["bbox_px"]) + _area(k["bbox_px"]) - i + 1e-6)
            same = e["label"] == k["label"]
            if iou > (0.55 if same else 0.85):
                dup = True
                break
            if same and i / (_area(e["bbox_px"]) + 1e-6) > 0.85:
                dup = True  # e 是后来者，面积必不大于 k：被包含即重复
                break
        if not dup:
            survivors.append(e)
    return survivors


def detect_doc(model, doc_id: str) -> dict:
    pages_meta = json.loads(
        (config.PAGES_DIR / doc_id / "pages.json").read_text(encoding="utf-8"))
    out_dir = config.LAYOUT_DIR
    overlay_dir = out_dir / "overlays" / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)

    result_pages = []
    for pmeta in pages_meta["pages"]:
        img_path = config.PAGES_DIR / doc_id / pmeta["image"]
        res = model.predict(str(img_path), imgsz=config.LAYOUT_IMGSZ,
                            conf=config.LAYOUT_CONF, device="cpu", verbose=False)[0]
        scale = 72.0 / pmeta["render_dpi"]  # px -> pt
        elements = []
        for box in res.boxes:
            x0, y0, x1, y1 = [round(float(v), 1) for v in box.xyxy[0].tolist()]
            cls_id = int(box.cls[0])
            elements.append({
                "label": config.LAYOUT_CLASSES.get(cls_id, f"cls{cls_id}"),
                "conf": round(float(box.conf[0]), 3),
                "bbox_px": [x0, y0, x1, y1],
                "bbox_pdf": [round(v * scale, 2) for v in (x0, y0, x1, y1)],
            })
        n_raw = len(elements)
        elements = dedup_elements(elements)
        # 按 y 再 x 排序，方便人工查看（真正的阅读顺序由 VLM 分组决定）
        elements.sort(key=lambda e: (e["bbox_px"][1], e["bbox_px"][0]))
        result_pages.append({"page": pmeta["page"], "elements": elements,
                             "n_raw_detections": n_raw,
                             "n_deduped": n_raw - len(elements)})
        draw_boxes(img_path, elements,
                   overlay_dir / f"p{pmeta['page']}_elements.png",
                   label_key="label", color_map=ELEMENT_COLORS, width=2)

    result = {"doc_id": doc_id, "model": config.LAYOUT_MODEL_FILE,
              "imgsz": config.LAYOUT_IMGSZ, "conf_threshold": config.LAYOUT_CONF,
              "pages": result_pages}
    (out_dir / f"{doc_id}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", nargs="*", help="指定 doc_id 列表")
    args = ap.parse_args()
    doc_ids = args.docs or config.DEMO_DOCS

    weights = get_model_weights()
    from doclayout_yolo import YOLOv10
    model = YOLOv10(str(weights))

    for doc_id in doc_ids:
        result = detect_doc(model, doc_id)
        n = sum(len(p["elements"]) for p in result["pages"])
        n_dd = sum(p["n_deduped"] for p in result["pages"])
        print(f"[layout] {doc_id}: {len(result['pages'])} pages, "
              f"{n} elements ({n_dd} duplicates removed)")


if __name__ == "__main__":
    main()
