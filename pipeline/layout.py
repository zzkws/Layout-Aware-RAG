"""Stage 2: DocLayout-YOLO element detection and box de-duplication.

Reads the stage-1 page images and writes, per page, element boxes in both
coordinate systems (pixels and PDF points), plus an element-level overlay image
for inspection.

The raw DocLayout-YOLO output contains duplicates: co-located double boxes (one
table detected twice) and same-class nesting (a small figure boxed inside a
large one). De-duplication runs immediately after detection and **ranks by area
-- the box containing more content wins -- falling back to confidence only on a
tie**. This matters because YOLO frequently assigns *lower* confidence to the
box with more content: in document 6u the complete table_footnote box scores
0.37 against 0.73 for the truncated one, so confidence-first ranking silently
cuts off the start of the Note line.

  1. any class, near-identical boxes (IoU > 0.85)      -> keep the larger
  2. same class, high overlap (IoU > 0.55)             -> keep the larger
  3. same class, 85% of the smaller inside the larger  -> keep the larger

Cross-class nesting -- for example plain_text annotations inside a figure -- is
real content and is preserved, to be resolved downstream by the VLM grouping
pass.

Weights download from Hugging Face automatically, falling back to
hf-mirror.com when the direct connection fails.

Usage:
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
        print(f"[layout] direct Hugging Face fetch failed ({e}); retrying via hf-mirror.com")
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
    """De-duplicate detection boxes; the rules are in the module docstring.

    A greedy pass sorted by (area descending, confidence descending): larger
    boxes are taken first, and a later, smaller box is dropped when it overlaps
    a kept box beyond the IoU threshold, or when it is same-class and 85%
    contained by one. Because the sort is area-descending, dropping the
    candidate is exactly "keep the box that covers more content"."""
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
                dup = True  # e comes later, so its area <= k: containment means duplicate
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
        # Sort by y then x for readability; true reading order comes from the VLM grouping
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
    ap.add_argument("--docs", nargs="*", help="explicit list of doc_ids")
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
