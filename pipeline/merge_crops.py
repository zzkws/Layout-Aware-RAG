"""Stage ④.5: chunk 成员裁剪图合并（vlm_blocks 之后运行）。

每个可索引 chunk 的多张成员裁剪图，保持原始像素（同一 DPI 渲染下即同一
PPI，不缩放），按阅读顺序自上而下拼接成一张图：画布宽 = 成员最大宽，
窄图左对齐、右侧白色留白，成员之间 12px 白色分隔。

产出按约定路径存放（webapp 证据分析按此约定直接取图，每 chunk 一张）：
    data/blocks/merged/{doc}/{block_id}.png

用法:
    python pipeline/merge_crops.py [--docs 6u 7m ...] [--all]
"""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

GAP = 12   # 成员之间的白色分隔（px）


def merge_chunk_crops(doc_id: str) -> int:
    blocks = json.loads(
        (config.BLOCKS_DIR / f"{doc_id}.json").read_text(encoding="utf-8"))
    out_dir = config.BLOCKS_DIR / "merged" / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for b in blocks["blocks"]:
        if not b["indexable"]:
            continue
        imgs = [Image.open(config.DATA_DIR / p.replace("\\", "/"))
                for p in b["crop_images"]]
        if not imgs:
            continue
        width = max(im.width for im in imgs)
        height = sum(im.height for im in imgs) + GAP * (len(imgs) - 1)
        canvas = Image.new("RGB", (width, height), "white")
        y = 0
        for im in imgs:
            canvas.paste(im.convert("RGB"), (0, y))   # 左对齐，右侧留白
            y += im.height + GAP
        canvas.save(out_dir / f"{b['block_id']}.png")
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", nargs="*")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if args.all:
        doc_ids = sorted(p.stem for p in config.BLOCKS_DIR.glob("*.json"))
    else:
        doc_ids = args.docs or config.DEMO_DOCS
    total = 0
    for doc_id in doc_ids:
        total += merge_chunk_crops(doc_id)
    print(f"[merge_crops] {len(doc_ids)} docs, {total} merged images")


if __name__ == "__main__":
    main()
