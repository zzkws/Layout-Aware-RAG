"""Stage 5: compose one evidence image per chunk (runs after vlm_blocks).

The member crops of each indexable chunk are stacked top to bottom in reading
order at native pixel size -- no rescaling, so every crop from one render shares
a PPI. Canvas width is the widest member; narrower crops are left-aligned with
white padding on the right, and members are separated by 12 px of white.

Output goes to a conventional path, one image per chunk, which the webapp reads
directly:
    data/blocks/merged/{doc}/{block_id}.png

Usage:
    python pipeline/merge_crops.py [--docs 6u 7m ...] [--all]
"""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

GAP = 12   # white separator between members, in px


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
            canvas.paste(im.convert("RGB"), (0, y))   # left-aligned, padded right
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
