"""bbox 可视化工具：在页图上画元素框/板块框。"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# 每个类别一个固定颜色，元素级（DocLayout-YOLO 原始输出）
ELEMENT_COLORS = {
    "title": "#e6194b",
    "plain_text": "#3cb44b",
    "abandon": "#9a9a9a",
    "figure": "#4363d8",
    "figure_caption": "#42d4f4",
    "table": "#f58231",
    "table_caption": "#ffe119",
    "table_footnote": "#911eb4",
    "isolate_formula": "#f032e6",
    "formula_caption": "#a9a9a9",
}
BLOCK_COLOR = "#d62728"  # 合并后大板块统一红色粗框


def _font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_boxes(page_image: Path, boxes: list, out_path: Path,
               label_key: str = "label", color_map: dict | None = None,
               width: int = 3):
    """boxes: [{"bbox_px": [x0,y0,x1,y1], label_key: str, "conf": float?}, ...]"""
    img = Image.open(page_image).convert("RGB")
    drw = ImageDraw.Draw(img)
    font = _font(max(14, img.width // 90))
    for i, b in enumerate(boxes):
        x0, y0, x1, y1 = b["bbox_px"]
        label = str(b.get(label_key, "?"))
        if "color" in b:  # 调用方指定颜色（如同一 chunk 的成员框同色）
            color = b["color"]
        else:
            color = (
                (color_map or {}).get(label, BLOCK_COLOR)
                if color_map is not None
                else BLOCK_COLOR
            )
        drw.rectangle([x0, y0, x1, y1], outline=color, width=width)
        if not label:  # 空 label 只画框不贴标签
            continue
        tag = label if "tag" in b else f"{i}:{label}"
        if "conf" in b:
            tag += f" {b['conf']:.2f}"
        tw = drw.textlength(tag, font=font)
        ty = max(0, y0 - font.size - 4)
        drw.rectangle([x0, ty, x0 + tw + 6, ty + font.size + 4], fill=color)
        drw.text((x0 + 3, ty + 2), tag, fill="white", font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
