"""Cut the background out of every image so only the gift itself is compared.

Reads:   data/human/*  and  data/ai/*
Writes:  data_nobg/human/*.png  and  data_nobg/ai/*.png

Each gift is cut out, cropped to its own outline, and centred on a plain
white square, so room, lighting and framing no longer differ between sets.

Install once:   pip install "rembg[cpu]"
Run:            python remove_backgrounds.py
Other folders:  python remove_backgrounds.py data/ai_category data/ai_open
                (writes data_nobg/ai_category and data_nobg/ai_open)
The first run downloads the cut-out model (about 170 MB).
"""
import sys
from pathlib import Path

from PIL import Image
from rembg import new_session, remove

SOURCES = {"human": Path("data/human"), "ai": Path("data/ai")}
OUT_ROOT = Path("data_nobg")
EXTS = {".jpg", ".jpeg", ".png", ".webp"}
SIZE = 512        # final square size in pixels
MARGIN = 0.06     # white border around the gift, as a fraction of SIZE


def cut_out(img, session):
    rgba = remove(img.convert("RGB"), session=session)  # transparent background
    box = rgba.getchannel("A").getbbox()
    if box is None:                                     # nothing found: keep the original
        return img.convert("RGB")
    rgba = rgba.crop(box)
    inner = int(SIZE * (1 - 2 * MARGIN))
    rgba.thumbnail((inner, inner), Image.LANCZOS)
    canvas = Image.new("RGB", (SIZE, SIZE), "white")
    offset = ((SIZE - rgba.width) // 2, (SIZE - rgba.height) // 2)
    canvas.paste(rgba, offset, mask=rgba.getchannel("A"))
    return canvas


def main():
    sources = {Path(a).name: Path(a) for a in sys.argv[1:]} or SOURCES
    session = new_session("u2net")
    for name, folder in sources.items():
        out_dir = OUT_ROOT / name
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTS)
        for p in files:
            cut_out(Image.open(p), session).save(out_dir / f"{p.stem}.png")
            print(f"{name}: {p.name} done")
    print(f"Finished. Check the images in {OUT_ROOT} before analysing.")


if __name__ == "__main__":
    main()