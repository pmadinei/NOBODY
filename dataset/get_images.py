"""Copy the NOBODY images from a local V3Det copy into images/ (the images are not redistributed with NOBODY).

usage: python get_images.py --v3det-root /path/to/V3Det [--json nobody.json] [--out images]

`--v3det-root` is the folder of the official V3Det release that contains `images/` (image paths in the release
look like images/<category>/<file>.jpg). Every pair of nobody.json names its image by `image_file`, the path
inside that release; the script copies each image to <out>/<image_file without the leading "images/"> and reports
the images it could not find.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3det-root", required=True)
    ap.add_argument("--json", default=str(HERE / "nobody.json"))
    ap.add_argument("--out", default=str(HERE / "images"))
    a = ap.parse_args()
    root, out = Path(a.v3det_root), Path(a.out)
    files = sorted({p["image_file"] for p in json.loads(Path(a.json).read_text(encoding="utf-8"))["pairs"]})
    missing = []
    for f in files:
        rel = Path(f).relative_to("images") if f.startswith("images/") else Path(f)
        src = next((s for s in (root / f, root / "images" / rel, root / rel) if s.exists()), None)
        if src is None:
            missing.append(f)
            continue
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copyfile(src, dst)
    print(f"{len(files) - len(missing)} of {len(files)} images in {out}")
    if missing:
        print(f"{len(missing)} images not found under {root}, e.g. {missing[0]}")


if __name__ == "__main__":
    main()
