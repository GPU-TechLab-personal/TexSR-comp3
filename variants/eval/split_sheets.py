#!/usr/bin/env python3
"""Per-seed copies of a run's montage sheets, cut from the sheets montage.py wrote.

With MatUp's seeds 0-2 and four arms, a sheet has 14 rows (GT, bilinear, 4 arms x 3 seeds),
too many to compare by eye. This keeps, for each seed k, the rows without a seed (GT,
bilinear) and every seeded row of seed k (<arm>_s<k>), in the order of the full sheet:

  montage/by_seed/s<k>/material/<sheet>_s<k>.png    rows cut out of montage/material/
  montage/by_seed/s<k>/zoom/<sheet>_s<k>.png        rows cut out of montage/zoom/
  montage/by_seed/s<k>/overview/<sheet>_s<k>.png    columns cut out of montage/overview/

Pixels are copied, nothing is re-rendered, so it takes seconds and runs anywhere with numpy
and Pillow (montage.py calls it at the end; scripts/eval.sh split re-cuts existing sheets).
by_seed/ is rebuilt from scratch every time; it travels with git like the rest of montage/. A run with fewer than two seeds or no seeded row is left alone.

    python variants/eval/split_sheets.py runs/eval/matup_shared_vs_split/ep10
"""
import argparse
import json
import os
import re
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# geometry of variants/eval/sheets.py (kept in step by hand: sheets.py needs cv2)
PAD = 6                 # sheet(): gap between rows
OV_LEFT, OV_PAD, OV_TOP = 230, 4, 34    # overview(): label gutter, gap, title band
BG = 18
SEP = (95, 95, 95)
SEEDED = re.compile(r"^(.+)_s(\d+)$")


def plan(rows, seed):
    """Indices of the rows kept for one seed, and a group id per kept row."""
    keep, groups = [], []
    for i, k in enumerate(rows):
        m = SEEDED.match(k)
        if m is None or int(m.group(2)) == seed:
            keep.append(i)
            groups.append(k if m is None else "seeded")
    return keep, groups


def cut_rows(a, n_rows, cell, keep, groups):
    """sheet(): methods down. a = HxWx3; the rows end PAD above the bottom edge."""
    y0 = a.shape[0] - PAD - n_rows * (cell + PAD)            # title + notes + column heads
    if y0 <= 0 or a.shape[0] != y0 + n_rows * (cell + PAD) + PAD:
        raise SystemExit("unexpected sheet height %d for %d rows of %d" % (a.shape[0], n_rows, cell))
    parts = [a[:y0]]
    for r, i in enumerate(keep):
        if r:
            gap = np.full((PAD, a.shape[1], 3), BG, np.uint8)
            if groups[r] != groups[r - 1]:                     # as sheets.separators
                gap[PAD // 2 - 1:PAD // 2 + 1, 6:-6] = SEP
            parts.append(gap)
        y = y0 + i * (cell + PAD)
        parts.append(a[y:y + cell])
    parts.append(np.full((PAD, a.shape[1], 3), BG, np.uint8))
    return np.concatenate(parts)


def cut_columns(a, n_cols, keep, groups, title):
    """overview(): methods across. Returns the image and the title to draw."""
    cell = (a.shape[1] - OV_LEFT - OV_PAD) // n_cols - OV_PAD
    if OV_LEFT + n_cols * (cell + OV_PAD) + OV_PAD != a.shape[1]:
        raise SystemExit("unexpected overview width %d for %d columns" % (a.shape[1], n_cols))
    parts = [a[:, :OV_LEFT].copy()]
    for c, i in enumerate(keep):
        x = OV_LEFT + i * (cell + OV_PAD)
        col = a[:, x:x + cell + OV_PAD].copy()
        col[OV_TOP:, cell:] = BG                               # trailing gap: the old separator
        if c and groups[c] != groups[c - 1]:                   # new one in the previous gap
            parts[-1][OV_TOP + 4:-4, -OV_PAD + 1:-OV_PAD + 3] = SEP
        parts.append(col)
    out = np.ascontiguousarray(np.concatenate(parts, axis=1))
    out[:OV_TOP] = BG                                          # the full title no longer fits
    return out


def font(size):
    try:
        return ImageFont.load_default(size=size)              # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def save(path, a, title=None):
    im = Image.fromarray(a)
    if title:
        ImageDraw.Draw(im).text((10, 8), title, fill=(255, 255, 255), font=font(18))
    im.save(path, optimize=True)


def split_run(run_dir, log=print):
    mont = os.path.join(run_dir, "montage")
    meta = json.load(open(os.path.join(mont, "montage.json")))
    rows, seeds = meta["rows"], meta.get("seeds") or []
    out_root = os.path.join(mont, "by_seed")
    if os.path.isdir(out_root):
        shutil.rmtree(out_root)
    if len(seeds) < 2 or not any(SEEDED.match(k) for k in rows):
        return 0
    cells = {"material": meta["cell"], "zoom": meta["zoom_cell"]}
    n = 0
    for k in seeds:
        keep, groups = plan(rows, k)
        for kind in ("material", "zoom", "overview"):
            src_dir = os.path.join(mont, kind)
            if not os.path.isdir(src_dir):
                continue
            dst_dir = os.path.join(out_root, "s%d" % k, kind)
            os.makedirs(dst_dir, exist_ok=True)
            for f in sorted(os.listdir(src_dir)):
                if not f.endswith(".png"):
                    continue
                a = np.asarray(Image.open(os.path.join(src_dir, f)).convert("RGB"))
                dst = os.path.join(dst_dir, f[:-4] + "_s%d.png" % k)
                if kind == "overview":
                    what = f[:-4].split("_overview_")[-1]
                    save(dst, cut_columns(a, len(rows), keep, groups, None),
                         "%s  -  seed %d  -  %s" % (what, k, meta["name"]))
                else:
                    save(dst, cut_rows(a, len(rows), cells[kind], keep, groups))
                n += 1
    log("[split] %d sheet(s), seeds %s -> %s" % (n, ",".join(map(str, seeds)), out_root))
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir", help="runs/eval/<experiment>/<checkpoint> (has montage/montage.json)")
    split_run(ap.parse_args().run_dir)


if __name__ == "__main__":
    main()
