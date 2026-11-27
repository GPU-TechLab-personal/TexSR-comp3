#!/usr/bin/env python3
"""Montages for eyeballing every arm against GT, before (or without) the metrics.

Runs in the matup container (torch, MatUp's renderer, cv2); scripts/eval.sh montage
runs

    python /workspace/variants/eval/montage.py --name deeptex_shared_vs_split/40k --only deeptex [--seeds 0] [--material M]

after `bash scripts/eval.sh infer` has written the DeepTextures arm folders. It does
NOT need `score`: bilinear and MatUp are read from their sources, exactly as
evaluate.py reads them (evaluate.available_columns / load_column / measure_border).

Port of comp1 variants/deeptex/render_loss/eval/deeptex_dt_montage.py (methods
down, maps and renders across, wrapped labels, group separators, an overview per
map) on comp3's evaluation code, with the comp3 conventions:

  HEIGHT   every arm is rendered with the GT height (evaluate.py, SPLIT_THEORY C-3),
           so there is no height column.
  CROP     MatUp's zero rim, measured on the MatUp runs' _opt maps
           (evaluate.measure_border; the largest when runs differ, 8 without a run),
           crops every row, the GT included.
  MAPS     the stored PNGs, displayed without rescaling (sheets.map_cell): an
           all-zero metallic is black, a normal shows its OpenGL RGB encoding.
           MatUp's normal_opt (DirectX) gets its G flipped for display, as
           evaluate.write_maps does; roughness_opt is sqrt(alpha) = roughness.
  RENDERS  MatUp's renderer through render_core.prepare / eval_image (what every
           render metric uses): EVAL_LIGHT (67 deg) and the grazing held-out light
           of the render sheets (render_core.build_lights()["sheet"]["grazing"]).
  ERROR    |render - GT render| under EVAL_LIGHT, x HEAT_GAIN (4: 0.25 saturates),
           inferno, GT-invalid texels grey. The PSNR in its corner is that render's
           over the GT-valid texels of the cell (the whole crop, or the zoom window).

EXPERIMENTS  one set of sheets each, in its own folder <dir> under montage/ and with
  <exp> starting every file name. DeepTex is one experiment; MatUp is one per training
  setting (one sheet with every MatUp arm and seed was 32 rows), with seed k's rows for
  each --seeds k (default 0 = MatUp as released):
  <exp>  (<dir>)                      rows after GT | bilinear
  deeptex_shared_vs_split (montage/)  DeepTex shared (renloss_21), split nb23, nb7, nb7 B=8,
                                      nb23 / nb7 RM grey (each when its maps exist)
  matup_width (width/)                MatUp shared MLP, split MLP (1.32x), split 1x, split 3x
  matup_noreg (noreg/)                shared, split | shared, split without the metallic reg
  matup_zinit (zinit/)                shared, split | shared, split with zero-init last layer
  The variant sheets repeat the released pair (shared, split) as the reference, above a
  separator. An arm with no output at all is left out (warning); one missing a material
  is left out of that material's sheets (--keep-missing draws an n/a row instead). An
  experiment with none of its own arms (the reference does not count) writes nothing.

OUTPUTS  <eval-root>/<name>/montage/<dir>/{material,zoom,overview}/, one folder per kind
         of sheet (montage/{material,...}/ for DeepTex). scripts/eval.sh gives each
         family its own <name> = <family>/<checkpoint> and --only, one family per folder
  material/<exp>_<material>.png        the whole crop; the zoom window outlined in yellow
  zoom/<exp>_<material>_zoom.png       the same rows and columns on the zoom window, texels
                              enlarged (nearest). Window: --zoom texels square, at
                              --zoom-at y,x (top-left, crop coordinates) or, by
                              default, where the GT EVAL_LIGHT render has the most
                              |grad| (sheets.busiest_window); the same for both
                              experiments.
  overview/<exp>_overview_<key>.png    materials down, rows across, for key in baseColor,
                              normal, roughness, metallic, render (EVAL_LIGHT), error
  montage.json (in montage/)  sheets: per <dir>, its <exp> and row keys; per material:
                              border, crop, zoom window, sources, EVAL PSNR (whole crop)
                              per row, missing rows, files ({path relative to montage/:
                              its row keys})
  <dir>/by_seed/s<k>/<kind>/  with two or more --seeds: every sheet of <dir> again with
                              GT, bilinear and seed k's rows only (split_sheets.py)
  Earlier layouts' sheets are removed on every run: the single MatUp set
  montage/<kind>/matup_shared_vs_split_* with montage/by_seed/ (before the settings),
  montage/<exp>/<kind>/ and the flat montage/<exp>_*.png, and the first version's
  <material>.png, <material>_zoom.png, overview_<key>.png.
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                *[os.pardir] * 2)))
from variants.eval import render_core as rc  # noqa: E402  (puts MatUp on sys.path first)

import torch  # noqa: E402
from matup.renderer import Renderer  # noqa: E402

from variants.eval import arms as A  # noqa: E402
from variants.eval import evaluate as E  # noqa: E402
from variants.eval import sheets as S  # noqa: E402
from variants.eval import split_sheets  # noqa: E402
from variants.eval.matup_maps import crop  # noqa: E402
from variants.util.report import images as rimg  # noqa: E402

die = rc.die
MAP4 = rc.MAP4                                     # baseColor, normal, roughness, metallic
OV_KEYS = list(MAP4) + ["render", "error"]
HEADS = list(MAP4) + ["render EVAL_LIGHT (67 deg)", "render held-out grazing",
                      "|render - GT| x%g (EVAL)" % S.HEAT_GAIN]
GROUP = {"gt": "gt", "bilinear": "bilinear", "deeptex": "deeptex", "matup": "matup"}
SHARED_GROUPS = ("gt", "bilinear")                 # head every experiment's sheets
MATUP_REF = ("matup_shared", "matup_split")        # MatUp as released: the variants' reference
# One set of sheets per experiment. dir: its folder under montage/ ("" = montage/ itself);
# prefix: starts every file name; kind: the arms' kind; arms: its own arms (None = every
# arm of the kind); ref: arms repeated above a separator as the reference.
EXPERIMENTS = (
    {"dir": "", "prefix": "deeptex_shared_vs_split", "kind": "deeptex", "arms": None, "ref": (),
     "title": "DeepTextures: shared vs split"},
    {"dir": "width", "prefix": "matup_width", "kind": "matup", "ref": (),
     "arms": ("matup_shared", "matup_split", "matup_split_x1", "matup_split_x3"),
     "title": "MatUp: shared vs split MLP (1.32x, 1x, 3x the parameters)"},
    {"dir": "noreg", "prefix": "matup_noreg", "kind": "matup", "ref": MATUP_REF,
     "arms": ("matup_shared_noreg", "matup_split_noreg"),
     "title": "MatUp: released | no metallic regulariser"},
    {"dir": "zinit", "prefix": "matup_zinit", "kind": "matup", "ref": MATUP_REF,
     "arms": ("matup_shared_zinit", "matup_split_zinit"),
     "title": "MatUp: released | zero-init last layer"},
)
LEGACY_PREFIXES = ("deeptex_shared_vs_split", "matup_shared_vs_split")   # earlier layouts
_unsheeted = set(A.MATUP_ARMS) - {a for e in EXPERIMENTS if e["kind"] == "matup"
                                  for a in e["arms"]}
assert not _unsheeted, "MatUp arms on no montage sheet: %s" % sorted(_unsheeted)


def label(col):
    return col["display"] if col["kind"] != "gt" else "GT"


def arm_of(col):
    return col.get("arm", col["key"])


def masked_psnr(img, gt, mask_hw):
    d = (img.astype(np.float64) - gt.astype(np.float64))[mask_hw]
    return rc.psnr(float((d * d).mean())) if d.size else None


def psnr_text(v):
    return "PSNR inf" if v is None else "PSNR %.2f dB" % v


def display_maps(stored, kind, border):
    """Stored maps -> cropped arrays ready for sheets.map_cell (MatUp normal -> OpenGL)."""
    out = {}
    for m in MAP4:
        a = stored[m]
        if m == "normal" and kind == "matup":
            a = rimg.to_u16(a).copy()
            a[..., 1] = rimg.U16_MAX - a[..., 1]      # DirectX -> OpenGL (BGR: G is 1)
        out[m] = crop(a, border)
    return out


def win(a, box):
    y0, x0, z = box
    return a[y0:y0 + z, x0:x0 + z]


def cells_for(row, gt_eval, mask_hw, side, size, box=None):
    """One sheet row: four maps, two renders, the heat map (or the GT note)."""
    cut = (lambda a: win(a, box)) if box else (lambda a: a)
    mk = cut(mask_hw)
    cells = [S.map_cell(cut(row["maps"][m]), size) for m in MAP4]
    cells += [S.render_cell(cut(row["eval"]), size), S.render_cell(cut(row["graz"]), size)]
    if row["kind"] == "gt":
        cells.append(S.text_cell(size, ["GT = reference", "no |render - GT|"]))
    else:
        v = masked_psnr(cut(row["eval"]), cut(gt_eval), mk)
        cells.append(S.heat_cell(cut(row["eval"]), cut(gt_eval), mk, size, psnr_text(v)))
    return cells


def render_two(maps4, gt_pos, border, lights, device, where):
    """(EVAL_LIGHT render, grazing held-out render), float RGB side x side."""
    rmaps, _ = rc.prepare(maps4 + [gt_pos], border, device)
    j = lights["sheet"]["grazing"]
    ev = rc.eval_image(rmaps, lights["eval"], where + " eval")
    gr = rc.eval_image(rmaps, lights["held"][j:j + 1], where + " grazing")
    del rmaps
    return ev, gr


# --------------------------------------------------------------- per material

def process(mat_dir, cols, args, device, lights):
    """One material -> (info, rows, GT-valid mask). rows: [{key, label, group, kind,
    maps, eval, graz}] with None in place of a row whose arm has no output here."""
    t0 = time.time()
    name = os.path.basename(mat_dir.rstrip("/"))
    rc.require_readable([os.path.join(mat_dir, sub, m + ".png")
                         for sub in ("GT", "SVBRDF") for m in rc.SLOT], name)

    runs = {c["key"]: E.newest_matup_run(args.matup_root, c["arm"], c["seed"], name)
            for c in cols if c["kind"] == "matup"}
    detected = {k: E.measure_border(r) for k, r in runs.items() if r}
    if detected:
        vals = sorted({b for b, _ in detected.values()})
        border = vals[-1]
        if len(vals) > 1:
            print("  ! %s: MatUp borders differ %s - cropping all rows by %d" % (
                name, {k: b for k, (b, _) in detected.items()}, border))
    else:
        border = E.DEFAULT_BORDER
        print("  ! %s: no MatUp run to measure the border on - using %d" % (name, border))

    gt4 = rc.load_gt(mat_dir)
    full = gt4[0].shape[0]
    side = full - 2 * border
    gt_pos = rc.gt_position(mat_dir)
    mask_hw = rc.gt_valid_mask(os.path.join(mat_dir, "GT"), border)
    gt_stored = {m: E.read_stored(os.path.join(mat_dir, "GT", m + ".png"), m) for m in MAP4}
    gt_eval, gt_graz = render_two(gt4, gt_pos, border, lights, device, "%s gt" % name)

    rows = [{"key": "gt", "arm": "gt", "label": "GT", "group": "gt", "kind": "gt",
             "maps": display_maps(gt_stored, "gt", border), "eval": gt_eval, "graz": gt_graz}]
    info = {"material": name, "border": border, "full": full, "side": side,
            "valid_frac": float(mask_hw.mean()), "matup_runs": runs, "sources": {},
            "eval_psnr": {}, "missing": []}
    for col in cols:
        key = col["key"]
        got = E.load_column(col, mat_dir, name, full, runs.get(key))
        if got is None:
            info["missing"].append(key)
            rows.append(None if not args.keep_missing else
                        {"key": key, "arm": arm_of(col), "label": label(col),
                         "group": GROUP[col["kind"]], "kind": "missing"})
            continue
        maps4, stored, src = got
        if any(t.shape[0] != full for t in maps4) or \
                any(a.shape[:2] != (full, full) for a in stored.values()):
            die(2, "%s / %s: map sizes %s, GT is %d" % (
                name, key, [tuple(t.shape) for t in maps4], full))
        ev, gr = render_two(maps4, gt_pos, border, lights, device, "%s %s" % (name, key))
        info["sources"][key] = src.get("dir")
        info["eval_psnr"][key] = masked_psnr(ev, gt_eval, mask_hw)
        rows.append({"key": key, "arm": arm_of(col), "label": label(col),
                     "group": GROUP[col["kind"]], "kind": col["kind"], "maps": display_maps(stored, col["kind"], border),
                     "eval": ev, "graz": gr})

    if args.zoom_at:
        y0, x0 = (int(v) for v in args.zoom_at.split(","))
    else:
        y0, x0 = S.busiest_window(gt_eval, mask_hw, args.zoom)
    z = min(args.zoom, side)
    y0, x0 = max(0, min(y0, side - z)), max(0, min(x0, side - z))
    info["zoom"] = {"y0": y0, "x0": x0, "size": z,
                    "rule": "--zoom-at" if args.zoom_at else "busiest GT EVAL render window"}
    print("  %-28s border=%d side=%d zoom=(%d,%d)+%d  %s  %.0fs" % (
        name, border, side, y0, x0, z,
        " ".join("%s=%.2f" % (k, v) for k, v in info["eval_psnr"].items() if v is not None),
        time.time() - t0))
    return info, rows, mask_hw


# ------------------------------------------------------------------- pictures

def in_experiment(group, arm, exp):
    """A row (its group and arm) belongs on the experiment's sheets."""
    if group in SHARED_GROUPS:
        return True
    return group == GROUP[exp["kind"]] and (
        exp["arms"] is None or arm in exp["arms"] or arm in exp["ref"])


def own_arm(group, arm, exp):
    """One of the experiment's own arms (not GT / bilinear / the reference)."""
    return group == GROUP[exp["kind"]] and (exp["arms"] is None or arm in exp["arms"])


def sep_group(group, arm, exp):
    """Separator group: the reference rows apart from the experiment's own."""
    return "ref" if group not in SHARED_GROUPS and arm in exp["ref"] else group


def remove_if(path):
    if os.path.isfile(path):
        os.remove(path)


def sheet_path(out, exp, kind, fname):
    """montage/<dir>/<kind>/<fname> (the file name starts with the experiment)."""
    d = os.path.join(out, exp["dir"], kind)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, fname)


def rel(exp, kind, fname):
    """The sheet's path relative to montage/ (montage.json)."""
    return "/".join(p for p in (exp["dir"], kind, fname) if p)


def write_material(out, info, rows, mask_hw, args):
    """One full and one zoom sheet per experiment that has a row for this material;
    returns {row key: row} for the overviews."""
    name, side = info["material"], info["side"]
    gt_eval = rows[0]["eval"]
    zm = info["zoom"]
    box = (zm["y0"], zm["x0"], zm["size"])
    present = [r for r in rows if r is not None]
    for old in (name + ".png", name + "_zoom.png"):   # the first version's file names
        remove_if(os.path.join(out, old))

    def sheet_rows(sel, size, cut, exp):
        out_rows = []
        for r in sel:
            if r["kind"] == "missing":
                cells = [S.placeholder(size, ["no output"])] * len(HEADS)
            else:
                cells = cells_for(r, gt_eval, mask_hw, side, size, cut)
            out_rows.append((r["label"], sep_group(r["group"], r["arm"], exp), cells))
        return out_rows

    notes = ["%d^2 crop (MatUp's %d px rim removed from every row). Every arm rendered with "
             "the GT height; MatUp's renderer, raw maps." % (side, info["border"]),
             "Heat: |render - GT| x%g under EVAL_LIGHT (%.2f saturates), grey = GT-invalid "
             "texels; PSNR over the cell's GT-valid texels. Yellow box = the area the _zoom "
             "sheet enlarges." % (S.HEAT_GAIN, 1.0 / S.HEAT_GAIN)]
    znotes = ["Zoom: texels [%d:%d, %d:%d] of the %d^2 crop (%s), enlarged x%.1f with "
              "nearest - every screen block is one texel." % (
                  zm["y0"], zm["y0"] + zm["size"], zm["x0"], zm["x0"] + zm["size"], side,
                  zm["rule"], args.zoom_cell / float(zm["size"])),
              notes[1].split(" Yellow box")[0]]
    info["files"] = {}
    for exp in EXPERIMENTS:
        sel = [r for r in present if in_experiment(r["group"], r["arm"], exp)]
        if not any(own_arm(r["group"], r["arm"], exp) for r in sel):
            continue
        prefix, title = exp["prefix"], exp["title"]
        full_rows = sheet_rows(sel, args.cell, None, exp)
        for c in full_rows[0][2][:6]:                 # the GT row: where the zoom is
            S.draw_box(c, box, side)
        f_full, f_zoom = "%s_%s.png" % (prefix, name), "%s_%s_zoom.png" % (prefix, name)
        S.write_png(sheet_path(out, exp, "material", f_full),
                    S.sheet("%s  -  %s  -  %s" % (title, name, args.name), notes, HEADS,
                            full_rows, args.cell))
        S.write_png(sheet_path(out, exp, "zoom", f_zoom),
                    S.sheet("%s  -  %s  -  zoom  -  %s" % (title, name, args.name), znotes,
                            HEADS, sheet_rows(sel, args.zoom_cell, box, exp), args.zoom_cell))
        keys = [r["key"] for r in sel]                # split_sheets.py cuts by these
        info["files"][rel(exp, "material", f_full)] = keys
        info["files"][rel(exp, "zoom", f_zoom)] = keys
    return {r["key"]: r for r in present}


def overview_cells(by_key, keys, gt_eval, mask_hw, size):
    """{ov key: [cell per row key]} for one material."""
    out = {k: [] for k in OV_KEYS}
    for key in keys:
        r = by_key.get(key)
        ok = r is not None and r["kind"] != "missing"
        for m in MAP4:
            out[m].append(S.map_cell(r["maps"][m], size) if ok else None)
        out["render"].append(S.render_cell(r["eval"], size) if ok else None)
        if not ok:
            out["error"].append(None)
        elif r["kind"] == "gt":
            out["error"].append(S.text_cell(size, ["GT"]))
        else:
            v = masked_psnr(r["eval"], gt_eval, mask_hw)
            out["error"].append(S.heat_cell(r["eval"], gt_eval, mask_hw, size,
                                            "%.2f dB" % v if v is not None else "inf"))
    return out


# ------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="evaluation folder under the eval root: <experiment>/<checkpoint>, e.g. deeptex_shared_vs_split/40k (scripts/eval.sh)")
    ap.add_argument("--eval-root", default=A.EVAL_ROOT)
    ap.add_argument("--root", default=A.MATSYNTH_ROOT, help="evaluation materials (<mat>/{SVBRDF,GT})")
    ap.add_argument("--matup-root", default=A.MATUP_RUNS)
    ap.add_argument("--seeds", default="0", help="MatUp seed indices k shown (default 0)")
    ap.add_argument("--material", default="", help="one material (default: all)")
    ap.add_argument("--cell", type=int, default=256,
                    help="cell size of material/<exp>_<material>.png")
    ap.add_argument("--zoom", type=int, default=96, help="zoom window, texels square")
    ap.add_argument("--zoom-cell", type=int, default=288,
                    help="cell size of zoom/<exp>_<material>_zoom.png")
    ap.add_argument("--zoom-at", default="", help="y,x top-left of the zoom window (crop coords)")
    ap.add_argument("--ov-cell", type=int, default=160, help="cell size of the overviews")
    ap.add_argument("--no-overview", action="store_true")
    ap.add_argument("--only", default="", choices=["", "deeptex", "matup"],
                    help="bilinear + this experiment's arms only (scripts/eval.sh sets it from "
                         "the folder: deeptex_shared_vs_split/<ckpt>, matup_shared_vs_split/<ckpt>)")
    ap.add_argument("--keep-missing", action="store_true",
                    help="draw an n/a row for an arm with no output for a material")
    args = ap.parse_args()

    if args.zoom < 8 or args.cell < 64 or args.zoom_cell < 64 or args.ov_cell < 32:
        die(2, "--zoom >= 8, --cell / --zoom-cell >= 64, --ov-cell >= 32")
    if args.zoom_at and len(args.zoom_at.split(",")) != 2:
        die(2, "--zoom-at must be y,x")
    args.seeds = tuple(int(s) for s in args.seeds.split(",") if s != "")
    base = os.path.join(E.check_out(args.eval_root), args.name)
    args.maps_root = A.maps_dir(args.name, args.eval_root)
    out = os.path.join(base, "montage")
    os.makedirs(out, exist_ok=True)

    mats = [os.path.join(args.root, args.material)] if args.material else \
        sorted(os.path.join(args.root, d) for d in os.listdir(args.root)
               if os.path.isdir(os.path.join(args.root, d, "SVBRDF")))
    if not mats or not all(os.path.isdir(os.path.join(m, "GT")) for m in mats):
        die(2, "no material with SVBRDF/ and GT/ under %s" % args.root)

    cols, skipped = E.available_columns(args)
    # earlier layouts: montage/<exp>/<kind>/ (third version), montage/<exp>_*.png (second)
    # and the single MatUp set montage/<kind>/matup_shared_vs_split_* (before the settings;
    # its montage/by_seed/ goes in split_sheets.split_run)
    for prefix in LEGACY_PREFIXES:
        if os.path.isdir(os.path.join(out, prefix)):
            shutil.rmtree(os.path.join(out, prefix))
        for old in glob.glob(os.path.join(out, prefix + "_*.png")):
            os.remove(old)
    for old in glob.glob(os.path.join(out, "*", "matup_shared_vs_split_*.png")):
        os.remove(old)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Renderer.init_renderer(argparse.Namespace(light_falloff=False), device)
    lights = rc.build_lights(device)
    if "grazing" not in lights["sheet"]:
        die(2, "no grazing held-out light (render_core.build_lights)")
    print("[montage] device=%s materials=%d rows=gt,%s out=%s" % (
        device, len(mats), ",".join(c["key"] for c in cols), out))

    keys = ["gt"] + [c["key"] for c in cols]
    heads = ["GT"] + [c["display"] for c in cols]
    groups = ["gt"] + [GROUP[c["kind"]] for c in cols]
    arms = ["gt"] + [arm_of(c) for c in cols]
    exp_rows = {}                                 # experiment with an own arm -> row indices
    for exp in EXPERIMENTS:
        idx = [i for i in range(len(keys)) if in_experiment(groups[i], arms[i], exp)]
        if any(own_arm(groups[i], arms[i], exp) for i in idx):
            exp_rows[exp["prefix"]] = idx
    infos, ov = [], {k: [] for k in OV_KEYS}
    with torch.no_grad():
        for d in mats:
            info, rows, mask_hw = process(d, cols, args, device, lights)
            by_key = write_material(out, info, rows, mask_hw, args)
            infos.append(info)
            if not args.no_overview:
                cells = overview_cells(by_key, keys, rows[0]["eval"], mask_hw, args.ov_cell)
                for k in OV_KEYS:
                    ov[k].append((info["material"], cells[k]))

    if not args.no_overview and not args.material:
        for old in glob.glob(os.path.join(out, "overview_*.png")):   # first version's names
            os.remove(old)
        for exp in EXPERIMENTS:
            idx = exp_rows.get(exp["prefix"])
            if idx is None:
                continue
            for k in OV_KEYS:
                what = {"render": "render EVAL_LIGHT (GT height)",
                        "error": "|render - GT| x%g EVAL_LIGHT, EVAL PSNR" % S.HEAT_GAIN
                        }.get(k, k)
                rows_k = [(mat, [cells[i] for i in idx]) for mat, cells in ov[k]]
                S.write_png(sheet_path(out, exp, "overview",
                                       "%s_overview_%s.png" % (exp["prefix"], k)),
                            S.overview("%s  -  %s  -  %s  -  %d materials" % (
                                exp["title"], what, args.name, len(rows_k)),
                                rows_k, [heads[i] for i in idx],
                                [sep_group(groups[i], arms[i], exp) for i in idx],
                                args.ov_cell))

    sheets = {e["dir"]: {"prefix": e["prefix"], "rows": [keys[i] for i in exp_rows[e["prefix"]]]}
              for e in EXPERIMENTS if e["prefix"] in exp_rows}
    with open(os.path.join(out, "montage.json"), "w") as fh:
        json.dump(rc.clean({"name": args.name, "rows": keys, "displays": heads, "sheets": sheets,
                            "seeds": list(args.seeds), "skipped": skipped,
                            "heat_gain": S.HEAT_GAIN, "cell": args.cell, "zoom": args.zoom,
                            "zoom_cell": args.zoom_cell, "materials": infos}), fh, indent=1)
    print("[montage] %d material(s) -> %s" % (len(infos), out))
    if not args.material:                  # a one-material run would cut sheets it did not write
        split_sheets.split_run(base)
    if skipped:
        print("[montage] arms with no output (not drawn): " +
              "; ".join("%s (%s)" % kv for kv in skipped))


if __name__ == "__main__":
    main()
