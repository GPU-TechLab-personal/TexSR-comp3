#!/usr/bin/env python3
"""Stage (b) of the shared-vs-split evaluation: every arm's maps, renders and metrics.

Runs in the matup container (torch 2.7.1, Python 3.10, lpips, flip-evaluator);
scripts/eval.sh score runs

    python /workspace/variants/eval/evaluate.py --name main

after stage (a) (variants/eval/infer_deeptex.py, deeptex container) has written
the DeepTextures arm folders. docs/SPLIT_THEORY.md is the question; this is the
port of comp1's svbrdf_render_eval.py + svbrdf_perceptual_eval.py (via
render_core.py / perceptual.py) with the changes below.

HEIGHT. EVERY ARM IS RENDERED WITH THE GT HEIGHT. None of the split arms
predicts height (split/groups.py has no height network; MatUp writes none), so
for a like-for-like render comparison the position map (displacement) of every
arm - bilinear, the shared DeepTex arm (whose SR height is ignored), the split
DeepTex arms and every MatUp run - is the GT height's (SPLIT_THEORY A2-3, C-3).
comp1 rendered bilinear / MatUp with the LR height bilinear x4 and DeepTex with
its own SR height; that survives here only in the MatUp self-check.

COLUMNS (variants/eval/arms.py; an arm with no outputs is skipped with a warning)
  bilinear              <root>/<mat>/SVBRDF. Renders: MatUp's own initialisation
                        (render_core.load_bilinear). Map metrics / maps PNGs:
                        cv2.INTER_LINEAR x4 of the stored LR PNGs (comp1's
                        bilinear column, matup_outputs.build_columns).
  deeptex_renloss_21,   <eval>/<name>/maps/<arm>/<mat>/{baseColor,normal,roughness,
  deeptex_split_nb23,   metallic}.png (arm_output contract, OpenGL normal, 8 bit),
  deeptex_split_nb7     read like a GT folder (load_set(..., opengl=True, floor=True)).
  matup_shared_s<k>,    <matup-root>/<arm>/s<k>/<mat>/<stamp>/{baseColor,normal,
  matup_split_s<k>      roughness,metallic}_opt.png: MatUp's output_maps_from_list
                        (data_utils.py:107-143) - 512^2 with the 8 px zero border it
                        padded on, 16 bit, normal in MatUp's internal DirectX frame
                        (write_normal(..., opengl_normals=False)), roughness_opt =
                        sqrt(alpha). <stamp> is MatUp's "%b%d_%Hh%Mm%Ss", which does
                        not sort by time, so the newest stamp folder holding all
                        four _opt maps is chosen by baseColor_opt.png's mtime.

CROP. The border is detected on the MatUp runs' _opt maps (baseColor, roughness,
metallic; matup_outputs.detect_border, as comp1) and the SAME border crops every
arm and the GT. If runs disagree the largest border is used (warning); with no
MatUp run for a material, MatUp's tile_width // 2 = 8.

METRICS per (material, arm)
  maps     baseColor / roughness / metallic PSNR and SSIM on the stored maps, both
           sides rounded to 8 bit (texsr_metrics.full_reference); normal mean
           angular error (deg) between the decoded unit normals on GT-valid texels
           (comp1's normal_angle_deg).
  render   (GT height) train100 and held-out PSNR / l2,1 (+ elevation bands,
           per light), EVAL_LIGHT PSNR and |grad| %GT; LPIPS (held-out lights),
           FLIP (sheet lights), baseColor LPIPS (perceptual.py).
  groups   SPLIT_THEORY C-1: l2,1 of the GT material with only group g's SR maps
           (a = baseColor, n = normal, rm = roughness + metallic), train100 and
           held-out, next to the full-SR l2,1; full / sum < 1 = compensation.

SELF-CHECKS (exit 3): every MatUp run's EVAL_LIGHT render (with MatUp's LR-height
displacement) reproduces its render_opt.png within 5e-4; LPIPS / FLIP behave
(perceptual.metric_check, first material).

OUTPUTS  <eval>/<name>/report.json   everything, per material
         <eval>/<name>/summary.md    tables: means per arm (MatUp seeds + mean ± std),
                                      paired split - shared differences
         <eval>/<name>/maps/{bilinear,matup_*}/<mat>/<map>.png   the maps scored
                                      (MatUp: normal G flipped to OpenGL, border kept)
"""
import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                *[os.pardir] * 2)))
from variants.eval import render_core as rc  # noqa: E402  (puts MatUp on sys.path first)

import torch  # noqa: E402
from matup.renderer import Renderer  # noqa: E402

from variants.eval import arms as A  # noqa: E402
from variants.eval import perceptual as pc  # noqa: E402
from variants.eval import summary as sm  # noqa: E402
from variants.eval.matup_maps import crop  # noqa: E402
from variants.matup.baseline import matup_outputs as mo  # noqa: E402
from variants.util import texsr_metrics as tm  # noqa: E402
from variants.util.report import images as rimg  # noqa: E402

die = rc.die
MAP4 = rc.MAP4
SCORED_MAPS = ("baseColor", "roughness", "metallic")
GREY = {"baseColor": False, "normal": False, "roughness": True, "metallic": True}
DEFAULT_BORDER = 17 // 2          # MatUp main.py --tile_width 17 -> tile_width // 2
OUTPUTS = ("report.json", "summary.md")


# ------------------------------------------------------------------- sources

def newest_matup_run(root, arm, seed, name):
    base = os.path.join(root, arm, "s%d" % seed, name)
    cands = [d for d in glob.glob(os.path.join(base, "*")) if os.path.isdir(d)
             and all(os.path.isfile(os.path.join(d, m + "_opt.png")) for m in MAP4)]
    if not cands:
        return None
    return max(cands, key=lambda d: (os.path.getmtime(os.path.join(d, "baseColor_opt.png")), d))


def measure_border(run_dir):
    """comp1 svbrdf_perceptual_eval.measure_border: MatUp's zero rim, never
    measured on normal (mid-grey rim) or the pre-cropped height."""
    opts = {}
    for m, (_, opt_name) in mo.MAPS.items():
        p = os.path.join(run_dir, opt_name)
        if os.path.exists(p):
            opts[m] = rimg.read(p)
    return mo.detect_border(
        [a for m, a in opts.items() if m != "normal" and m not in mo.PRECROPPED])


def read_stored(path, m):
    """A map PNG as cv2 stores it (BGR, uint8/uint16); grey maps 2-D."""
    a = rimg.read(path)
    if GREY[m] and a.ndim == 3:
        a = a[:, :, 0]
    elif not GREY[m] and a.ndim == 3 and a.shape[2] == 4:
        a = a[:, :, :3]
    return a


def available_columns(args):
    """Every arms.py column whose outputs exist at all, plus (key, reason) skips."""
    cols, skipped = [], []
    for c in A.columns(args.seeds):
        if c["kind"] == "deeptex":
            d = os.path.join(args.maps_root, c["key"])
            if not os.path.isdir(d) or not glob.glob(os.path.join(d, "*", "baseColor.png")):
                print("[warn] %s: no arm folder %s - skipped (bash scripts/eval.sh infer)"
                      % (c["key"], d))
                skipped.append((c["key"], "no maps in %s" % d))
                continue
            c["dir"] = d
        elif c["kind"] == "matup":
            d = os.path.join(args.matup_root, c["arm"], "s%d" % c["seed"])
            if not glob.glob(os.path.join(d, "*", "*", "baseColor_opt.png")):
                print("[warn] %s: no MatUp output under %s - skipped" % (c["key"], d))
                skipped.append((c["key"], "no MatUp output in %s" % d))
                continue
            c["dir"] = d
        cols.append(c)
    return cols, skipped


def load_column(col, mat_dir, name, full, run_dir):
    """(four renderer maps, stored maps, source info) or None when the arm has no
    readable output for this material."""
    if col["kind"] == "bilinear":
        maps = rc.load_bilinear(mat_dir, full)
        stored = {m: cv2.resize(read_stored(os.path.join(mat_dir, "SVBRDF", m + ".png"), m),
                                (full, full), interpolation=cv2.INTER_LINEAR) for m in MAP4}
        return maps, stored, {"dir": os.path.join(mat_dir, "SVBRDF")}
    if col["kind"] == "deeptex":
        d = os.path.join(col["dir"], name)
        need = [os.path.join(d, m + ".png") for m in MAP4]
        bad = rc.unreadable(need)
        if bad:
            print("  ! %s: %s has no readable %s - skipped" % (
                name, col["key"], ", ".join(os.path.basename(p) for p in bad)))
            return None
        src = {"dir": d}
        man = os.path.join(d, "sr_manifest.json")
        if os.path.exists(man):
            with open(man) as fh:
                mj = json.load(fh)
            src.update({k: mj.get(k) for k in ("checkpoint", "step", "scalar_collapse")})
        maps = rc.load_arm(col["dir"], name)
        stored = {m: read_stored(p, m) for m, p in zip(MAP4, need)}
        return maps, stored, src
    if run_dir is None:
        print("  ! %s: %s has no MatUp run - skipped" % (name, col["key"]))
        return None
    need = [os.path.join(run_dir, m + "_opt.png") for m in MAP4]
    bad = rc.unreadable(need)
    if bad:
        print("  ! %s: %s run %s unreadable: %s - skipped" % (name, col["key"], run_dir, bad))
        return None
    maps = rc.load_matup(run_dir)
    stored = {m: read_stored(p, m) for m, p in zip(MAP4, need)}
    return maps, stored, {"dir": run_dir}


def write_maps(out_dir, stored, kind, source):
    """The bilinear / MatUp maps as scored, in arm_output's layout: GT size, OpenGL
    normal (MatUp's G flipped, as matup_outputs.build_columns does)."""
    os.makedirs(out_dir, exist_ok=True)
    for m in MAP4:
        a = stored[m]
        if m == "normal" and kind == "matup":
            a = rimg.to_u16(a).copy()
            a[..., 1] = rimg.U16_MAX - a[..., 1]      # DirectX -> OpenGL
        rimg.write(os.path.join(out_dir, m + ".png"), a)
    with open(os.path.join(out_dir, "source.json"), "w") as fh:
        json.dump(source, fh, indent=2)


# ------------------------------------------------------------------- metrics

def map_metrics(stored, gt_stored, border, where):
    out = {}
    for m in SCORED_MAPS:
        p, g = crop(stored[m], border), crop(gt_stored[m], border)
        if p.shape != g.shape:
            die(2, "%s: %s is %s, GT %s" % (where, m, p.shape, g.shape))
        fr = tm.full_reference(p, g, 8)
        out[m + "_psnr"] = fr["psnr"]
        out[m + "_ssim"] = fr["ssim"]
        # 8-bit MSE (levels^2), from the same quantised PSNR; 0 = identical maps.
        out[m + "_mse"] = 0.0 if fr["psnr"] is None else 255.0 ** 2 / 10.0 ** (fr["psnr"] / 10.0)
    return out


def srgb01(stored_bc, border):
    """A stored baseColor -> HxWx3 RGB float in [0,1], cropped."""
    return crop(tm.to_float(stored_bc)[..., ::-1], border)


# --------------------------------------------------------------- per material

def process(mat_dir, cols, args, device, lights, metric, sets, check):
    t0 = time.time()
    name = os.path.basename(mat_dir.rstrip("/"))
    rc.require_readable([os.path.join(mat_dir, sub, m + ".png")
                         for sub in ("GT", "SVBRDF") for m in rc.SLOT], name)

    runs = {c["key"]: newest_matup_run(args.matup_root, c["arm"], c["seed"], name)
            for c in cols if c["kind"] == "matup"}
    detected = {k: measure_border(r) for k, r in runs.items() if r}
    if detected:
        vals = sorted({b for b, _ in detected.values()})
        border = vals[-1]
        if len(vals) > 1:
            print("  ! %s: MatUp borders differ %s - cropping all arms by %d" % (
                name, {k: b for k, (b, _) in detected.items()}, border))
    else:
        border = DEFAULT_BORDER
        print("  ! %s: no MatUp run to measure the border on - using %d" % (name, border))

    gt4 = rc.load_gt(mat_dir)
    full = gt4[0].shape[0]
    gt_pos = rc.gt_position(mat_dir)
    if gt_pos.shape[0] != full:
        die(2, "%s: GT height is %d^2, GT maps %d^2" % (name, gt_pos.shape[0], full))
    side = full - 2 * border
    mask_hw = rc.gt_valid_mask(os.path.join(mat_dir, "GT"), border)
    n_valid = int(mask_hw.sum())
    if n_valid == 0:
        die(2, "%s: no GT texel passes the validity rule" % name)
    metal = bool(rc.read_raw(os.path.join(mat_dir, "GT", "metallic.png"), False).max() > 0)

    rep = {"material": name, "border": border,
           "border_detected": {k: [b, list(s)] for k, (b, s) in detected.items()},
           "size": side, "full": full, "metal": metal, "n_valid": n_valid,
           "valid_frac": n_valid / float(side * side),
           "matup_runs": runs, "self_check": {}, "arms": {}, "missing": []}

    gt_rmaps, gt_fix = rc.prepare(gt4 + [gt_pos], border, device)
    gt_eval = rc.eval_image(gt_rmaps, lights["eval"], "%s gt eval" % name)
    gt_held = rc.render_chunks(gt_rmaps, lights["held"], args.chunk, "%s gt heldout" % name)
    gt_train = rc.render_chunks(gt_rmaps, lights["train"], args.chunk, "%s gt train100" % name)
    gt_grad = tm.sharpness(rc.to_u16(gt_eval))[0]
    rep["gt"] = {"render_grad": gt_grad, "nz_zero_fixed": gt_fix,
                 "nz_below_min": int((gt_rmaps[2][..., 2] < rc.NZ_MIN).sum()),
                 "metallic_mean": float(gt_rmaps[1].double().mean())}
    ctx = {"n_valid": n_valid, "mask_hw": mask_hw, "mask_flat": mask_hw.reshape(-1),
           "mask": torch.from_numpy(mask_hw.reshape(-1).astype(np.float32)).to(device),
           "held": lights["held"], "train": lights["train"], "eval": lights["eval"],
           "bands_held": lights["bands_held"], "bands_train": lights["bands_train"],
           "gt_rmaps": gt_rmaps, "gt_held": gt_held, "gt_train": gt_train,
           "gt_eval": gt_eval, "gt_grad": gt_grad}

    gt_stored = {m: read_stored(os.path.join(mat_dir, "GT", m + ".png"), m) for m in MAP4}
    gt_bc = srgb01(gt_stored["baseColor"], border)
    pctx = None
    if metric is not None:
        pctx = pc.make_ctx(gt_rmaps, sets, args.chunk, side, mask_hw, device, "%s gt" % name)
        if check:
            rep["metric_check"] = pc.metric_check(pctx, metric, gt_bc, device, args.chunk, name)

    line = []
    for col in cols:
        key = col["key"]
        got = load_column(col, mat_dir, name, full, runs.get(key))
        if got is None:
            rep["missing"].append(key)
            continue
        maps4, stored, src = got
        if any(t.shape[0] != full or t.shape[:2] != maps4[0].shape[:2] for t in maps4) or \
                any(a.shape[:2] != (full, full) for a in stored.values()):
            die(2, "%s / %s: map sizes %s, GT is %d" % (
                name, key, [tuple(t.shape) for t in maps4], full))
        where = "%s %s" % (name, key)

        if col["kind"] == "matup":
            sc_maps, _ = rc.prepare(maps4 + [rc.matup_lr_position(mat_dir, full)], border, device)
            rep["self_check"][key] = rc.self_check(
                rc.eval_image(sc_maps, lights["eval"], where + " self-check"),
                runs[key], border, full, where)
            del sc_maps

        rmaps, fix = rc.prepare(maps4 + [gt_pos], border, device)
        m, _ = rc.column_metrics(rmaps, ctx, args.chunk, where)
        m.update(map_metrics(stored, gt_stored, border, where))
        if metric is not None:
            m.update(pc.score_column(rmaps, pctx, metric, args.chunk, where))
            m["basecolor_lpips"] = pc.map_lpips(metric, srgb01(stored["baseColor"], border),
                                                gt_bc, device, where)
        m["nz_zero_fixed"] = fix
        del rmaps

        per_light = {k: m.pop(k) for k in list(m) if k.endswith("_per_light")}
        rep["arms"][key] = {"metrics": m, "per_light": per_light, "source": src}
        if col["kind"] in ("bilinear", "matup"):
            write_maps(os.path.join(args.maps_root, key, name), stored, col["kind"], src)
        line.append("%s %.4f/%.4f" % (key, m["train100_l21"], m["train100_l21_full_over_groupsum"]
                                      if m["train100_l21_full_over_groupsum"] is not None else 0))

    sc = rep["self_check"]
    print("  %-28s border=%d valid=%.4f%s  MatUp self-check max LSB %s  %.0fs" % (
        name, border, rep["valid_frac"], " metal" if metal else "",
        " ".join("%.1f" % (v["max_abs"] * rc.U16) for v in sc.values()
                 if v.get("max_abs") is not None) or "-", time.time() - t0))
    print("      train100 l2,1 / full-over-groupsum: " + "  ".join(line))
    return rep


# ------------------------------------------------------------------------ main

def check_out(path):
    out = os.path.normpath(path)
    if not os.path.isabs(out):
        die(2, "--eval-root must be absolute (the matup container starts in %s)" % rc.MATUP_ROOT)
    ext = os.path.normpath(os.path.join(rc.paths.REPO, "external"))
    if out == ext or out.startswith(ext + os.sep):
        die(2, "output %s is under %s; the submodules must stay clean" % (out, ext))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="main", help="evaluation name: <eval-root>/<name>")
    ap.add_argument("--eval-root", default=A.EVAL_ROOT)
    ap.add_argument("--root", default=A.MATSYNTH_ROOT, help="evaluation materials (<mat>/{SVBRDF,GT})")
    ap.add_argument("--matup-root", default=A.MATUP_RUNS,
                    help="<matup-root>/<arm>/s<k>/<material>/<stamp>/*_opt.png")
    ap.add_argument("--seeds", default="0,1,2", help="MatUp seed indices k (s<k>)")
    ap.add_argument("--material", default="")
    ap.add_argument("--chunk", type=int, default=10, help="lights rendered per call")
    ap.add_argument("--lpips-lights", default="heldout", choices=["heldout", "sheet"])
    ap.add_argument("--flip-lights", default="sheet", choices=["heldout", "sheet"])
    ap.add_argument("--no-perceptual", action="store_true", help="skip LPIPS / FLIP")
    ap.add_argument("--threads", type=int, default=0, help="cap torch / OpenCV CPU threads")
    args = ap.parse_args()

    if args.chunk < 1:
        die(2, "--chunk must be >= 1")
    if args.threads > 0:
        torch.set_num_threads(args.threads)
        cv2.setNumThreads(args.threads)
    args.seeds = tuple(int(s) for s in args.seeds.split(",") if s != "")
    out = os.path.join(check_out(args.eval_root), args.name)
    args.maps_root = A.maps_dir(args.name, args.eval_root)

    mats = [os.path.join(args.root, args.material)] if args.material else \
        sorted(d for d in glob.glob(os.path.join(args.root, "*"))
               if os.path.isdir(os.path.join(d, "SVBRDF")))
    if not mats or not all(os.path.isdir(os.path.join(m, "GT")) for m in mats):
        die(2, "no material with SVBRDF/ and GT/ under %s" % args.root)
    if not args.material and len(mats) != 12:
        print("[warn] %d materials under %s (the evaluation set has 12)" % (len(mats), args.root))

    cols, skipped = available_columns(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # matup_render.py:215-217: mandatory, and light falloff off as everywhere.
    Renderer.init_renderer(argparse.Namespace(light_falloff=False), device)
    lights = rc.build_lights(device)
    metric = sets = None
    if not args.no_perceptual:
        metric = pc.Perceptual(device)
        sets = pc.light_sets(lights, args.lpips_lights, args.flip_lights)
    os.makedirs(out, exist_ok=True)
    for f in OUTPUTS:
        p = os.path.join(out, f)
        if os.path.isfile(p):
            os.remove(p)
    print("[eval] device=%s materials=%d columns=%s out=%s" % (
        device, len(mats), ",".join(c["key"] for c in cols), out))
    if not lights["train_equals_matup_lights"]:
        print("  ! MatUp's light table differs from matup_lights.training_lights")

    reports = []
    with torch.no_grad():
        for i, d in enumerate(mats):
            reports.append(process(d, cols, args, device, lights, metric, sets, check=(i == 0)))

    # Columns that scored nothing are dropped (and listed as skipped).
    scored = {k for r in reports for k in r["arms"]}
    for c in cols:
        if c["key"] not in scored:
            skipped.append((c["key"], "no material scored"))
    cols = [c for c in cols if c["key"] in scored]
    if not cols:
        die(2, "nothing scored")
    keys = [c["key"] for c in cols]
    names = {c["key"]: c["display"] for c in cols}

    common = sm.common_materials(reports, keys)
    table_means = sm.means(common, keys)
    keys_by_arm = {arm: [c["key"] for c in cols if c.get("arm") == arm] for arm in A.MATUP_ARMS}
    sstats = sm.seed_stats(table_means, keys_by_arm)
    pair_list = [p for p in A.pairs(args.seeds) if p[0] in scored and p[1] in scored]
    pair_results = {"%s_vs_%s" % p: sm.paired(reports, *p) for p in pair_list}
    # per MatUp split arm: split - shared over seeds (one row each in the paired tables)
    seed_agg = {}
    for arm in A.MATUP_SPLIT_ARMS:
        seed_pairs = [p for p in pair_list if p[0] in {A.matup_key(arm, s) for s in args.seeds}]
        if len(seed_pairs) > 1:
            seed_agg[arm] = sm.paired_seed_stats(pair_results, seed_pairs)
    seed_agg = {k: v for k, v in seed_agg.items() if v} or None

    sm.write_summary(os.path.join(out, "summary.md"), args.name, reports, common, cols, names,
                     table_means, sstats, pair_list, pair_results, seed_agg, skipped, lights,
                     metric is not None, args)
    report = {
        "name": args.name, "root": args.root, "maps_root": args.maps_root,
        "matup_root": args.matup_root, "seeds": list(args.seeds),
        "height": "GT height for every arm",
        "columns": [{k: c.get(k) for k in ("key", "kind", "display", "arm", "seed", "split", "dir")}
                    for c in cols],
        "skipped": [{"key": k, "why": w} for k, w in skipped],
        "pairs": [list(p) for p in pair_list],
        "lights": {"train100": lights["train_np"], "heldout": lights["held_np"],
                   "train100_equals_matup_lights": lights["train_equals_matup_lights"],
                   "bands_train100": lights["bands_train"], "bands_heldout": lights["bands_held"],
                   "sheet_lights": lights["sheet"],
                   "heldout_min_sep_deg": lights["held_min_sep_deg"],
                   "lpips": sets["lpips"] if sets else None,
                   "flip": sets["flip"] if sets else None},
        "constants": {"nz_min": rc.NZ_MIN, "mask_min_norm": rc.MASK_MIN_NORM,
                      "normal_norm_floor": rc.NORMAL_NORM_FLOOR, "l21_eps": rc.L21_EPS,
                      "self_check_tol_matup": rc.SELF_TOL_MATUP, "map_bits": 8,
                      "chunk": args.chunk, "device": str(device)},
        "perceptual": None if metric is None else {
            "versions": metric.versions, "flip_parameters": metric.flip_params},
        "common_materials": [r["material"] for r in common],
        "means": {k: {m: {"mean": v[0], "n": v[1]} for m, v in t.items()}
                  for k, t in table_means.items()},
        "matup_seed_stats": sstats,
        "paired": pair_results,
        "paired_matup_over_seeds": seed_agg,
        "materials": reports,
    }
    with open(os.path.join(out, "report.json"), "w") as fh:
        json.dump(rc.clean(report), fh, indent=1)
    print("[eval] %d material(s), %d common, %d arm(s) -> %s" % (
        len(reports), len(common), len(cols), os.path.join(out, "summary.md")))
    if skipped:
        print("[eval] skipped: " + "; ".join("%s (%s)" % kv for kv in skipped))


if __name__ == "__main__":
    main()
