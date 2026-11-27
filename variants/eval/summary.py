"""Aggregation and summary.md for evaluate.py (numpy only).

Means are over the COMMON materials: those on which every available column was
scored (comp1 common_materials), so every number in one table is over the same
set. Paired differences are per material over the materials both arms of the
pair have, (split - shared); "wins" counts the materials where the split arm is
strictly better.

A map PSNR is None when the 8-bit maps are identical (texsr_metrics: PSNR = inf;
frequent for metallic, an all-zero GT predicted as all zero). Such materials
drop out of that PSNR mean (the cell then shows its n) and out of the PSNR mean
difference, but they still count in the wins, which compare the 8-bit MSE
(0 = identical = best).
"""
import numpy as np

from variants.eval import arms as A

# (metric, header, format, +1 higher is better / -1 lower is better, win key)
MAP_COLS = [
    ("baseColor_psnr", "baseColor PSNR", "%.2f", +1, "baseColor_mse"),
    ("baseColor_ssim", "baseColor SSIM", "%.4f", +1, None),
    ("roughness_psnr", "roughness PSNR", "%.2f", +1, "roughness_mse"),
    ("roughness_ssim", "roughness SSIM", "%.4f", +1, None),
    ("metallic_psnr", "metallic PSNR", "%.2f", +1, "metallic_mse"),
    ("metallic_ssim", "metallic SSIM", "%.4f", +1, None),
    ("normal_mae_deg", "normal MAE (deg)", "%.2f", -1, None),
]
RENDER_COLS = [
    ("train100_psnr", "train100 PSNR", "%.2f", +1, None),
    ("train100_l21", "train100 l2,1", "%.4f", -1, None),
    ("heldout_psnr", "held-out PSNR", "%.2f", +1, None),
    ("heldout_l21", "held-out l2,1", "%.4f", -1, None),
    ("render_lpips", "LPIPS", "%.4f", -1, None),
    ("render_flip", "FLIP", "%.4f", -1, None),
    ("basecolor_lpips", "baseColor LPIPS", "%.4f", -1, None),
    ("render_grad_pct_gt", "EVAL |grad| %GT", "%.0f", 0, None),
]


def diag_cols(prefix):
    return [
        ("%s_l21" % prefix, "full SR", "%.4f", -1, None),
        ("%s_l21_only_a" % prefix, "only a", "%.4f", -1, None),
        ("%s_l21_only_n" % prefix, "only n", "%.4f", -1, None),
        ("%s_l21_only_rm" % prefix, "only rm", "%.4f", -1, None),
        ("%s_l21_groupsum" % prefix, "sum of only-g", "%.4f", -1, None),
        ("%s_l21_full_over_groupsum" % prefix, "full / sum", "%.3f", -1, None),
    ]


DIAG_COLS = {"train100": diag_cols("train100"), "heldout": diag_cols("heldout")}
ALL_COLS = MAP_COLS + RENDER_COLS + DIAG_COLS["train100"] + DIAG_COLS["heldout"]


def fmt(v, f):
    return "-" if v is None else f % v


def value(rep, key, metric):
    e = rep["arms"].get(key)
    return None if e is None else e["metrics"].get(metric)


# ------------------------------------------------------------------- aggregation

def common_materials(reports, keys):
    return [r for r in reports if all(k in r["arms"] for k in keys)]


def mean_of(reports, key, metric):
    xs = [value(r, key, metric) for r in reports]
    xs = [x for x in xs if x is not None]
    return (float(np.mean(xs)) if xs else None), len(xs)


def means(reports, keys):
    return {k: {m[0]: mean_of(reports, k, m[0]) for m in ALL_COLS} for k in keys}


def seed_stats(table, keys_by_arm):
    """MatUp: mean and sample std (ddof=1) over seeds of the per-seed means."""
    out = {}
    for arm, keys in keys_by_arm.items():
        if not keys:
            continue
        out[arm] = {"seeds": keys}
        for m in ALL_COLS:
            xs = [table[k][m[0]][0] for k in keys if table[k][m[0]][0] is not None]
            out[arm][m[0]] = {
                "mean": float(np.mean(xs)) if xs else None,
                "std": float(np.std(xs, ddof=1)) if len(xs) > 1 else None,
                "n_seeds": len(xs)}
    return out


def better(x, y, direction):
    return x > y if direction > 0 else x < y


def paired(reports, a, b):
    """Per metric: mean of (a - b) over the materials both have, n, and wins of a."""
    both = [r for r in reports if a in r["arms"] and b in r["arms"]]
    out = {"n_materials": len(both), "materials": [r["material"] for r in both]}
    for metric, _, _, direction, win_key in ALL_COLS:
        d, wins, n_win = [], 0, 0
        for r in both:
            x, y = value(r, a, metric), value(r, b, metric)
            if x is not None and y is not None:
                d.append(x - y)
            if direction == 0:
                continue
            if win_key:
                wx, wy, wd = value(r, a, win_key), value(r, b, win_key), -1
            else:
                wx, wy, wd = x, y, direction
            if wx is not None and wy is not None:
                n_win += 1
                wins += better(wx, wy, wd)
        out[metric] = {"mean_diff": float(np.mean(d)) if d else None, "n": len(d),
                       "wins": int(wins), "n_win": n_win}
    return out


def paired_seed_stats(pair_results, seed_pairs):
    """matup_split - matup_shared over seeds: mean +- std of the per-seed mean
    differences, total wins."""
    got = [pair_results["%s_vs_%s" % p] for p in seed_pairs if "%s_vs_%s" % p in pair_results]
    if not got:
        return None
    out = {"seeds": len(got)}
    for metric, _, _, _, _ in ALL_COLS:
        xs = [g[metric]["mean_diff"] for g in got if g[metric]["mean_diff"] is not None]
        out[metric] = {"mean": float(np.mean(xs)) if xs else None,
                       "std": float(np.std(xs, ddof=1)) if len(xs) > 1 else None,
                       "wins": int(sum(g[metric]["wins"] for g in got)),
                       "n_win": int(sum(g[metric]["n_win"] for g in got))}
    return out


# ----------------------------------------------------------------------- tables

def mean_cell(cell, f, n_common):
    v, n = cell
    s = fmt(v, f)
    return s if v is None or n == n_common else "%s (n=%d)" % (s, n)


def pm_cell(st, f):
    if st["mean"] is None:
        return "-"
    if st["std"] is None:
        return f % st["mean"]
    return "%s ± %s" % (f % st["mean"], f % st["std"])


def table(L, cols, rows_keys, names, table_means, sstats, n_common):
    L.append("| arm | " + " | ".join(c[1] for c in cols) + " |")
    L.append("|---|" + "---|" * len(cols))
    for item in rows_keys:
        if item[0] == "key":
            k = item[1]
            L.append("| %s | %s |" % (names[k], " | ".join(
                mean_cell(table_means[k][c[0]], c[2], n_common) for c in cols)))
        else:
            arm = item[1]
            st = sstats[arm]
            L.append("| **%s, mean ± std over %d seeds** | %s |" % (
                A.DISPLAY[arm], len(st["seeds"]), " | ".join(pm_cell(st[c[0]], c[2]) for c in cols)))
    L.append("")


def diff_cell(d, f):
    if d["mean_diff"] is None and not d["n_win"]:
        return "-"
    s = "-" if d["mean_diff"] is None else ("%+" + f[1:]) % d["mean_diff"]
    if d["n_win"]:
        s += " (%d/%d)" % (d["wins"], d["n_win"])
    return s


def diff_pm_cell(d, f):
    if d["mean"] is None:
        return "-"
    s = ("%+" + f[1:]) % d["mean"]
    if d["std"] is not None:
        s += " ± " + f % d["std"]
    if d["n_win"]:
        s += " (%d/%d)" % (d["wins"], d["n_win"])
    return s


def diff_table(L, cols, pair_list, pair_results, seed_agg, names):
    L.append("| split - shared | n | " + " | ".join(c[1] for c in cols) + " |")
    L.append("|---|---|" + "---|" * len(cols))
    for a, b in pair_list:
        r = pair_results.get("%s_vs_%s" % (a, b))
        if r is None:
            continue
        L.append("| %s - %s | %d | %s |" % (names[a], names[b], r["n_materials"], " | ".join(
            diff_cell(r[c[0]], c[2]) for c in cols)))
    for arm, agg in (seed_agg or {}).items():
        L.append("| **%s - %s, mean ± std over %d seeds** | | %s |" % (
            A.DISPLAY.get(arm, arm),
            "shared" if A.matup_counterpart(arm) == "matup_shared" else "shared, no metallic reg",
            agg["seeds"], " | ".join(diff_pm_cell(agg[c[0]], c[2]) for c in cols)))
    L.append("")


def write_summary(path, name, reports, common, cols, names, table_means, sstats,
                  pair_list, pair_results, seed_agg, skipped, lights, perceptual_on, args):
    keys = [c["key"] for c in cols]
    rows = []
    for c in cols:
        if c["kind"] != "matup":
            rows.append(("key", c["key"]))
    for arm in A.MATUP_ARMS:
        ks = [c["key"] for c in cols if c.get("arm") == arm]
        rows += [("key", k) for k in ks]
        if ks:
            rows.append(("seeds", arm))
    in_common = {r["material"] for r in common}
    excluded = [(r["material"], [k for k in keys if k not in r["arms"]])
                for r in reports if r["material"] not in in_common]
    borders = sorted({r["border"] for r in reports})

    L = ["# Shared vs split networks: `%s`, %d MatSynth evaluation materials" % (name, len(reports)),
         "",
         "docs/SPLIT_THEORY.md D. x4 (128 -> 512). Generated by variants/eval/evaluate.py.",
         "",
         "- **Height**: every arm is rendered with the **GT height** (no split arm predicts",
         "  height; the shared DeepTex arm's SR height and MatUp's LR-height displacement are",
         "  not used). Only the MatUp self-check renders with MatUp's own LR height.",
         "- **Crop**: every arm and the GT are cropped by MatUp's border (%s px, measured on the"
         % "/".join(str(b) for b in borders),
         "  MatUp _opt maps as comp1 did); render metrics also skip GT-invalid normals",
         "  (|2v-1| < 0.1 or unit n_z < 1e-3).",
         "- **Map metrics**: PSNR / SSIM on the stored maps rounded to 8 bit on both sides",
         "  (texsr_metrics; MatUp writes 16 bit, the GT and DeepTex 8 bit). baseColor in sRGB,",
         "  roughness / metallic raw ([0,1]; MatUp's roughness_opt is sqrt(alpha)). Normal MAE:",
         "  mean angle between the decoded unit normals, GT-valid texels. bilinear's maps",
         "  here are cv2.INTER_LINEAR of the stored LR PNGs; its renders use MatUp's own",
         "  initialisation (decode at 128, then bilinear, normal rebuilt from XY).",
         "- **Render metrics**: MatUp's torch renderer, display values (Reinhard + gamma).",
         "  train100 = MatUp's 1.4 x fib(100) training lights; held-out = EVAL_LIGHT +",
         "  1.4 x fib(16) turned 43.25 deg (%d lights, >= %.2f deg from every training light)."
         % (len(lights["held_np"]), float(np.min(lights["held_min_sep_deg"]))),
         "  PSNR: MSE over texels x lights x RGB. l2,1: MatUp's loss (mean RGB L2 norm per",
         "  texel-light), lower is better.",
         ("  LPIPS-alex over the %s lights, FLIP (LDR) over the %s lights, on the same renders;"
          % (args.lpips_lights, args.flip_lights)) if perceptual_on else
         "  LPIPS / FLIP: not computed (--no-perceptual).",
         "  baseColor LPIPS on the stored sRGB map. Lower is better for all three.",
         "- **Compensation** (SPLIT_THEORY C-1): l2,1 of the GT material with only group g's",
         "  SR maps swapped in (a = baseColor, n = normal, rm = roughness + metallic), next to",
         "  the full-SR l2,1. full / sum < 1 means the groups' render errors partly cancel.",
         "- **MatUp seeds** are shown individually and as mean ± std (ddof 1) of the per-seed means.",
         "- Means are over the **%d common materials** (every listed arm scored); a cell's"
         % len(common),
         "  (n=k) means k of them had a finite value (an identical 8-bit map has PSNR = inf).",
         ""]
    L.append("Arms: " + "; ".join("`%s` = %s" % (k, names[k]) for k in keys) + ".")
    if skipped:
        L.append("")
        L.append("**Skipped arms**: " + "; ".join("`%s` (%s)" % kv for kv in skipped) + ".")
    if excluded:
        L.append("")
        L.append("Not in the means (an arm missing): " + "; ".join(
            "%s (%s)" % (m, ", ".join(ks)) for m, ks in excluded) + ".")
    L.append("")

    L += ["## Map metrics (mean over %d materials)" % len(common), ""]
    table(L, MAP_COLS, rows, names, table_means, sstats, len(common))
    L += ["## Render metrics, GT height (mean over %d materials)" % len(common), ""]
    table(L, RENDER_COLS, rows, names, table_means, sstats, len(common))
    L += ["## Cross-map error compensation (SPLIT_THEORY C-1)", ""]
    for prefix, title in (("train100", "train100 lights (the render loss's lights)"),
                          ("heldout", "held-out lights")):
        L += ["### l2,1 under %s" % title, ""]
        table(L, DIAG_COLS[prefix], rows, names, table_means, sstats, len(common))

    L += ["## Paired per-material differences: split - shared counterpart", "",
          "Cell = mean over materials of (split - shared); (w/n) = materials where the split",
          "arm is strictly better. PSNR / SSIM: positive is better for split; l2,1, LPIPS,",
          "FLIP, normal MAE, full/sum: negative is better for split.", ""]
    L += ["### Maps", ""]
    diff_table(L, MAP_COLS, pair_list, pair_results, seed_agg, names)
    L += ["### Renders (GT height)", ""]
    diff_table(L, [c for c in RENDER_COLS if c[3] != 0], pair_list, pair_results, seed_agg, names)
    L += ["### Compensation, train100 lights", ""]
    diff_table(L, DIAG_COLS["train100"], pair_list, pair_results, seed_agg, names)
    L += ["### Compensation, held-out lights", ""]
    diff_table(L, DIAG_COLS["heldout"], pair_list, pair_results, seed_agg, names)

    mk = [c["key"] for c in cols if c["kind"] == "matup"]
    if mk:
        L += ["## MatUp self-check (max |ours - render_opt.png| in 16-bit LSB, tol %.1f)"
              % (5e-4 * 65535.0), "",
              "Our EVAL_LIGHT render of each MatUp run's _opt maps (with MatUp's own LR-height",
              "displacement) against the render_opt.png MatUp wrote; checks the normal",
              "convention, grid and crop used for every MatUp number above.", "",
              "| material | " + " | ".join(mk) + " |", "|---|" + "---|" * len(mk)]
        for r in reports:
            sc = r["self_check"]
            L.append("| %s | %s |" % (r["material"], " | ".join(
                fmt(sc[k]["max_abs"] * 65535.0 if sc.get(k, {}).get("max_abs") is not None
                    else None, "%.2f") for k in mk)))
        L.append("")
    steps = {}
    for r in reports:
        for k, e in r["arms"].items():
            s = e.get("source", {}).get("step")
            if s is not None:
                steps.setdefault(k, set()).add(s)
    if steps:
        L.append("Checkpoint steps (sr_manifest.json): " + "; ".join(
            "`%s` %s" % (k, ",".join(str(s) for s in sorted(v))) for k, v in steps.items()) + ".")
        L.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
