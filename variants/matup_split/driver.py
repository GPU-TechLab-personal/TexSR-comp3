#!/usr/bin/env python3
"""Every MatUp run of the split experiment, one subprocess per run (scripts/matup_split.sh).

    python /workspace/variants/matup_split/driver.py prep
    python /workspace/variants/matup_split/driver.py optim [--arms matup_shared,matup_split,...] [--seeds 0,1,2]

prep   per material: LR renders + SwinIR-L upscaled renders into
       runs/matup_split/cache/<material>/ (once; every arm and seed reads them)
optim  per arm x seed x material: MatUp's optimisation with the shared or split
       MLP into runs/matup_split/<arm>/s<k>/<material>/ (run.json marks it done)

Seeds: s0 = 1996 (MatUp's own seed, main_optim.py:36-37), s1 = 1997, s2 = 1998.
A run that already has run.json is skipped, so rerunning continues where it stopped.
Each run is its own process: MatUp captures CUDA graphs and keeps global renderer
state, so runs never share one.
"""

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_SET = "/data/TexSR-data/matsynth"
OUT = "/workspace/runs/matup_split"
# arm -> (--mlp, --split-width, --alpha-reg); width 0 = MatUp's 128, alpha_reg None = MatUp's
# 0.1. x1 / x3: parameter-matched to 1x / 3x the shared MLP (split_mlp.WIDTHS, SPLIT_THEORY
# B1-3). *_noreg: the metallic regulariser off (SPLIT_THEORY B1-4), otherwise as without.
ARMS = {"matup_shared": ("shared", 0, None), "matup_split": ("split", 0, None),
        "matup_split_x1": ("split", 104, None), "matup_split_x3": ("split", 227, None),
        "matup_shared_noreg": ("shared", 0, 0.0), "matup_split_noreg": ("split", 0, 0.0)}
SEEDS = {0: 1996, 1: 1997, 2: 1998}


def materials(root):
    if not os.path.isdir(root):
        sys.exit("[fatal] no evaluation set at %s - run: bash scripts/data.sh evalset" % root)
    names = sorted(n for n in os.listdir(root)
                   if os.path.isfile(os.path.join(root, n, "SVBRDF", "baseColor.png")))
    if not names:
        sys.exit("[fatal] %s holds no <material>/SVBRDF/baseColor.png" % root)
    return names


def n_png(d):
    return len([f for f in os.listdir(d) if f.endswith(".png")]) if os.path.isdir(d) else 0


def run(cmd):
    print("[driver] %s" % " ".join(cmd), flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd)
    print("[driver] exit %d after %.0f s" % (rc, time.time() - t0), flush=True)
    return rc


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["prep", "optim"])
    ap.add_argument("--eval-set", default=EVAL_SET)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--seeds", default=",".join(str(k) for k in SEEDS))
    ap.add_argument("--only", default="", help="comma-separated material names (default: all)")
    args = ap.parse_args()

    names = materials(args.eval_set)
    if args.only:
        keep = set(args.only.split(","))
        names = [n for n in names if n in keep]
    run_py = os.path.join(HERE, "run.py")
    failed = []

    for i, name in enumerate(names):
        mat = os.path.join(args.eval_set, name)
        cache = os.path.join(args.out, "cache", name)
        if args.stage == "prep":
            if all(n_png(os.path.join(cache, d)) == 100 for d in ("radiance_LR", "radiance_Upscaled")):
                print("[driver] %d/%d %s: cache present" % (i + 1, len(names), name))
                continue
            if run([sys.executable, run_py, "--material", mat, "--cache", cache,
                    "--stage", "prep"]):
                failed.append(("prep", name))
            continue
        for arm in args.arms.split(","):
            if arm not in ARMS:
                sys.exit("[fatal] unknown arm %r (%s)" % (arm, ", ".join(ARMS)))
            for k in (int(s) for s in args.seeds.split(",")):
                out = os.path.join(args.out, arm, "s%d" % k, name)
                if os.path.isfile(os.path.join(out, "run.json")):
                    print("[driver] %s s%d %s: done" % (arm, k, name))
                    continue
                mlp, width, alpha_reg = ARMS[arm]
                cmd = [sys.executable, run_py, "--material", mat, "--cache", cache,
                       "--stage", "optim", "--mlp", mlp, "--seed", str(SEEDS[k]), "--out", out]
                if width:
                    cmd += ["--split-width", str(width)]
                if alpha_reg is not None:
                    cmd += ["--alpha-reg", repr(alpha_reg)]
                if run(cmd):
                    failed.append((arm, k, name))

    if failed:
        print("[driver] FAILED %d run(s): %s" % (len(failed), failed))
        sys.exit(1)
    print("[driver] all %s runs done" % args.stage)


if __name__ == "__main__":
    main()
