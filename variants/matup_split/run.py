#!/usr/bin/env python3
"""One MatUp run on one material, with the shared or the split MLP and a chosen seed.

Runs in the matup container (scripts/matup_split.sh calls it through driver.py):

    python /workspace/variants/matup_split/run.py --material <eval dir>/<name> \
        --cache /workspace/runs/matup_split/cache/<name> --stage prep
    python /workspace/variants/matup_split/run.py --material ... --cache ... \
        --stage optim --mlp split --seed 1997 --out /workspace/runs/matup_split/matup_split/s1/<name>

external/matup_egsr/main.py runs unmodified as __main__ via runpy, with MatUp's
own defaults (--upscaler swinir, the released SwinIR-L weights) plus
--ogl_normals (MatSynth normals are OpenGL; variants/util/matsynth_extract.py).
Around it this file changes four things, by patching module attributes before
main.py imports them (MatUp's files on disk are never edited):

1. STAGES. MatUp renders the LR material under 100 lights, upscales the renders
   with SwinIR-L, then optimises. Here `prep` does the first two into --cache
   once per material, and `optim` skips them and reads that cache. Every arm and
   seed is therefore fitted against byte-identical target renders.
2. KEEP THE CACHE. main_optim deletes render_dir and upsamplings_dir when it ends
   (main_optim.py:102-104); its shutil.rmtree is replaced by a no-op.
3. THE MLP. --mlp split swaps main_optim.FullyConnected for SplitMLP
   (split_mlp.py); --mlp shared leaves MatUp's own class. --split-width W gives
   the three sub-MLPs width W instead of MatUp's 128 (parameter-matched arms).
4. THE SEED. main_optim seeds numpy and torch with 1996 (main_optim.py:36-37).
   --seed S makes those two calls seed S instead; S = 1996 is MatUp as released.
5. THE METALLIC REGULARISER. --alpha-reg X is passed to main.py as --alpha_reg X
   (weight of the metallic term, train.loss_fn; MatUp's default 0.1). Only the
   *_noreg arms set it (0, docs/SPLIT_THEORY.md B1-4); omitted, MatUp's default holds.
6. CLAMP-DEATH TESTS (docs/CLAMP_DEATH.md). All three leave the default run unchanged.
   --probe       clamp_probe.py: per-channel clamp statistics at initialisation and
                 after every epoch -> <out>/probe.json (adds --save_interval 1 for
                 MatUp's per-epoch eval; the per-epoch PNGs are not written)
   --zero-init   the last layer of every MatUp MLP (shared, or each split sub-MLP)
                 starts at 0, so f(x) = 0 and every texel starts at the bilinear init
                 (zeroed after construction: the seed's random stream is unchanged)
   --st-clamp    training clamps straight-through: forward as MatUp, backward
                 d clamp(z)/dz = 1 everywhere (albedo, metallic, alpha)

Outputs: MatUp writes <out>/<timestamp>/{baseColor,metallic,normal,roughness,render}_opt.png
(512^2 with an 8 px border, DirectX normals). This file adds <out>/run.json
(arm, seed, parameter count, seconds) once MatUp returns.
"""

import argparse
import glob
import json
import os
import runpy
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *[os.pardir] * 2))
MATUP = os.path.join(REPO, "external", "matup_egsr")
MATUP_SEED = 1996                      # main_optim.py:36-37
MATUP_ALPHA_REG = 0.1                  # main.py --alpha_reg default
N_LIGHTS = 100                         # main.py --nb_lights default


def n_png(d):
    return len(glob.glob(os.path.join(d, "*.png"))) if os.path.isdir(d) else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--material", required=True, help="<eval set>/<name> holding SVBRDF/")
    ap.add_argument("--cache", required=True, help="per-material render cache")
    ap.add_argument("--stage", required=True, choices=["prep", "optim"])
    ap.add_argument("--mlp", default="shared", choices=["shared", "split"])
    ap.add_argument("--seed", type=int, default=MATUP_SEED)
    ap.add_argument("--split-width", type=int, default=0,
                    help="split only: sub-MLP hidden width (0 = MatUp's --hid_layer_width, 128)")
    ap.add_argument("--alpha-reg", type=float, default=None,
                    help="optim: metallic regulariser weight (main.py --alpha_reg; default MatUp's 0.1)")
    ap.add_argument("--probe", action="store_true", help="optim: write <out>/probe.json (clamp_probe.py)")
    ap.add_argument("--zero-init", action="store_true", help="optim: last MLP layer starts at 0")
    ap.add_argument("--st-clamp", action="store_true", help="optim: straight-through clamp in training")
    ap.add_argument("--out", default="", help="optim: output folder of this run")
    args = ap.parse_args()

    svbrdf = os.path.join(args.material, "SVBRDF")
    if not os.path.isfile(os.path.join(svbrdf, "baseColor.png")):
        sys.exit("[fatal] no %s/baseColor.png - run: bash scripts/data.sh evalset" % svbrdf)
    render_dir = os.path.join(args.cache, "radiance_LR")
    up_dir = os.path.join(args.cache, "radiance_Upscaled")
    if args.stage == "optim":
        if not args.out:
            sys.exit("[fatal] --stage optim needs --out")
        if n_png(render_dir) != N_LIGHTS or n_png(up_dir) != N_LIGHTS:
            sys.exit("[fatal] cache %s incomplete (LR %d, upscaled %d of %d) - run the prep stage"
                     % (args.cache, n_png(render_dir), n_png(up_dir), N_LIGHTS))

    os.chdir(MATUP)                    # sr_swinir.py loads ./pretrained_models/...
    sys.path.insert(0, MATUP)
    sys.path.insert(0, REPO)
    import numpy as np
    import torch
    import main_optim as mo
    import render_lowres
    import sr_swinir

    info = {"stage": args.stage}
    if args.stage == "prep":
        mo.main_optim = lambda a: print("[matup_split] prep stage: optimisation skipped")
    else:
        render_lowres.main_render = lambda a: print("[matup_split] LR renders from cache")
        sr_swinir.main_upsample_swinir = lambda a: print("[matup_split] SwinIR renders from cache")

        class _KeepCache(object):
            @staticmethod
            def rmtree(path, *a, **k):
                print("[matup_split] keeping %s" % path)
        mo.shutil = _KeepCache

        if args.split_width and args.mlp != "split":
            sys.exit("[fatal] --split-width needs --mlp split")
        if args.mlp == "split":
            from variants.matup_split.split_mlp import SplitMLP, with_width
            mo.FullyConnected = with_width(args.split_width) if args.split_width else SplitMLP

        built = {}
        cls = mo.FullyConnected

        def counted(*a, **k):
            m = cls(*a, **k)
            if args.zero_init:                 # after construction: same random stream
                from matup.model import FullyConnected
                for mod in m.modules():
                    if isinstance(mod, FullyConnected):
                        torch.nn.init.zeros_(mod.mlp[-1].weight)
                        torch.nn.init.zeros_(mod.mlp[-1].bias)
            built["params"] = sum(p.numel() for p in m.parameters())
            print("[matup_split] mlp=%s width=%s params=%d seed=%d" % (
                args.mlp, args.split_width or "default", built["params"], args.seed), flush=True)
            return m
        mo.FullyConnected = counted

        torch_seed, np_seed = torch.manual_seed, np.random.seed

        def seed_torch(s):
            return torch_seed(args.seed if s == MATUP_SEED else s)

        def seed_np(s):
            return np_seed(args.seed if s == MATUP_SEED else s)
        mo.torch.manual_seed = seed_torch
        mo.np.random.seed = seed_np

        if args.st_clamp:
            import matup.data_utils as du
            clamped = du.get_renorm_clamped_maps

            def st_clamped(in_maps):          # forward as MatUp, gradient 1 through the clamps
                raw = {i: in_maps[i] for i in (0, 1, 3)}      # albedo, metallic, alpha
                out = clamped(in_maps)
                for i, r in raw.items():
                    out[i] = r + (out[i] - r).detach()
                return out
            du.get_renorm_clamped_maps = st_clamped
        if args.probe:
            from matup.renderer import Renderer
            from variants.matup_split import clamp_probe
            probe = clamp_probe.Probe(os.path.join(args.out, "probe.json"),
                                      Renderer.min_alpha, 17)          # main.py --tile_width
            clamp_probe.install(probe, 10)                             # main.py --nb_epochs
        os.makedirs(args.out, exist_ok=True)
        info.update({"mlp": args.mlp, "seed": args.seed, "built": built})

    argv = ["main.py", "--upscaler", "swinir", "--ogl_normals",
            "--input_mat", svbrdf + "/", "--render_dir", render_dir + "/",
            "--upsamplings_dir", up_dir + "/",
            "--output_folder", (args.out or os.path.join(args.cache, "_unused")) + "/"]
    if args.stage == "optim" and args.alpha_reg is not None:
        argv += ["--alpha_reg", repr(args.alpha_reg)]
    if args.stage == "optim" and args.probe:
        argv += ["--save_interval", "1"]
    print("[matup_split] %s" % " ".join(argv), flush=True)
    sys.argv = argv
    t0 = time.time()
    runpy.run_path(os.path.join(MATUP, "main.py"), run_name="__main__")
    dt = time.time() - t0

    if args.stage == "prep":
        if n_png(render_dir) != N_LIGHTS or n_png(up_dir) != N_LIGHTS:
            sys.exit("[fatal] prep wrote LR %d / upscaled %d renders, expected %d"
                     % (n_png(render_dir), n_png(up_dir), N_LIGHTS))
        print("[matup_split] cache ready: %s (%.0f s)" % (args.cache, dt))
        return
    outs = sorted(glob.glob(os.path.join(args.out, "*", "baseColor_opt.png")))
    if not outs:
        sys.exit("[fatal] MatUp wrote no baseColor_opt.png under %s" % args.out)
    with open(os.path.join(args.out, "run.json"), "w") as f:
        json.dump({"material": os.path.basename(os.path.normpath(args.material)),
                   "mlp": args.mlp, "split_width": args.split_width or None, "seed": args.seed,
                   "alpha_reg": MATUP_ALPHA_REG if args.alpha_reg is None else args.alpha_reg,
                   "zero_init": args.zero_init, "st_clamp": args.st_clamp, "probe": args.probe,
                   "params": info["built"].get("params"), "seconds": round(dt, 1),
                   "maps": os.path.relpath(os.path.dirname(outs[-1]), args.out),
                   "cache": args.cache}, f, indent=2)
    print("[matup_split] done %s (%.0f s)" % (args.out, dt), flush=True)


if __name__ == "__main__":
    main()
