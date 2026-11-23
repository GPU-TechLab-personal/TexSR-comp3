#!/usr/bin/env python3
"""Stage (a) of the shared-vs-split evaluation: DeepTextures inference into arm folders.

Runs in the deeptex container (TF 2.10.1, Python 3.8); scripts/eval.sh infer runs

    python /workspace/variants/eval/infer_deeptex.py --name main

and writes, for every DeepTextures arm whose checkpoint exists (variants/eval/arms.py),

    /workspace/runs/eval/<name>/maps/<arm>/<material>/<map>.png + sr_manifest.json

in variants/util/arm_output.py's contract (GT size, 8-bit, OpenGL normal, grey
maps 1-channel). Stage (b), variants/eval/evaluate.py, reads these folders.

ARMS (port of comp1 variants/deeptex/infer/deeptex_compare.py run() +
variants/deeptex/matsynth/eval_arms.py)

  deeptex_renloss_21   ONE generator, tf.train.Checkpoint(model=G) (paper_loop/loop.py).
                       Each map image goes through G separately: baseColor, normal
                       as RGB; roughness, metallic, height replicated GGG
                       (map_encode.to_rgb8) and collapsed back by the channel MEAN
                       (arm_output.COLLAPSE["mean"]), the decode the render loss
                       trained on. This is arm_output.write_arm verbatim, as
                       eval_arms.py --maps all --scalar-collapse mean ran it in comp1,
                       minus the "render" job (MatUp's render_lr.png SR; not scored here).
                       Its SR height is written but never rendered: evaluate.py
                       renders every arm with the GT height.
  deeptex_split_nb23   THREE generators, tf.train.Checkpoint(G_a=, G_n=, G_rm=)
  deeptex_split_nb7    (split/train.py). Inputs and outputs are packed exactly as
                       split/groups.py does in training:
                         a  = baseColor               -> baseColor  (RGB as is)
                         n  = normal                  -> normal     (RGB as is)
                         rm = [0, mean_c(R), mean_c(M)] -> roughness = out ch1,
                                                         metallic  = out ch2
                       (groups.pack_rm / groups.unpack_rm, called on the same
                       float32 [0,1] images training decodes). No height: there is
                       no height network. All four maps of a material go into ONE
                       folder with ONE merged sr_manifest.json; the RM output's
                       unused channel 0 is recorded there (rm_ch0_255).

Every image is super-resolved whole (128 -> 512; RRDB built with size=None,
deeptex_sr.py). Restore checks are deeptex_sr's / deeptex_compare's: the
checkpoint index and shards resolve, every generator variable is matched
(assert_existing_objects_matched) and restoring changed the weights.

A checkpoint dir that is missing or holds no checkpoint skips that arm with a
warning. An arm whose checkpoint exists but cannot be restored is reported as
FAILED and the remaining arms still run. Exit 0 when at least one arm was
written and none failed, 1 otherwise; 4 on a non-finite output, 10 on OOM.

Before an arm is written, the files this script would write in that arm's
folder (the four or five map PNGs and sr_manifest.json per material) are
removed, so a stale map from an earlier checkpoint can never survive.
"""

import argparse
import json
import os
import sys

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *[os.pardir] * 2))
sys.path.insert(0, REPO)
from variants.util import paths  # noqa: E402

paths.use_deeptex()

import cv2  # noqa: E402
import tensorflow as tf  # noqa: E402

from modules.models import RRDB_Model  # noqa: E402
from modules.utils import load_yaml, set_memory_growth  # noqa: E402

from variants.util import arm_output  # noqa: E402
from variants.deeptex.infer import deeptex_sr  # noqa: E402
from variants.deeptex.split import groups as grp  # noqa: E402
from variants.eval import arms as A  # noqa: E402

die = deeptex_sr.die

# arm_output.ALL_JOBS minus "render" (a MatUp-run source): the five LR maps.
SHARED_JOBS = [j for j in arm_output.ALL_JOBS if j[1] == "svbrdf"]
SPLIT_MAPS = ("baseColor", "normal", "roughness", "metallic")
ALL_FILES = [j[0] + ".png" for j in SHARED_JOBS] + ["sr_manifest.json"]


# ------------------------------------------------------------------ checkpoints

def find_checkpoint(key, ckpt_dir):
    """deeptex_sr.check_checkpoint, but a missing checkpoint is a skip, not exit 3."""
    if not os.path.isdir(ckpt_dir):
        print("[warn] %s: checkpoint dir %s does not exist - arm skipped" % (key, ckpt_dir))
        return None
    if not tf.train.latest_checkpoint(ckpt_dir):
        print("[warn] %s: no checkpoint in %s - arm skipped" % (key, ckpt_dir))
        return None
    try:
        return deeptex_sr.check_checkpoint(ckpt_dir, "")
    except SystemExit:
        print("[warn] %s: checkpoint in %s is incomplete (above) - arm skipped" % (key, ckpt_dir))
        return None


def restore(models, prefix):
    """deeptex_compare.run's restore for every generator in `models`
    ({checkpoint attribute: keras model}): expect_partial (optimizers and D's are
    in the file), every generator variable matched, and each generator's weights
    actually changed by the restore."""
    before = {k: [v.numpy().copy() for v in m.trainable_variables] for k, m in models.items()}
    status = tf.train.Checkpoint(**models).restore(prefix)
    status.expect_partial()
    status.assert_existing_objects_matched()
    for k, m in models.items():
        after = [v.numpy() for v in m.trainable_variables]
        same = sum(1 for b, a in zip(before[k], after) if np.array_equal(b, a))
        if same == len(after):
            raise ValueError("restoring %s changed none of %s's %d weight tensors"
                             % (prefix, k, len(after)))
        if same:
            print("[note] %s: %d/%d weight tensors bit-identical to their init"
                  % (k, same, len(after)))


def make_sr_fn(predict, scale, args):
    def sr_fn(lr, where):
        try:
            return arm_output.super_resolve(predict, lr, scale, args.tile_size, args.overlap)
        except tf.errors.ResourceExhaustedError as e:
            die(10, "OOM on %s - lower --tile-size\n%s" % (where, e))
    return sr_fn


# ----------------------------------------------------------------------- output

def clear_arm(out, mats):
    """Remove only the files this script writes, per material (never a wildcard)."""
    gone = 0
    for mat_dir in mats:
        d = os.path.join(out, os.path.basename(mat_dir.rstrip("/")))
        for f in ALL_FILES:
            p = os.path.join(d, f)
            if os.path.isfile(p):
                os.remove(p)
                gone += 1
    if gone:
        print("  removed %d file(s) of an earlier run from %s" % (gone, out))


def save_grey(path, x):
    """One channel back to an 8-bit 1-channel PNG: arm_output.save_sr's rint / clip,
    without a collapse (the channel was already chosen)."""
    u8 = np.rint(np.clip(x.astype(np.float64), 0.0, 1.0) * 255.0).astype(np.uint8)
    if not cv2.imwrite(path, u8, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise RuntimeError("cv2.imwrite failed for " + path)
    return list(u8.shape)


def provenance(key, spec, cfg, prefix, step):
    return {"arm": key, "kind": spec["kind"], "checkpoint": prefix, "step": step,
            "sub_name": cfg["sub_name"], "scale": cfg["scale"], "cfg": spec["cfg"],
            "network_G": cfg["network_G"],
            "note": "height is never rendered: evaluate.py renders every arm with the GT height"}


# ------------------------------------------------------------------------- arms

def run_shared(key, spec, cfg, prefix, step, mats, out, args):
    model = RRDB_Model(None, cfg["ch_size"], cfg["network_G"])
    restore({"model": model}, prefix)
    sr_fn = make_sr_fn(deeptex_sr.predictor(model), cfg["scale"], args)
    done, skipped = arm_output.write_arm(
        mats, out, SHARED_JOBS, sr_fn,
        lambda path, sr, kind: arm_output.save_sr(path, sr, kind, "mean"),
        ("mean", arm_output.COLLAPSE["mean"]),
        provenance(key, spec, cfg, prefix, step), run="", die=die)
    return done, skipped


def pack_rm(rough, metal):
    """(H, W, 3) GGG roughness and metallic -> (H, W, 3) [0, R, M] via groups.pack_rm."""
    return grp.pack_rm(tf.constant(rough[None]), tf.constant(metal[None]))[0].numpy()


def unpack_rm(y_rm):
    """(H, W, 3) RM output -> (roughness, metallic), each (H, W): groups.unpack_rm's
    channels 1 and 2 (tiled GGG there; one of the equal channels here)."""
    r, m = grp.unpack_rm(tf.constant(y_rm[None]))
    return r[0, :, :, 0].numpy(), m[0, :, :, 0].numpy()


def run_split(key, spec, cfg, prefix, step, mats, out, args):
    if int(cfg["ch_size"]) != 3:
        raise ValueError("%s: ch_size %s, the split groups are 3-channel" % (key, cfg["ch_size"]))
    G = {g: RRDB_Model(None, 3, cfg["network_G"], name="RRDB_model_" + g) for g in grp.GROUPS}
    restore({"G_" + g: G[g] for g in grp.GROUPS}, prefix)
    sr_fn = {g: make_sr_fn(deeptex_sr.predictor(G[g]), cfg["scale"], args) for g in grp.GROUPS}
    prov = provenance(key, spec, cfg, prefix, step)

    os.makedirs(out, exist_ok=True)
    done, skipped = [], []
    for mat_dir in mats:
        name = os.path.basename(mat_dir.rstrip("/"))
        lr, src = {}, {}
        for m in SPLIT_MAPS:
            p = os.path.join(mat_dir, "SVBRDF", m + ".png")
            if not os.path.exists(p):
                skipped.append("%s/%s (%s missing)" % (name, m, os.path.basename(p)))
                continue
            lr[m], src[m] = arm_output.load_as_rgb01(p)
            src[m] = {"src": p, "source_encoding": src[m], "lr_shape": list(lr[m].shape)}
        if len(lr) != len(SPLIT_MAPS):
            print("  %-28s SKIP (LR map missing)" % name)
            continue

        x = {"a": lr["baseColor"], "n": lr["normal"],
             "rm": pack_rm(lr["roughness"], lr["metallic"])}
        sr = {}
        for g in grp.GROUPS:
            sr[g] = sr_fn[g](x[g], "%s/%s" % (name, g))
            if not np.isfinite(sr[g]).all():
                die(4, "non-finite output on %s/%s (%s)" % (name, g, key))

        out_dir = os.path.join(out, name)
        os.makedirs(out_dir, exist_ok=True)
        wrote = {}
        for m, g in (("baseColor", "a"), ("normal", "n")):
            shape, _ = arm_output.save_colour(os.path.join(out_dir, m + ".png"), sr[g])
            wrote[m] = dict(src[m], kind="colour", group=g, sr_shape=shape)
        rough, metal = unpack_rm(sr["rm"])
        for m, ch, v in (("roughness", 1, rough), ("metallic", 2, metal)):
            shape = save_grey(os.path.join(out_dir, m + ".png"), v)
            wrote[m] = dict(src[m], kind="scalar", group="rm", rm_channel=ch, sr_shape=shape)
        ch0 = sr["rm"][..., 0].astype(np.float64)

        manifest = {"material": name}
        manifest.update(prov)
        manifest.update({
            "matup_run": None,
            "scalar_collapse": "rm_channel",
            "collapse_weights": {"roughness": [0.0, 1.0, 0.0], "metallic": [0.0, 0.0, 1.0]},
            "groups": {"a": "baseColor", "n": "normal", "rm": "[0, roughness, metallic]"},
            "rm_ch0_255": {"mean": float(ch0.mean() * 255.0),
                           "max_abs": float(np.abs(ch0).max() * 255.0)},
            "outputs": wrote})
        with open(os.path.join(out_dir, "sr_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        done.append(name)
        print("  %-28s %s   rm ch0 mean %.2f/255" % (
            name, ", ".join(sorted(wrote)), manifest["rm_ch0_255"]["mean"]))
    return done, skipped


# ------------------------------------------------------------------------- main

def parse_ckpt_overrides(specs):
    out = {}
    for s in specs:
        k, _, d = s.partition("=")
        if k not in A.DEEPTEX or not d:
            die(2, "--ckpt wants ARM=DIR with ARM in %s, got %r" % (", ".join(A.DEEPTEX_ORDER), s))
        out[k] = d
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="main", help="evaluation name: runs/eval/<name>")
    ap.add_argument("--eval-root", default=A.EVAL_ROOT)
    ap.add_argument("--root", default=A.MATSYNTH_ROOT, help="evaluation materials (<mat>/SVBRDF)")
    ap.add_argument("--arms", default="all",
                    help="comma-separated subset of %s, or 'all'" % ",".join(A.DEEPTEX_ORDER))
    ap.add_argument("--ckpt", action="append", default=[],
                    help="ARM=CHECKPOINT_DIR, overrides arms.py's default (repeatable)")
    ap.add_argument("--material", default="")
    ap.add_argument("--tile-size", type=int, default=0)
    ap.add_argument("--overlap", type=int, default=16)
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()

    keys = list(A.DEEPTEX_ORDER) if args.arms == "all" else [k for k in args.arms.split(",") if k]
    bad = [k for k in keys if k not in A.DEEPTEX]
    if bad or not keys:
        die(2, "--arms: unknown %s (known: %s)" % (bad, ", ".join(A.DEEPTEX_ORDER)))
    ckpts = parse_ckpt_overrides(args.ckpt)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    set_memory_growth()
    paths.check_pin("DeepTextures")

    mats = arm_output.materials(args.root, args.material)
    if not mats:
        die(2, "no materials (<mat>/SVBRDF) under %s" % args.root)
    out_root = A.maps_dir(args.name, args.eval_root)
    print("[infer] %d material(s) from %s -> %s" % (len(mats), args.root, out_root))

    written, skipped_arms, failed = [], [], []
    for key in keys:
        spec = A.DEEPTEX[key]
        ckpt_dir = ckpts.get(key, spec["ckpt_dir"])
        fallback = spec.get("fallback_ckpt_dir") if key not in ckpts else None
        if fallback and not (os.path.isdir(ckpt_dir) and tf.train.latest_checkpoint(ckpt_dir)) \
                and os.path.isdir(fallback) and tf.train.latest_checkpoint(fallback):
            print("[infer] %s: nothing in %s, using the locally trained %s"
                  % (key, ckpt_dir, fallback))
            ckpt_dir = fallback
        print("[infer] %s (%s) checkpoint dir %s" % (key, spec["kind"], ckpt_dir))
        found = find_checkpoint(key, ckpt_dir)
        if found is None:
            skipped_arms.append(key)
            continue
        prefix, step, _ = found
        cfg = load_yaml(spec["cfg"])
        out = os.path.join(out_root, key)
        clear_arm(out, mats)
        run = run_split if spec["kind"] == "split" else run_shared
        try:
            done, skipped = run(key, spec, cfg, prefix, step, mats, out, args)
        except (AssertionError, ValueError, tf.errors.OpError) as e:
            print("[FAILED] %s: %s" % (key, str(e).splitlines()[0] if str(e) else repr(e)))
            failed.append(key)
            tf.keras.backend.clear_session()
            continue
        print("[infer] %s: %d material(s) -> %s (step %s)" % (key, len(done), out, step))
        for s in skipped:
            print("  [skip] %s" % s)
        if done:
            written.append(key)
        tf.keras.backend.clear_session()

    print("[infer] written: %s; skipped (no checkpoint): %s; failed: %s" % (
        ", ".join(written) or "-", ", ".join(skipped_arms) or "-", ", ".join(failed) or "-"))
    if failed or not written:
        sys.exit(1)


if __name__ == "__main__":
    main()
