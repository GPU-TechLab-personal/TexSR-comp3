"""The arms of the comp3 shared-vs-split evaluation (docs/SPLIT_THEORY.md, section D).

Pure Python, Python 3.8 clean: imported by infer_deeptex.py (deeptex container,
TF 2.10.1 / Python 3.8) and evaluate.py (matup container, torch 2.7.1 / Python 3.10).

Report column order (summary.md, report.json):

  bilinear              LR maps bilinear x4 - MatUp's own initialisation for the
                        renders, cv2.INTER_LINEAR of the stored LR PNG for the map
                        metrics (comp1's bilinear column)
  deeptex_renloss_21    shared DeepTextures G (RRDB nb=23) + render loss, TF
  deeptex_split_nb23    three G's (a / n / rm), nb=23 each, trained jointly
  deeptex_split_nb7     three G's, nb=7 each
  matup_shared_s0..s2   MatUp, one MLP per material (MatUp as released), seed k
  matup_split_s0..s2    MatUp, three group MLPs per material, seed k

Pairs (split vs its shared counterpart, paired per material):
  deeptex_split_nb23 - deeptex_renloss_21, deeptex_split_nb7 - deeptex_renloss_21,
  matup_split_s<k> - matup_shared_s<k> for every seed k.
"""

import os

# Where things are, as container paths (docker/compose.yaml: /workspace = repo,
# /data = the data drive, host D:/TexSR-data is /data/TexSR-data).
EVAL_ROOT = "/workspace/runs/eval"
MATSYNTH_ROOT = "/data/TexSR-data/matsynth"
MATUP_RUNS = "/workspace/runs/matup_split"

SEEDS = (0, 1, 2)
MATUP_ARMS = ("matup_shared", "matup_split")

# DeepTextures arms: config, default checkpoint dir, and how the generator(s) sit
# in the tf.train.Checkpoint.
#   shared: tf.train.Checkpoint(model=G)                   (paper_loop/loop.py:57-61)
#   split:  tf.train.Checkpoint(G_a=..., G_n=..., G_rm=...) (split/train.py, objs["G_" + g])
DEEPTEX = {
    "deeptex_renloss_21": {
        "kind": "shared",
        "cfg": "/workspace/variants/deeptex/render_loss/deeptex_renloss_21.yaml",
        "ckpt_dir": "/data/TexSR-data/checkpoints/deeptex_renloss_21/esrgan_40k",
        # used only when ckpt_dir holds no checkpoint: the config's own
        # checkpoint_root/sub_name, where `bash scripts/deeptex_split.sh train shared`
        # retrains it if comp2's checkpoint could not be imported.
        "fallback_ckpt_dir": "/workspace/runs/deeptex/deeptex_renloss_21/checkpoints/esrgan_40k",
    },
    "deeptex_split_nb23": {
        "kind": "split",
        "cfg": "/workspace/variants/deeptex/split/deeptex_split_nb23.yaml",
        "ckpt_dir": "/workspace/runs/deeptex/deeptex_split_nb23/checkpoints/esrgan_40k",
    },
    "deeptex_split_nb7": {
        "kind": "split",
        "cfg": "/workspace/variants/deeptex/split/deeptex_split_nb7.yaml",
        "ckpt_dir": "/workspace/runs/deeptex/deeptex_split_nb7/checkpoints/esrgan_40k",
    },
}
DEEPTEX_ORDER = ("deeptex_renloss_21", "deeptex_split_nb23", "deeptex_split_nb7")

DISPLAY = {
    "bilinear": "bilinear",
    "deeptex_renloss_21": "DeepTex shared (renloss_21, nb23)",
    "deeptex_split_nb23": "DeepTex split nb23 x3",
    "deeptex_split_nb7": "DeepTex split nb7 x3",
    "matup_shared": "MatUp shared MLP",
    "matup_split": "MatUp split MLP",
}


def matup_key(arm, seed):
    return "%s_s%d" % (arm, seed)


def columns(seeds=SEEDS):
    """Every column in report order: dicts with key, kind, display (+ arm, seed for MatUp)."""
    cols = [{"key": "bilinear", "kind": "bilinear", "display": DISPLAY["bilinear"]}]
    for k in DEEPTEX_ORDER:
        cols.append({"key": k, "kind": "deeptex", "display": DISPLAY[k],
                     "split": DEEPTEX[k]["kind"] == "split"})
    for arm in MATUP_ARMS:
        for s in seeds:
            cols.append({"key": matup_key(arm, s), "kind": "matup", "arm": arm, "seed": s,
                         "display": "%s s%d" % (DISPLAY[arm], s),
                         "split": arm == "matup_split"})
    return cols


def pairs(seeds=SEEDS):
    """(split key, shared key) in report order."""
    out = [("deeptex_split_nb23", "deeptex_renloss_21"),
           ("deeptex_split_nb7", "deeptex_renloss_21")]
    out += [(matup_key("matup_split", s), matup_key("matup_shared", s)) for s in seeds]
    return out


def eval_dir(name, root=EVAL_ROOT):
    return os.path.join(root, name)


def maps_dir(name, root=EVAL_ROOT):
    """runs/eval/<name>/maps: one <arm>/<material>/<map>.png folder per arm."""
    return os.path.join(root, name, "maps")
