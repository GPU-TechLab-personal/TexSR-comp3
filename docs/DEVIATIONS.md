# Deviations from TexSR-comp2

comp3 holds only what the split experiment needs. The maths, and what differs from the original papers, is [SPLIT_THEORY.md](SPLIT_THEORY.md). This file lists what differs from how comp2 ran things.

## 1. What was carried over, and what was not

- **Copied unchanged:** `variants/deeptex/` (the renloss_21 trainer, grouped dataset, renderer port, hooks), `variants/util/`, `variants/matup/baseline/`, and `docker/{dataprep,deeptex,matup}`. The submodules (DeepTextures, esrgan-tf2, matup_egsr, KAIR) are at comp2's commits.
- **Not carried over:** PBR-SR (submodule, `variants/pbr_sr/`, `scripts/init.sh`, the `pbrsr` image) and the MatUp SwinIR fine-tune (`variants/matup_ft/`, `scripts/matup_ft.sh`).
  - Evaluation is on the 12 MatSynth evaluation materials only, not on Cradle_Globe.
  - MatUp's SwinIR-L upscales renders, not maps, so splitting it by map group has no meaning (SPLIT_THEORY B1). The split experiment uses the released SwinIR-L weight, as the MatUp paper does.
- **Removed:** `scripts/deeptex_renloss_21.sh`. Its `grouped` stage is now `scripts/data.sh grouped`. Retraining the shared arm, if comp2's checkpoint cannot be imported, is `scripts/deeptex_split.sh train shared`.

## 2. Data lives on the data drive

- **comp2:** derived data sat under `runs/` inside the checkout.
  - `runs/matsynth/`: the evaluation set
  - `runs/datasets/matsynth/`: the pairs
  - `runs/deeptex/datasets/matsynth_grouped/`: the grouped TFRecord
- **comp3:** all of it is under `<DATA_ROOT>/TexSR-data/` (D:/TexSR-data on the lab PC; `/data/TexSR-data` in the containers), next to MatSynth. This is the rule from now on: datasets and other data do not go inside a checkout.

| what | comp2 | comp3 |
|---|---|---|
| evaluation set | `runs/matsynth/` | `TexSR-data/matsynth/` |
| training pairs | `runs/datasets/matsynth/` | `TexSR-data/datasets/matsynth/` |
| grouped TFRecord | `runs/deeptex/datasets/matsynth_grouped/` | `TexSR-data/deeptex/matsynth_grouped/` |
| renloss_21 checkpoints | `runs/deeptex/deeptex_renloss_21/checkpoints/` | `TexSR-data/checkpoints/deeptex_renloss_21/` |

- `scripts/data.sh import <comp2>` copies them (robocopy, resumable). `scripts/data.sh verify` re-runs the pair/eval degradation check and the grouped TFRecord byte verification on the copy.
- The defaults of `matsynth_extract.py`, `matsynth_make_pairs.py`, `texsr_build_grouped_tfrecord.py` and `matup_height.py`, and the `train_dataset.path` of `deeptex_renloss_21.yaml`, point at the new locations. Nothing else in the copied code changed.
- **`dataprep` mounts `/data` read-write**, because the evaluation set and pairs are now written there. It writes only under `/data/TexSR-data`.

## 3. DeepTex split trainer (`variants/deeptex/split/`)

- It is a separate entry point (`split/train.py`), not a fourth experiment of `paper_loop/main.py`. The paper loop is built around one G and one D; the split step has three of each, six optimizers, and one shared render term.
- **Reused as is:** `grouped_dataset.load_grouped_dataset` (same stream, crops and resume), `render_tf.RenderLoss` (same decoder, renderer and lights), `render_loss.hooks.check_grouped_dataset` / `nan_debug_exit` / `nan_reason`, and `hooks_common` (seed, safety checks, `MetricsWindow`, `weights_sha`).
- **Log lines:**
  - `[metrics]` is unchanged.
  - `[render] step=N l21=...` adds `l21_a`, `l21_n`, `l21_rm`: the l2,1 with only that group's SR and every other map GT.
  - `[terms]` has one block per group.
- **Checkpoint:** one `tf.train.Checkpoint` with keys `G_a G_n G_rm D_a D_n D_rm opt_G_* opt_D_* step`. `max_to_keep` stays 3 (30k / 35k / 40k).

## 4. MatUp split (`variants/matup_split/`)

- MatUp's `main.py` runs unmodified through runpy. The changes are module-attribute patches made before it imports them:
  - LR render + SwinIR upscaling run once per material into `runs/matup_split/cache/`. Every arm and seed reads the same target renders.
  - `main_optim`'s `shutil.rmtree` of those folders becomes a no-op.
  - `FullyConnected` becomes `SplitMLP` for the split arm.
  - MatUp's fixed seed 1996 becomes the run's seed. s0 = 1996 is MatUp as released; s1 = 1997, s2 = 1998.
- **Normals:** `--ogl_normals`, because MatSynth is OpenGL (as in comp1's evaluation-set runs).

## 5. Evaluation

- Every arm is rendered with the **GT height**. The split arms predict no height, so for the shared arms the SR height is replaced by GT before rendering.
- **Map metrics:** PSNR and SSIM for albedo, roughness and metallic; mean normal angle error.
- **Render metrics:** PSNR, l2,1, LPIPS and FLIP, on train100 and held-out lights.
- **Diagnostic:** per group, the l2,1 with only that group's SR.
- **Images:** the `matup` image gains `flip-evaluator` (comp1 had a separate `exp` image for it).
