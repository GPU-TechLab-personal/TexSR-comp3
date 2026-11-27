# TexSR-comp3

Map-group split networks for PBR texture super-resolution (x4). Does a network that only ever sees one map group (albedo / normal / roughness+metallic) beat one network shared by every map?

Two methods, each run **shared** (as in TexSR-comp2) and **split**:

- **DeepTextures + MatUp render loss** (`deeptex_renloss_21` of comp2): three RRDB generators, nb=23 each (3x parameters) and nb=7 each (~0.93x), trained **jointly**. Their outputs form one material, rendered under MatUp's 100 lights, and the l2,1 render loss backpropagates into all three.
- **MatUp** (EGSR 2024): its per-material MLP split into three group MLPs, each seeing only its own channels. Seeds 0-2.

The formulas, what differs from the original papers and why, are in **[docs/SPLIT_THEORY.md](docs/SPLIT_THEORY.md)**. Every command, in order, is in **[COMMANDS.txt](COMMANDS.txt)**. What differs from comp2 is in [docs/DEVIATIONS.md](docs/DEVIATIONS.md).

| path | what |
|---|---|
| `external/` | DeepTextures, esrgan-tf2, MatUp, KAIR as git submodules at comp2's commits, never edited |
| `docker/` | one image per method (`dataprep`, `deeptex`, `matup`) |
| `scripts/setup.sh` | images, weights, GPU check (`data`, `deeptex`, `matup`) |
| `scripts/data.sh` | the data drive: import comp2's data, or build it from MatSynth |
| `scripts/deeptex_split.sh` | DeepTex split: unit test, parameter counts, smoke, 40k training |
| `scripts/matup_split.sh` | MatUp shared / split MLP on the 12 evaluation materials |
| `scripts/eval.sh` | every arm on the 12 evaluation materials: montages (`montage`), map and render metrics (`score`) |
| `variants/deeptex/split/` | the joint split trainer, group packing, tests |
| `variants/matup_split/` | SplitMLP, the MatUp wrapper, the run driver |
| `variants/eval/` | inference, montages (`montage.py`, `sheets.py`) and metrics for every arm |
| `D:/TexSR-data/` | data (outside the repository): evaluation set, pairs, grouped TFRecord, baseline checkpoints |
| `runs/` | everything a run produces (gitignored except `runs/eval/<experiment>/<checkpoint>/{report.json,summary.md,montage/}`: `deeptex_shared_vs_split/{30k,35k,40k}`, `matup_shared_vs_split/ep10`) |

The upstream licenses apply to each submodule.
