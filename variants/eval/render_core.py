"""Map loading, MatUp rendering and render scoring for evaluate.py.

Port of comp1 variants/deeptex/render_loss/eval/svbrdf_render_eval.py (the
parts evaluate.py uses), kept line for line where the logic is unchanged.
Runs in the matup container; importing this module puts external/matup_egsr on
sys.path (paths.use_matup()).

What changed against comp1 (everything else is comp1's):
  * HEIGHT. Every loader returns the four material maps only; the position map
    (slot 4) is always the GT one, gt_position(). comp1 rendered bilinear and
    MatUp with the LR height bilinear x4 and DeepTextures with its own SR height.
    The LR-height position survives only for the MatUp self-check
    (matup_lr_position), which must reproduce MatUp's own render_opt.png.
  * MAP SWAP. comp1 swapped one map at a time into the GT (swap_l21_<map>,
    held-out lights). Here one map GROUP at a time - a = baseColor, n = normal,
    rm = roughness + metallic (docs/SPLIT_THEORY.md C-1) - on both the train100
    and the held-out lights, next to the full-SR l2,1.
  * No offset-removed ("off") variant: every number is on the maps as delivered
    (comp1's "raw").
  * Self-check only for MatUp columns (render_opt.png is in every MatUp run);
    comp3 has no committed render_gt / render_bilinear PNGs to check against.
"""
import math
import os
import sys

# Before anything from external/ is imported: Python 3.10 must not leave
# cpython-310.pyc files inside the submodule.
sys.dont_write_bytecode = True

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                *[os.pardir] * 2)))
from variants.util import paths  # noqa: E402

MATUP_ROOT = paths.use_matup()

from matup import data_utils, file_utils, light_utils, render_utils  # noqa: E402
from matup.renderer import Renderer  # noqa: E402

from variants.util import matup_lights, texsr_metrics  # noqa: E402
from variants.util.report import images as rimg  # noqa: E402
from variants.eval import matup_maps as mr  # noqa: E402

# ------------------------------------------------------------------- constants

# Training's GT validity rule (render_tf.py:125-127, applied at :366-371).
NZ_MIN = 1e-3
MASK_MIN_NORM = 0.1
NORMAL_NORM_FLOOR = 1e-4
L21_EPS = 1e-15                  # train.py:18-21
N_TRAIN = 100                    # main_optim.py:122 default --nb_lights
N_HELDOUT_FIB = 16
TINY = float(np.finfo(np.float32).tiny)
U16 = 65535.0

# Index in load_set/position3d's list [albedo, metallic, normal, alpha, pos].
SLOT = {"baseColor": 0, "metallic": 1, "normal": 2, "roughness": 3, "height": 4}
MAP4 = ("baseColor", "normal", "roughness", "metallic")
# Map groups (split/groups.py, SPLIT_THEORY C-1) -> their slots.
GROUPS = ("a", "n", "rm")
GROUP_SLOTS = {"a": (SLOT["baseColor"],), "n": (SLOT["normal"],),
               "rm": (SLOT["roughness"], SLOT["metallic"])}
BAND_NAMES = ("grazing", "mid", "high")

# render_opt.png was rendered by MatUp from its IN-MEMORY float maps; the 16-bit
# _opt PNGs read back here move the re-render by up to 3.3e-4 (comp1
# matup_render.py --verify), so the tolerance is comp1's 5e-4.
SELF_TOL_MATUP = 5e-4
# Above SELF_TOL_MATUP but below this the run is scored with a warning, not stopped. A
# wrong map preparation (normal convention, grid, crop) moves the render by >> 1e-2; a
# few 1e-3 is the 16-bit round trip at the sharpest highlights, e.g. matup_split_st
# brick where alpha sits at Renderer.min_alpha everywhere (1.2e-3, CLAMP_DEATH.md).
SELF_FATAL_MATUP = 1e-2
RENDER_OPT = "render_opt.png"


def die(code, msg):
    print("[fatal] %s" % msg)
    sys.stdout.flush()
    sys.exit(code)


# ------------------------------------------------------------------- reading

def read_raw(path, multi):
    """A map PNG as STORED: float64 in [0,1], RGB, grey maps as one channel
    (file_utils.read_img_as_float minus its normal renormalisation)."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        die(2, "cannot read %s" % path)
    if not multi and img.ndim > 2:
        img = img[:, :, 0]                                    # file_utils.py:84-85
    return file_utils.process_image_as_float(img, multi)     # :54-62 (BGR->RGB, /255|/65535)


def unreadable(paths_):
    """The paths that are missing or do not decode. mr.load_set / read_img_as_float
    end in file_utils.exit_with_message on a bad PNG, and that is exit(0)."""
    return [p for p in paths_
            if not os.path.isfile(p) or cv2.imread(p, cv2.IMREAD_UNCHANGED) is None]


def require_readable(paths_, where):
    bad = unreadable(paths_)
    if bad:
        die(2, "%s: missing or unreadable: %s" % (where, ", ".join(bad)))


def read_height(path):
    """float32 tensor, as matup_render.py:102-103 / :125-126 read it."""
    return torch.from_numpy(file_utils.read_img_as_float(path, False)).float()


def gt_position(mat_dir):
    """position3d of the 16-bit GT height at GT size - slot 4 for EVERY column."""
    h = read_height(os.path.join(mat_dir, "GT", "height.png"))
    return mr.position3d(h, h.shape[0])


def load_gt(mat_dir):
    """matup_render.render_source(..., "GT", ...) without the render (:97-111)."""
    path = os.path.join(mat_dir, "GT")
    albedo, metallic, normal, alpha = mr.load_set(path, "", True, True)
    return [albedo, metallic, normal, alpha]


def load_arm(arm_dir, name):
    """An arm_output folder (DeepTextures) read exactly like a GT folder."""
    albedo, metallic, normal, alpha = mr.load_set(os.path.join(arm_dir, name), "", True, True)
    return [albedo, metallic, normal, alpha]


def load_matup(run_dir):
    """matup_render.render_matup's map preparation (:143-160) minus the height:
    normal_opt is already in MatUp's frame (opengl=False), roughness_opt is
    sqrt(alpha) and is not floored."""
    albedo, metallic, normal, alpha = mr.load_set(run_dir, "_opt", False, False)  # :145
    return [albedo, metallic, normal, alpha]


def matup_lr_position(mat_dir, size):
    """The position MatUp itself rendered render_opt.png with (:151-156): the LR
    height scaled first, then bilinear x4. Self-check only."""
    h_lr = file_utils.read_img_as_float(os.path.join(mat_dir, "SVBRDF", "height.png"), False)
    z = mr.DISP_VAL * (h_lr - 0.5)                                                # :153
    z = cv2.resize(z, (size, size), interpolation=cv2.INTER_LINEAR)               # :154
    pos2d = data_utils.compute_2D_position_grid(size).float()                     # :155
    return torch.cat([pos2d, torch.from_numpy(z).float()[..., None]], dim=-1).float()


def load_bilinear(mat_dir, hr):
    """matup_render.render_bilinear's map preparation (:114-140) minus the height:
    MatUp's initialisation - every map decoded at 128 first and only then resized
    (dataLoader.py:105-128), the normal from its upsampled XY."""
    path = os.path.join(mat_dir, "SVBRDF")
    albedo, metallic, normal, alpha = mr.load_set(path, "", True, True)           # :124

    def up(t):                                                                    # :129-130
        return torch.from_numpy(cv2.resize(
            t.numpy(), (hr, hr), interpolation=cv2.INTER_LINEAR)).float()

    albedo, metallic = up(albedo), up(metallic)[..., None]                        # :132
    alpha = up(alpha)[..., None]                                                  # :133
    normal = data_utils.normal_from_data_vector(up(normal[..., :2]))              # :134
    return [albedo, metallic, normal, alpha]


def gt_valid_mask(gt_dir, b):
    """Training's GT rule (render_tf.py:366-371) on the STORED GT normal."""
    n = 2.0 * read_raw(os.path.join(gt_dir, "normal.png"), True) - 1.0
    sq = (n * n).sum(axis=-1)
    nz_unit = n[..., 2] / np.sqrt(np.maximum(sq, NORMAL_NORM_FLOOR))
    return mr.crop((np.sqrt(sq) >= MASK_MIN_NORM) & (nz_unit >= NZ_MIN), b)


# ------------------------------------------------------------------ rendering

def prepare(maps, b, device):
    """Crop, then MatUp's renormalise/clamp (data_utils.py:74-88) on the device -
    matup_render.render's order (:90-94). Fixes only n_z == 0 exactly (comp1's
    docstring: unreachable for PNG-decoded normals) and returns how many texels
    that touched."""
    m = data_utils.get_renorm_clamped_maps([mr.crop(t, b).to(device) for t in maps])
    nz = m[2][..., 2]
    zero = nz == 0
    k = int(zero.sum())
    if k:
        n = m[2].clone()
        n[..., 2] = torch.where(zero, torch.full_like(nz, TINY), nz)
        m[2] = n
    return m, k


def render_display(rmaps, lights, where):
    """(N, 3, L) display values: render_utils.batch_render_tn (:49-67) without its
    final reshape, keeping its finiteness assertion as an exit code."""
    flat = render_utils.reshape_maps_input_render(rmaps)
    raw = Renderer.torch_Render(*flat, lights, Renderer.z_vector)
    disp = render_utils.process_raw_render(raw)
    if not bool(torch.isfinite(disp).all()):
        die(4, "non-finite render: %s" % where)
    return disp


def spans(n, chunk):
    return [(lo, min(lo + chunk, n)) for lo in range(0, n, chunk)]


def render_chunks(rmaps, lights, chunk, where):
    return [render_display(rmaps, lights[lo:hi], where)
            for lo, hi in spans(lights.shape[0], chunk)]


def score(rmaps, lights, gt_chunks, mask, chunk, where):
    """Per-light masked sums against cached GT chunks: (SSE over RGB, l2,1 terms).

    The texel-light error is train.py:18-21 verbatim (float32); only the masked
    sums over texels are taken in float64.
    """
    sse, l21s = [], []
    for (lo, hi), g in zip(spans(lights.shape[0], chunk), gt_chunks):
        d = render_display(rmaps, lights[lo:hi], where)
        e2 = torch.sum(torch.pow(d - g, 2.0), dim=1)                     # (N, l)
        sse.append((e2 * mask[:, None]).double().sum(dim=0))
        l21s.append((torch.sqrt(torch.clamp(e2, min=L21_EPS)) * mask[:, None])
                    .double().sum(dim=0))
    return torch.cat(sse).cpu().numpy(), torch.cat(l21s).cpu().numpy()


def eval_image(rmaps, light, where):
    h, w = rmaps[0].shape[:2]
    return render_display(rmaps, light, where).reshape(h, w, 3).cpu().numpy()


def psnr(mse, peak=1.0):
    """None for an exact match (texsr_metrics.full_reference's JSON convention)."""
    return None if mse <= 0 else float(10.0 * math.log10(peak * peak / mse))


def light_block(prefix, sse, l21s, n_valid, bands):
    n_l = len(sse)
    out = {prefix + "_psnr": psnr(float(sse.sum()) / (n_valid * n_l * 3.0)),
           prefix + "_l21": float(l21s.sum()) / (n_valid * n_l),
           prefix + "_l21_per_light": [float(v) / n_valid for v in l21s]}
    for b in BAND_NAMES:
        idx = bands[b]
        out["%s_band_l21_%s" % (prefix, b)] = (
            float(l21s[idx].sum()) / (n_valid * len(idx)) if len(idx) else None)
    return out


def to_u16(img):
    """write_image_as_png's quantisation, truncation included (file_utils.py:114-115)."""
    return np.array(U16 * np.clip(img, 0.0, 1.0), dtype=np.uint16)


def column_metrics(rmaps, ctx, chunk, where):
    """Render metrics of one column (all with the GT height, rmaps[4]), the
    per-group swap diagnostic, and the normal angle.

    Returns (metrics, EVAL_LIGHT image)."""
    nv = ctx["n_valid"]
    h_sse, h_l21 = score(rmaps, ctx["held"], ctx["gt_held"], ctx["mask"], chunk,
                         where + " heldout")
    t_sse, t_l21 = score(rmaps, ctx["train"], ctx["gt_train"], ctx["mask"], chunk,
                         where + " train100")
    m = {}
    m.update(light_block("heldout", h_sse, h_l21, nv, ctx["bands_held"]))
    m.update(light_block("train100", t_sse, t_l21, nv, ctx["bands_train"]))

    # SPLIT_THEORY C-1: the GT material with ONE group's SR maps swapped in.
    # Slot 4 (position) is the GT's on both sides, so swapping all three groups
    # is the full-SR column itself.
    for prefix, lights, gt_chunks in (("heldout", ctx["held"], ctx["gt_held"]),
                                      ("train100", ctx["train"], ctx["gt_train"])):
        total = 0.0
        for g in GROUPS:
            swapped = list(ctx["gt_rmaps"])
            for s in GROUP_SLOTS[g]:
                swapped[s] = rmaps[s]
            _, s_l21 = score(swapped, lights, gt_chunks, ctx["mask"], chunk,
                             "%s only-%s %s" % (where, g, prefix))
            v = float(s_l21.sum()) / (nv * lights.shape[0])
            m["%s_l21_only_%s" % (prefix, g)] = v
            total += v
        m["%s_l21_groupsum" % prefix] = total
        m["%s_l21_full_over_groupsum" % prefix] = (m[prefix + "_l21"] / total
                                                   if total > 0 else None)

    img = eval_image(rmaps, ctx["eval"], where + " eval")
    mk = ctx["mask_hw"]
    d = (img.astype(np.float64) - ctx["gt_eval"].astype(np.float64))[mk]
    m["eval_psnr"] = psnr(float((d * d).mean()))
    m["render_grad"] = texsr_metrics.sharpness(to_u16(img))[0]
    m["render_grad_pct_gt"] = (100.0 * m["render_grad"] / ctx["gt_grad"]
                               if ctx["gt_grad"] else None)

    a = rmaps[2].reshape(-1, 3).double()
    g = ctx["gt_rmaps"][2].reshape(-1, 3).double()
    ang = torch.rad2deg(torch.acos(torch.clamp((a * g).sum(dim=1), -1.0, 1.0)))
    m["normal_mae_deg"] = float(ang.cpu().numpy()[ctx["mask_flat"]].mean())
    m["nz_below_min"] = int((rmaps[2][..., 2] < NZ_MIN).sum())
    m["metallic_mean"] = float(rmaps[1].double().mean())
    return m, img


# ---------------------------------------------------------------- self-check

def self_check(ours, run_dir, border, full, where):
    """Our EVAL_LIGHT render of MatUp's _opt maps (LR-height position, as MatUp
    rendered it) vs MatUp's own render_opt.png (comp1 self_check, matup column).

    render_opt.png keeps MatUp's zero rim and is first cropped by the border.
    A failure means the map preparation (normal convention, grid, crop) is wrong
    and every MatUp number would be meaningless: exit 3.
    """
    path = os.path.join(run_dir, RENDER_OPT)
    if not os.path.exists(path):
        print("  ! %s: %s missing - self-check skipped" % (where, path))
        return {"file": path, "pass": None}
    theirs = rimg.read(path)[..., ::-1].astype(np.float64) / U16    # BGR -> RGB
    if theirs.shape[0] == full:
        theirs = mr.crop(theirs, border)
        b_t = border
    else:
        b_t = (full - theirs.shape[0]) // 2
    b = max(border, b_t)
    a = mr.crop(ours.astype(np.float64), b - border)
    t = mr.crop(theirs, b - b_t)
    if a.shape != t.shape:
        die(3, "self-check %s: %s is %s, ours %s after alignment" % (where, path, t.shape, a.shape))
    diff = np.abs(a - t)
    rec = {"file": path, "max_abs": float(diff.max()), "mean_abs": float(diff.mean()),
           "frac_over_tol": float((diff > SELF_TOL_MATUP).mean()),
           "tol": SELF_TOL_MATUP, "fatal": SELF_FATAL_MATUP,
           "pass": bool(diff.max() <= SELF_TOL_MATUP)}
    if rec["max_abs"] > SELF_FATAL_MATUP:
        die(3, "self-check FAILED for %s: max |ours - render_opt| = %.3g > %.3g (%s). "
               "The map preparation here does not reproduce MatUp's render."
            % (where, rec["max_abs"], SELF_FATAL_MATUP, path))
    if not rec["pass"]:
        print("  ! self-check %s: max |ours - render_opt| = %.3g > %.3g on %.2g%% of the pixels "
              "(below the fatal %.3g: 16-bit round trip, scored anyway)"
              % (where, rec["max_abs"], SELF_TOL_MATUP, 100.0 * rec["frac_over_tol"], SELF_FATAL_MATUP))
        sys.stdout.flush()
    return rec


# --------------------------------------------------------------------- lights

def build_lights(device):
    """comp1 build_lights, verbatim."""
    held_np = matup_lights.heldout_lights(N_HELDOUT_FIB)
    # main_optim.py:87-89 verbatim (cast first, multiply in float32 on the device).
    train = 1.4 * light_utils.generate_hemisphere_pts(
        "Fibonacci", sample_count=N_TRAIN, near_z=False).float().to(device)
    train_np = train.cpu().numpy()
    uz = held_np[:, 2].astype(np.float64) / np.linalg.norm(held_np.astype(np.float64), axis=1)
    bands_held = matup_lights.elevation_bands(held_np)
    # Sheet lights: the median-elevation grazing and mid held-out light, and the
    # HIGHEST high one - EVAL_LIGHT (index 0, 67 deg) already shows a high light.
    sheet = {}
    for b in BAND_NAMES:
        idx = sorted((int(i) for i in bands_held[b] if i != 0), key=lambda i: uz[i])
        if idx:
            sheet[b] = idx[-1] if b == "high" else idx[(len(idx) - 1) // 2]
    return {
        "held": torch.from_numpy(held_np).to(device), "held_np": held_np,
        "train": train, "train_np": train_np,
        # matup_render.render (:93): EVAL_LIGHT as a float32 (1, 3).
        "eval": torch.from_numpy(mr.EVAL_LIGHT).float().reshape(1, 3).to(device),
        "bands_held": bands_held,
        "bands_train": matup_lights.elevation_bands(train_np),
        "sheet": sheet, "uz_held": uz,
        "held_min_sep_deg": matup_lights.min_separation_deg(held_np, N_TRAIN),
        "train_equals_matup_lights": bool(np.array_equal(
            train_np, matup_lights.training_lights(N_TRAIN))),
    }


def clean(o):
    """numpy -> Python, non-finite -> None: json.dump would write the bare tokens
    NaN/Infinity, which are not JSON (texsr_metrics.full_reference)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, np.bool_):
        return bool(o)
    return o
