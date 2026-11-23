"""LPIPS and FLIP on the renders (and LPIPS on the baseColor map) for evaluate.py.

Port of comp1 variants/deeptex/render_loss/eval/svbrdf_perceptual_eval.py (the
Perceptual class, the render layout, score_column and metric_check), unchanged
except that the renders come from render_core (GT height for every column) and
there is no offset-removed variant. comp1 ran this in its exp container; here it
runs in the matup container, which has lpips and - after the one-line
docker/matup/requirements.txt addition - flip-evaluator==1.6.0.1 (comp1's pin).

LIGHTS (render_core.build_lights; held-out index 0 is EVAL_LIGHT)
  heldout   all 17 held-out lights: EVAL_LIGHT + 1.4 x fib(16) turned 43.25 deg.
  sheet     EVAL_LIGHT + the grazing / mid / high held-out light comp1's render
            sheets showed.
  LPIPS over `heldout`, FLIP over `sheet` by default (comp1's defaults).

GT-INVALID TEXELS. LPIPS and FLIP filter over neighbourhoods, so a hole cannot
be left out: at texels whose GT normal fails training's rule the prediction's
render is REPLACED by the GT render before either metric (zero difference
there), identically for every column. baseColor is not masked.

METRICS (lower is better for every one)
  render_lpips / render_flip            mean over the light set
  render_lpips_eval / render_flip_eval  EVAL_LIGHT alone
  basecolor_lpips                       LPIPS-alex of the stored sRGB baseColor map,
                                        border-cropped

METRIC SELF-CHECK (exit 3, on the first material): LPIPS(gt, gt) and FLIP(gt, gt)
<= 1e-6, both > 0.05 against 1 - GT under EVAL_LIGHT, and lpips_batch's [-1,1]
shift equal to lpips's own normalize=True path (comp1 metric_check).
"""
import numpy as np
import torch

from variants.eval import render_core as rc

IDENTITY_TOL = 1e-6        # comp1 experiments/gates/gate_protocol.py:121
RESPONSE_MIN = 0.05        # GT vs 1 - GT: far above this for any textured render
SCALARS = ("render_lpips", "render_flip", "render_lpips_eval", "render_flip_eval",
           "basecolor_lpips")


class Perceptual:
    """LPIPS-alex and FLIP as comp1 experiments/common/metrics.py calls them."""

    def __init__(self, device):
        try:
            import flip_evaluator
            import lpips
        except ImportError as e:
            rc.die(2, "%s - rebuild the matup image (docker/matup/requirements.txt pins "
                      "flip-evaluator==1.6.0.1: cd docker && docker compose build matup), "
                      "or pass --no-perceptual" % e)
        # AlexNet's ImageNet weights come from TORCH_HOME (scripts/eval.sh sets
        # /workspace/runs/cache/torch); the linear heads ship with lpips.
        self.net = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        self.flip = flip_evaluator
        self.flip_params = None
        self.versions = {"lpips": getattr(lpips, "__version__", None),
                         "flip_evaluator": getattr(flip_evaluator, "__version__", None)}

    def lpips_batch(self, pred, gt):
        """(n, 3, H, W) display images in [0,1] on the net's device -> (n,) float64.
        LPIPS takes [-1,1] (metrics.py:53-54: x * 2 - 1)."""
        d = self.net(pred * 2.0 - 1.0, gt * 2.0 - 1.0)
        return d.reshape(-1).double().cpu().numpy()

    def lpips_lib01(self, pred, gt):
        """lpips_batch's inputs through lpips's OWN [0,1] path (normalize=True)."""
        try:
            d = self.net(pred, gt, normalize=True)
        except TypeError as e:
            rc.die(3, "lpips %s: LPIPS.forward has no normalize= flag (%s), so the input-range "
                      "self-check cannot run" % (self.versions["lpips"], e))
        return d.reshape(-1).double().cpu().numpy()

    def flip_mean(self, ref, test):
        """Mean FLIP of two HxWx3 float32 sRGB images in [0,1], REFERENCE first:
        flip.evaluate(reference, test, "LDR", applyMagma=False)[1] (metrics.py:131-133)."""
        res = self.flip.evaluate(np.ascontiguousarray(ref, dtype=np.float32),
                                 np.ascontiguousarray(test, dtype=np.float32),
                                 "LDR", applyMagma=False)
        if self.flip_params is None and len(res) > 2:
            p = res[2]
            self.flip_params = ({str(k): v if isinstance(v, (bool, int, float, str)) else str(v)
                                 for k, v in p.items()} if isinstance(p, dict) else str(p))
        return float(res[1])


def map_lpips(metric, pred, gt, device, where):
    """A stored sRGB map (HxWx3 RGB, [0,1]) -> LPIPS vs GT (metrics.lpips_srgb)."""
    if pred.shape != gt.shape or pred.ndim != 3 or pred.shape[2] != 3:
        rc.die(2, "%s: baseColor is %s, GT %s" % (where, pred.shape, gt.shape))

    def t(a):
        a = np.ascontiguousarray(np.clip(a, 0.0, 1.0), dtype=np.float32)
        return torch.from_numpy(a).permute(2, 0, 1)[None].to(device)
    return float(metric.lpips_batch(t(pred), t(gt))[0])


def light_sets(lights, lpips_lights, flip_lights):
    """Held-out indices per metric and the union that gets rendered."""
    named = {"heldout": list(range(len(lights["held_np"]))),
             "sheet": [0] + sorted(int(j) for j in lights["sheet"].values())}
    lp, fl = named[lpips_lights], named[flip_lights]
    union = sorted(set(lp) | set(fl))
    return {"lpips": lp, "flip": fl, "union": union,
            "pos": {j: i for i, j in enumerate(union)},
            "union_t": lights["held"][union]}


def render_images(rmaps, lights_t, lo, hi, side, where):
    """(hi - lo, 3, side, side) display images in [0,1] for lights lo:hi.

    render_display returns (N, 3, l) with N the texels in row-major order, so
    (l, 3, N) -> (l, 3, h, w). The clip is write_image_as_png's (file_utils.py:114).
    """
    d = rc.render_display(rmaps, lights_t[lo:hi], where)
    return d.permute(2, 1, 0).reshape(hi - lo, 3, side, side).clamp(0.0, 1.0)


def gt_stack(gt_rmaps, sets, chunk, side, where):
    """The GT renders of the union lights, cached once per material."""
    return torch.cat([render_images(gt_rmaps, sets["union_t"], lo, hi, side, where)
                      for lo, hi in rc.spans(len(sets["union"]), chunk)])


def hwc(t):
    return t.permute(1, 2, 0).cpu().numpy()


def mean(xs):
    return float(np.mean(xs)) if len(xs) else None


def make_ctx(gt_rmaps, sets, chunk, side, mask_hw, device, where):
    gt = gt_stack(gt_rmaps, sets, chunk, side, where)
    return {"sets": sets, "side": side, "gt": gt,
            "gt_np": {j: hwc(gt[sets["pos"][j]]) for j in set(sets["flip"]) | {0}},
            "invalid": torch.from_numpy(~mask_hw)[None, None].to(device)}


def score_column(rmaps, ctx, metric, chunk, where):
    """LPIPS / FLIP of one column's renders, one light chunk alive at a time.
    GT-invalid texels take the GT render first (module docstring)."""
    sets, gt = ctx["sets"], ctx["gt"]
    lp_set, fl_set = set(sets["lpips"]), set(sets["flip"])
    lp, fl = {}, {}
    for lo, hi in rc.spans(len(sets["union"]), chunk):
        img = render_images(rmaps, sets["union_t"], lo, hi, ctx["side"], where)
        g = gt[lo:hi]
        img = torch.where(ctx["invalid"], g, img)
        held = sets["union"][lo:hi]
        sel = [i for i, j in enumerate(held) if j in lp_set]
        if sel:
            for i, v in zip(sel, metric.lpips_batch(img[sel], g[sel])):
                lp[held[i]] = float(v)
        for i, j in enumerate(held):
            if j in fl_set:
                fl[j] = metric.flip_mean(ctx["gt_np"][j], hwc(img[i]))
        del img
    lp_l = [lp[j] for j in sets["lpips"]]
    fl_l = [fl[j] for j in sets["flip"]]
    return {"render_lpips": mean(lp_l), "render_flip": mean(fl_l),
            "render_lpips_eval": lp.get(0), "render_flip_eval": fl.get(0),
            "render_lpips_per_light": lp_l, "render_flip_per_light": fl_l}


def metric_check(ctx, metric, gt_bc, device, chunk, name):
    """LPIPS / FLIP of GT against itself must be 0, against 1 - GT clearly not,
    and lpips_batch must agree with lpips's own [0,1] path (comp1 metric_check)."""
    sets, gt, pos = ctx["sets"], ctx["gt"], ctx["sets"]["pos"]
    li = [pos[j] for j in sets["lpips"]]
    lp_id = []
    for lo, hi in rc.spans(len(li), chunk):
        x = gt[li[lo:hi]]
        lp_id += [float(v) for v in metric.lpips_batch(x, x)]
    fl_id = [metric.flip_mean(ctx["gt_np"][j], ctx["gt_np"][j]) for j in sets["flip"]]
    bc_id = map_lpips(metric, gt_bc, gt_bc, device, name + " baseColor")
    e = gt[pos[0]:pos[0] + 1]
    lp_neg = float(metric.lpips_batch(1.0 - e, e)[0])
    lp_lib = float(metric.lpips_lib01(1.0 - e, e)[0])
    fl_neg = metric.flip_mean(ctx["gt_np"][0], 1.0 - ctx["gt_np"][0])
    rec = {"material": name, "lpips_gt_gt_max": max(lp_id), "flip_gt_gt_max": max(fl_id),
           "basecolor_lpips_gt_gt": bc_id, "lpips_gt_vs_negative_eval": lp_neg,
           "lpips_gt_vs_negative_eval_lib01": lp_lib,
           "lpips_range_diff_eval": abs(lp_neg - lp_lib),
           "flip_gt_vs_negative_eval": fl_neg, "identity_tol": IDENTITY_TOL,
           "response_min": RESPONSE_MIN}
    bad = []
    for k in ("lpips_gt_gt_max", "flip_gt_gt_max", "basecolor_lpips_gt_gt",
              "lpips_range_diff_eval"):
        if not rec[k] <= IDENTITY_TOL:
            bad.append("%s = %.3g > %g" % (k, rec[k], IDENTITY_TOL))
    for k in ("lpips_gt_vs_negative_eval", "flip_gt_vs_negative_eval"):
        if not rec[k] > RESPONSE_MIN:
            bad.append("%s = %.3g <= %g" % (k, rec[k], RESPONSE_MIN))
    rec["pass"] = not bad
    print("  [check] %s: LPIPS(gt,gt) max %.2e, FLIP(gt,gt) max %.2e, baseColor %.2e; "
          "vs 1-GT (EVAL_LIGHT) LPIPS %.3f FLIP %.3f; LPIPS vs normalize=True %.2e" % (
              name, rec["lpips_gt_gt_max"], rec["flip_gt_gt_max"], bc_id, lp_neg, fl_neg,
              rec["lpips_range_diff_eval"]))
    if bad:
        rc.die(3, "metric self-check FAILED on %s: %s. The LPIPS / FLIP call does not "
                  "behave as metrics.py assumes; every number would be meaningless."
               % (name, "; ".join(bad)))
    return rec
