#!/usr/bin/env python3
"""DeepTextures + MatUp render loss, split into three group generators trained JOINTLY.

    python /workspace/variants/deeptex/split/train.py --cfg_path <split yaml> [flags]
    (scripts/deeptex_split.sh runs exactly this, detached)

The maths is docs/SPLIT_THEORY.md A2. In one step:

  1. lr5 / hr5 (B, 5, ...) -> per group inputs / targets (groups.split_stack):
       a = baseColor, n = normal, rm = [0, roughness, metallic]
  2. G_a, G_n, G_rm forward; their outputs and the GT height make ONE material
     (groups.compose), rendered under MatUp's 100 lights -> L_ren (unweighted l2,1).
     The GT render is computed before the tape, as in deeptex_renloss_21.
  3. per group:  L_g = reg + w_pixel*pix + w_feature*VGG + w_gan*RaGAN(D_g) + w_g*L_ren
     and theta_g takes the gradient of ITS OWN L_g only. L_ren depends on all three
     outputs, so every generator gets render gradient through its own slots: joint
     training. The render backward runs once (dL_ren/d sr_g), and each generator's
     gradient is tape.gradient([own_g, sr_g], theta_g, [1, w_g * dL_ren/d sr_g]),
     which equals grad_theta_g (own_g + w_g * L_ren).
     rm_loss: grey (default packed) computes rm's pixel / VGG / RaGAN terms on the 2B
     GGG roughness and metallic images instead of the B packed [0,R,M] images, as the
     shared run does (groups.rm_grey_stack, SPLIT_THEORY A2-7; w_rm is then 25).
  4. D_g: upstream's RaGAN discriminator loss on group g's B images (rm grey: 2B).
  5. every gradient (3 G + 3 D) checked finite BEFORE any update; then six Adams.

Everything else is deeptex_renloss_21's: grouped record, seed-keyed crop / flip /
rot stream (resume-exact), shuffle off, B materials per step, the paper's pixel /
feature / gan weights and Adam / MultiStepLR settings, the render block. Checks
and log lines reuse hooks_common / render_loss (same [metrics] / [render]
prefixes; [terms] carries one block per group).
"""

import argparse
import math
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *[os.pardir] * 3))
sys.path.insert(0, REPO)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # before tensorflow; see paper_loop/main.py

from variants.util import paths  # noqa: E402

paths.use_deeptex()

import tensorflow as tf  # noqa: E402
from modules.losses import (ContentLoss, DiscriminatorLoss, GeneratorLoss,  # noqa: E402
                            PixelLoss)
from modules.lr_scheduler import MultiStepLR  # noqa: E402
from modules.models import DiscriminatorVGG128, RRDB_Model  # noqa: E402
from modules.utils import load_yaml, set_memory_growth  # noqa: E402

from variants.deeptex.hooks_common import log, safety  # noqa: E402
from variants.deeptex.hooks_common import seed as seeding  # noqa: E402
from variants.deeptex.hooks_common.ckpt import weights_sha  # noqa: E402
from variants.deeptex.hooks_common.safety import die  # noqa: E402
from variants.deeptex.render_loss import grouped_dataset, render_tf  # noqa: E402
from variants.deeptex.render_loss.hooks import (check_grouped_dataset,  # noqa: E402
                                                nan_debug_exit, nan_reason)
from variants.deeptex.split import groups as grp  # noqa: E402

GROUPS = grp.GROUPS
RM_LOSS = ("packed", "grey")
MAX_TO_KEEP = 3          # upstream's; with save_steps 5000 -> 30k / 35k / 40k


def g_params_formula(nf, nb, gc=32):
    """RRDB parameter count (SPLIT_THEORY A2-1), 3x3 convs with bias, 3 channels in/out."""
    p_rdb = 9 * (nf * nf + 8 * nf * gc + 6 * gc * gc) + (4 * gc + nf)
    fixed = (9 * 3 * nf + nf) + (9 * nf * nf + nf) * 4 + (9 * nf * 3 + 3)
    return fixed + 3 * p_rdb * nb


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg_path", required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--max-steps", type=int, default=0, help="override niter (smoke runs)")
    ap.add_argument("--save-steps", type=int, default=0, help="override cfg save_steps")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--render-monitor-every", type=int, default=100,
                    help="'[render] step=N l21=... l21_a=... l21_n=... l21_rm=...' at step 1 and "
                         "every N steps: the joint render l2,1 and, per group, the l2,1 with "
                         "only that group's SR and every other slot GT (0 = off)")
    ap.add_argument("--tfrecord", default="", help="override train_dataset.path")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--checkpoint-root", default="")
    ap.add_argument("--log-root", default="")
    ap.add_argument("--allow-dead-lr-schedule", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--skip-verify-gate", action="store_true")
    ap.add_argument("--check-numerics", action="store_true")
    ap.add_argument("--count-only", action="store_true",
                    help="build the six networks, print their parameter counts, exit 0 "
                         "(no GPU or dataset needed)")
    return ap.parse_args(argv)


def resolve_group_weights(cfg):
    ws = cfg.get("w_render_groups")
    if not isinstance(ws, dict) or set(ws) != set(GROUPS):
        die(8, "w_render_groups must map exactly %s to numbers (SPLIT_THEORY A2-5), got %r"
            % (list(GROUPS), ws))
    out = {}
    for g in GROUPS:
        try:
            w = float(ws[g])               # PyYAML reads an untagged 1e-2 as a string
        except (TypeError, ValueError):
            die(8, "w_render_groups.%s = %r is not a number" % (g, ws[g]))
        if not math.isfinite(w) or w < 0:
            die(8, "w_render_groups.%s = %r must be finite and >= 0" % (g, w))
        out[g] = w
    return out


def load_config(a):
    cfg = load_yaml(a.cfg_path)
    if a.tfrecord:
        cfg["train_dataset"]["path"] = a.tfrecord
    if a.checkpoint_root:
        cfg["checkpoint_root"] = a.checkpoint_root
    if a.log_root:
        cfg["log_root"] = a.log_root
    if not bool((cfg.get("train_dataset") or {}).get("grouped", False)):
        die(8, "the split trainer needs a grouped config (train_dataset.grouped: True)")
    if cfg["ch_size"] != 3:
        die(8, "ch_size must be 3 (every group network is 3 channels in / out), got %r"
            % cfg["ch_size"])
    for k in ("checkpoint_root", "log_root"):
        if k not in cfg:
            die(8, "the split config must set %s (each arm has its own roots)" % k)
        safety.check_out_root(k, cfg[k])
    safety.check_gt_size(cfg)
    safety.check_lr_schedule(cfg, a.allow_dead_lr_schedule)   # against the config's own niter
    cfg["rm_loss"] = cfg.get("rm_loss", "packed")
    if cfg["rm_loss"] not in RM_LOSS:
        die(8, "rm_loss must be one of %s, got %r" % (RM_LOSS, cfg["rm_loss"]))
    cfg["niter"] = a.max_steps or cfg["niter"]
    cfg["save_steps"] = a.save_steps or cfg["save_steps"]
    return cfg


def build_networks(cfg):
    G = {g: RRDB_Model(cfg["input_size"], 3, cfg["network_G"], name="RRDB_model_" + g)
         for g in GROUPS}
    D = {g: DiscriminatorVGG128(cfg["gt_size"], 3, name="Discriminator_VGG_128_" + g)
         for g in GROUPS}
    return G, D


def report_params(cfg, G, D):
    nf, nb = cfg["network_G"]["nf"], cfg["network_G"]["nb"]
    counts = {g: int(G[g].count_params()) for g in GROUPS}
    total = sum(counts.values())
    shared = g_params_formula(nf, 23)
    expect = g_params_formula(nf, nb)
    print("[params] G nb=%d nf=%d: %s total=%d (shared nb=23 G: %d, ratio %.3f); formula per G %d %s"
          % (nb, nf, " ".join("%s=%d" % (g, counts[g]) for g in GROUPS), total, shared,
             total / float(shared), expect,
             "OK" if all(c == expect for c in counts.values()) else "MISMATCH"), flush=True)
    print("[params] D per group %d, total %d" % (D["a"].count_params(),
                                                 sum(D[g].count_params() for g in GROUPS)),
          flush=True)
    return counts


def main(argv=None):
    a = parse_args(sys.argv[1:] if argv is None else argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    cfg = load_config(a)
    w = resolve_group_weights(cfg)
    seed = a.seed if a.seed is not None else cfg.get("seed")
    if seed is None:
        die(8, "the split trainer needs a seed: it fixes the init and keys every crop draw")

    if a.count_only:
        seeding.apply_seed(seed)
        report_params(cfg, *build_networks(cfg))
        return 0

    paths.check_pin("DeepTextures")
    set_memory_growth()
    if a.check_numerics:
        tf.debugging.enable_check_numerics()
    try:
        render_cfg = render_tf.validate_render_cfg(cfg.get("render"), warn=False)
    except ValueError as e:
        die(8, str(e))
    seed = seeding.apply_seed(seed)
    safety.check_gpu(a.allow_cpu)
    check_grouped_dataset(cfg, require_verify=not a.skip_verify_gate)
    ckpt_dir = os.path.join(cfg["checkpoint_root"], cfg["sub_name"])
    log_dir = os.path.join(cfg["log_root"], cfg["sub_name"])
    os.makedirs(ckpt_dir, exist_ok=True)
    safety.check_disk(cfg["checkpoint_root"], 12.0)

    B, n_m = int(cfg["batch_size"]), grouped_dataset.N_MAPS
    s_lr, s_hr = int(cfg["input_size"]), int(cfg["gt_size"])
    niter, save_steps = int(cfg["niter"]), int(cfg["save_steps"])
    rm_grey = cfg["rm_loss"] == "grey"
    print("[*] split joint: %d materials/step, groups %s, w_render %s, seed=%s, niter=%d, rm_loss=%s"
          % (B, ",".join(GROUPS), " ".join("%s=%g" % (g, w[g]) for g in GROUPS), seed, niter,
             cfg["rm_loss"]))

    def loss_images(g, img):
        """What group g's pixel / VGG / D terms see: its own image, or for rm with
        rm_loss grey the GGG roughness and metallic images (2B of them)."""
        return grp.rm_grey_stack(img) if (g == "rm" and rm_grey) else img
    want_rm = 25.0 if rm_grey else 50.0 / 3.0
    if abs(w["rm"] - want_rm) > 1e-3 * want_rm:
        print("[!] w_render_groups.rm = %g, but rm_loss %s gives kappa_rm x 10 = %g (SPLIT_THEORY A2-5/7)"
              % (w["rm"], cfg["rm_loss"], want_rm), flush=True)

    # ---------------------------------------------------------------- networks
    G, D = build_networks(cfg)
    report_params(cfg, G, D)

    def adam(lr_key, b1, b2):
        sched = MultiStepLR(cfg[lr_key], cfg["lr_steps"], cfg["lr_rate"])
        return tf.keras.optimizers.Adam(learning_rate=sched, beta_1=cfg[b1], beta_2=cfg[b2])
    opt_G = {g: adam("lr_G", "adam_beta1_G", "adam_beta2_G") for g in GROUPS}
    opt_D = {g: adam("lr_D", "adam_beta1_D", "adam_beta2_D") for g in GROUPS}

    pixel_loss_fn = PixelLoss(criterion=cfg["pixel_criterion"])
    fea_loss_fn = ContentLoss(criterion=cfg["feature_criterion"])
    gen_loss_fn = GeneratorLoss(gan_type=cfg["gan_type"])
    dis_loss_fn = DiscriminatorLoss(gan_type=cfg["gan_type"])
    render_loss = render_tf.RenderLoss(render_cfg)
    w_pixel, w_feature, w_gan = (float(cfg["w_pixel"]), float(cfg["w_feature"]),
                                 float(cfg["w_gan"]))

    # -------------------------------------------------------------- checkpoint
    objs = {"step": tf.Variable(0, name="step")}
    for g in GROUPS:
        objs["G_" + g], objs["D_" + g] = G[g], D[g]
        objs["opt_G_" + g], objs["opt_D_" + g] = opt_G[g], opt_D[g]
    checkpoint = tf.train.Checkpoint(**objs)
    manager = tf.train.CheckpointManager(checkpoint, ckpt_dir, max_to_keep=MAX_TO_KEEP)
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint).assert_existing_objects_matched()
        print("[*] resumed from %s at step %d" % (manager.latest_checkpoint,
                                                   int(checkpoint.step.numpy())))
    else:
        print("[*] training from scratch")
    print("[init] step=%d %s" % (int(checkpoint.step.numpy()), " ".join(
        "g_%s=%s d_%s=%s" % (g, weights_sha(G[g]), g, weights_sha(D[g])) for g in GROUPS)),
        flush=True)

    spec = [tf.TensorSpec([B, n_m, s_lr, s_lr, 3], tf.float32),
            tf.TensorSpec([B, n_m, s_hr, s_hr, 3], tf.float32),
            tf.TensorSpec([B, s_hr, s_hr, 2], tf.float32)]

    # -------------------------------------------------------------------- step
    @tf.function(input_signature=spec)
    def train_step(lr5, hr5, pos):
        x = grp.split_stack(lr5)
        y = grp.split_stack(hr5)
        gt_disp, valid = render_loss.target(hr5, pos)         # outside the tape
        with tf.GradientTape(persistent=True) as tape:
            sr = {g: G[g](x[g], training=True) for g in GROUPS}
            sr5 = grp.compose(sr, hr5)
            l_ren = render_loss.loss(sr5, pos, gt_disp, valid)
            terms_G, own, terms_D, loss_D = {}, {}, {}, {}
            for g in GROUPS:
                y_g, sr_g = loss_images(g, y[g]), loss_images(g, sr[g])
                hr_out = D[g](y_g, training=True)
                sr_out = D[g](sr_g, training=True)
                t = {"reg": tf.reduce_sum(G[g].losses),
                     "pixel": w_pixel * pixel_loss_fn(y_g, sr_g),
                     "feature": w_feature * fea_loss_fn(y_g, sr_g),
                     "gan": w_gan * gen_loss_fn(hr_out, sr_out)}
                own[g] = tf.add_n(list(t.values()))
                t["render"] = w[g] * l_ren
                terms_G[g] = t
                terms_D[g] = {"reg": tf.reduce_sum(D[g].losses),
                              "gan": dis_loss_fn(hr_out, sr_out)}
                loss_D[g] = tf.add_n(list(terms_D[g].values()))

        # One render backward: dL_ren / d sr_g for all three outputs.
        d_sr = tape.gradient(l_ren, [sr[g] for g in GROUPS])
        grads_G, grads_D = {}, {}
        for g, dg in zip(GROUPS, d_sr):
            # grad_theta_g (own_g + w_g L_ren): own terms by their own backward, the
            # render part by pulling w_g * dL_ren/d sr_g back through G_g only.
            grads_G[g] = tape.gradient([own[g], sr[g]], G[g].trainable_variables,
                                       output_gradients=[tf.ones_like(own[g]), w[g] * dg])
            grads_D[g] = tape.gradient(loss_D[g], D[g].trainable_variables)
        del tape

        flat = [t for g in GROUPS for t in grads_G[g] + grads_D[g]]
        finite = tf.debugging.assert_all_finite(tf.linalg.global_norm(flat), "non-finite gradient")
        with tf.control_dependencies([finite]):
            for g in GROUPS:
                opt_G[g].apply_gradients(zip(grads_G[g], G[g].trainable_variables))
                opt_D[g].apply_gradients(zip(grads_D[g], D[g].trainable_variables))

        out = {"loss_G": tf.add_n([tf.add_n(list(terms_G[g].values())) for g in GROUPS]),
               "loss_D": tf.add_n([loss_D[g] for g in GROUPS]),
               "render_l21": l_ren}
        for g in GROUPS:
            for k, v in terms_G[g].items():
                out["%s.%s" % (g, k)] = v
            out["D%s.gan" % g] = terms_D[g]["gan"]
        diag = render_loss.diagnostics(tf.stop_gradient(sr5))
        diag.pop("oor_height", None)                          # GT slot: always 0
        diag["mask_frac"] = 1.0 - tf.reduce_mean(valid)
        for k, v in diag.items():
            out["diag/" + k] = v
        return out

    # Joint render l2,1 and, per group, only that group's SR with every other slot GT.
    # Measured BEFORE the step's update (as deeptex_renloss_21's [render] probe).
    @tf.function(input_signature=spec)
    def render_monitor(lr5, hr5, pos):
        x = grp.split_stack(lr5)
        gt_disp, valid = render_loss.target(hr5, pos)
        sr = {g: G[g](x[g], training=False) for g in GROUPS}   # RRDB has no BN / dropout
        res = {"l21": render_loss.loss(grp.compose(sr, hr5), pos, gt_disp, valid)}
        for g in GROUPS:
            res["l21_" + g] = render_loss.loss(grp.compose_one(sr[g], g, hr5), pos, gt_disp, valid)
        return res

    # -------------------------------------------------------------------- loop
    start = int(checkpoint.step.numpy())
    remain = max(niter - start, 0)
    try:
        ds = grouped_dataset.load_grouped_dataset(cfg, shuffle=False, seed=seed,
                                                  start_index=start * B)
    except (ValueError, TypeError) as e:
        die(8, "grouped dataset: %s" % e)
    writer = tf.summary.create_file_writer(log_dir)
    window = log.MetricsWindow()
    every = a.render_monitor_every
    print("[*] steps %d -> %d  ckpt=%s  logs=%s" % (start, niter, ckpt_dir, log_dir), flush=True)

    saw_step = False
    for lr5, hr5, pos in ds.take(remain):
        checkpoint.step.assign_add(1)
        steps = int(checkpoint.step.numpy())
        saw_step = True
        batch = (lr5, hr5, pos)
        if every and (steps == 1 or steps % every == 0 or steps == niter):
            m = {k: float(v.numpy()) for k, v in render_monitor(*batch).items()}
            print("[render] step=%d l21=%.6f %s sum_groups=%.6f" % (
                steps, m["l21"], " ".join("l21_%s=%.6f" % (g, m["l21_" + g]) for g in GROUPS),
                sum(m["l21_" + g] for g in GROUPS)), flush=True)
            with writer.as_default():
                for k, v in m.items():
                    tf.summary.scalar("monitor/render_" + k, v, step=steps)
        try:
            out = train_step(*batch)
            g_tot, d_tot = float(out["loss_G"].numpy()), float(out["loss_D"].numpy())
        except tf.errors.InvalidArgumentError as e:
            text = str(e)
            if not any(s in text for s in ("non-finite", "NaN", "Inf")):
                raise
            nan_debug_exit(checkpoint, cfg["checkpoint_root"], cfg["sub_name"], steps, batch,
                           nan_reason(text))
        if not (math.isfinite(g_tot) and math.isfinite(d_tot)):
            nan_debug_exit(checkpoint, cfg["checkpoint_root"], cfg["sub_name"], steps, batch,
                           "loss_G=%s loss_D=%s" % (g_tot, d_tot),
                           [(k, v) for k, v in out.items() if not k.startswith("diag/")])

        window.tick()
        if steps % a.log_every == 0 or steps == niter:
            window.line(steps, niter, g_tot, d_tot, opt_G["a"])
        if steps == 1 or steps % a.log_every == 0 or steps == niter:
            parts = ["render_l21=%.5g" % float(out["render_l21"].numpy())]
            for g in GROUPS:
                parts.append("| %s " % g + " ".join(
                    "%s=%.5g" % (k, float(out["%s.%s" % (g, k)].numpy()))
                    for k in ("pixel", "feature", "gan", "render")) +
                    " D=%.5g" % float(out["D%s.gan" % g].numpy()))
            parts.append("|" + "".join(" %s=%.5g" % (k[5:], float(v.numpy()))
                                       for k, v in out.items() if k.startswith("diag/")))
            print("[terms] step=%d %s" % (steps, " ".join(parts)), flush=True)
        if steps % 10 == 0:
            with writer.as_default():
                tf.summary.scalar("loss_G/total_loss", g_tot, step=steps)
                tf.summary.scalar("loss_D/total_loss", d_tot, step=steps)
                tf.summary.scalar("learning_rate_G", opt_G["a"].lr(steps), step=steps)
                for k, v in out.items():
                    if k not in ("loss_G", "loss_D"):
                        tf.summary.scalar(k if k.startswith("diag/") else "terms/" + k, v,
                                          step=steps)
        if steps % save_steps == 0:
            manager.save()
            print("[*] saved %s" % manager.latest_checkpoint, flush=True)

    final = int(checkpoint.step.numpy())
    if saw_step and final % save_steps != 0:
        manager.save()
        print("[*] saved %s" % manager.latest_checkpoint)
    if not saw_step and remain > 0:
        die(5, "the dataset yielded no batches - the TFRecord is empty or unreadable")
    print("[*] training done at step %d (checkpoint %s)" % (final, manager.latest_checkpoint))
    return 0


if __name__ == "__main__":
    sys.exit(main())
