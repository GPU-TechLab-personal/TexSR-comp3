#!/usr/bin/env python3
"""Checks of the split trainer's maths, in the deeptex container (CPU is enough, no data).

    cd docker && docker compose run --rm -T deeptex python /workspace/variants/deeptex/split/test_split.py

1. groups: split_stack -> compose reproduces a GT stack exactly (GGG grey maps),
   and compose_one puts SR in exactly the group's slots.
2. parameter counts: RRDB_Model.count_params == g_params_formula for nb 1, 7, 23
   (16,697,987 for nb=23, 5,187,203 for nb=7; SPLIT_THEORY A2-1).
3. the gradient each generator takes in train.py - one render backward, then
   tape.gradient([own_g, sr_g], theta_g, [1, w_g * dL_ren/d sr_g]) - equals the
   direct gradient of (own_g + w_g * L_ren) with respect to theta_g.
4. joint training: theta_n receives render gradient even though only sr_a's
   loss terms are switched on (L_ren couples the groups through the render),
   and w_g = 0 removes exactly the render part.
"""

import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *[os.pardir] * 3))
sys.path.insert(0, REPO)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from variants.util import paths  # noqa: E402

paths.use_deeptex()

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from modules.models import RRDB_Model  # noqa: E402

from variants.deeptex.render_loss import render_tf  # noqa: E402
from variants.deeptex.split import groups as grp  # noqa: E402
from variants.deeptex.split.train import g_params_formula  # noqa: E402

B, S_LR, S_HR = 2, 8, 32


def rand_stack(rng, side):
    """A (B, 5, side, side, 3) stack with GGG grey slots, values in (0.05, 0.95)."""
    x = rng.uniform(0.05, 0.95, size=(B, 5, side, side, 3)).astype(np.float32)
    for s in (2, 3, 4):
        x[:, s] = x[:, s, ..., :1]
    return tf.constant(x)


def check_groups(rng):
    x5 = rand_stack(rng, S_HR)
    y = grp.split_stack(x5)
    assert y["rm"].shape[-1] == 3 and float(tf.reduce_max(tf.abs(y["rm"][..., 0]))) == 0.0
    back = grp.compose(y, x5)
    np.testing.assert_allclose(back.numpy(), x5.numpy(), atol=1e-7)
    sr = {g: y[g] + 0.01 for g in grp.GROUPS}
    for g, slots in grp.SLOTS.items():
        one = grp.compose_one(sr[g], g, x5).numpy()
        for s in range(5):
            same = np.allclose(one[:, s], x5.numpy()[:, s])
            assert same == (s not in slots), (g, s)
    print("[ok] groups: split/compose round trip, compose_one slots")


def check_params():
    for nb in (1, 7, 23):
        m = RRDB_Model(None, 3, {"nf": 64, "nb": nb})
        assert m.count_params() == g_params_formula(64, nb), (nb, m.count_params())
    assert g_params_formula(64, 23) == 16697987 and g_params_formula(64, 7) == 5187203
    print("[ok] params: formula == count_params for nb 1, 7, 23")


def check_gradients(rng):
    render_loss = render_tf.RenderLoss(None)
    G = {g: RRDB_Model(S_LR, 3, {"nf": 8, "nb": 1}, name="G_" + g) for g in grp.GROUPS}
    lr5, hr5 = rand_stack(rng, S_LR), rand_stack(rng, S_HR)
    pos = tf.constant(rng.uniform(-1, 1, size=(B, S_HR, S_HR, 2)).astype(np.float32))
    x, y = grp.split_stack(lr5), grp.split_stack(hr5)
    gt_disp, valid = render_loss.target(hr5, pos)
    w = {"a": 50.0, "n": 50.0, "rm": 16.6666667}

    with tf.GradientTape(persistent=True) as tape:
        sr = {g: G[g](x[g], training=True) for g in grp.GROUPS}
        l_ren = render_loss.loss(grp.compose(sr, hr5), pos, gt_disp, valid)
        own = {g: tf.reduce_mean(tf.abs(sr[g] - y[g])) for g in grp.GROUPS}
        direct = {g: own[g] + w[g] * l_ren for g in grp.GROUPS}
    d_sr = tape.gradient(l_ren, [sr[g] for g in grp.GROUPS])
    for g, dg in zip(grp.GROUPS, d_sr):
        v = G[g].trainable_variables
        ours = tape.gradient([own[g], sr[g]], v, output_gradients=[tf.ones_like(own[g]), w[g] * dg])
        ref = tape.gradient(direct[g], v)
        for a, b in zip(ours, ref):
            np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=2e-4, atol=1e-7)
        ren_only = tape.gradient(sr[g], v, output_gradients=w[g] * dg)
        own_only = tape.gradient(own[g], v)
        n_ren = float(tf.linalg.global_norm(ren_only))
        assert n_ren > 0, "%s gets no render gradient" % g
        for a, b in zip(tape.gradient([own[g], sr[g]], v,
                                      output_gradients=[tf.ones_like(own[g]), 0.0 * dg]), own_only):
            np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-6, atol=1e-8)
        print("[ok] grad %s: split form == direct grad of own + w*L_ren; |render part| = %.3g"
              % (g, n_ren))
    del tape


def main():
    tf.config.set_visible_devices([], "GPU")
    rng = np.random.RandomState(0)
    tf.keras.utils.set_random_seed(0)
    check_groups(rng)
    check_params()
    check_gradients(rng)
    print("OK")


if __name__ == "__main__":
    main()
