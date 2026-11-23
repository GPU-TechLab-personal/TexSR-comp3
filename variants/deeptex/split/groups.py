"""The three map groups of the split generators, and how a 5-map stack maps onto them.

Pure tensor code (TensorFlow, no DeepTextures import), shared by the trainer
(split/train.py), inference (split/infer_split.py) and the unit test
(split/test_groups.py). docs/SPLIT_THEORY.md A2 is the maths.

A grouped batch is a (B, 5, H, W, 3) stack in grouped_dataset.MAP_KEYS order:
baseColor, normal, roughness, metallic, height. Roughness / metallic / height
are grey maps stored as three equal channels (GGG).

  group  network input / target (3 channels)    decode back to the stack
  a      baseColor                               slot 0 as is
  n      normal                                  slot 1 as is
  rm     [0, mean_c(roughness), mean_c(metallic)] slot 2 = tile(ch1), slot 3 = tile(ch2)

Height has no network. compose() puts the GT height in slot 4, so the render
term is computed on (A^, N^, R^, M^, H_GT).

Why tile on the way back: render_tf.decode_maps reads a grey slot as the mean
of its three channels. mean_c(tile(x, 3)) == x exactly, so the renderer is
untouched, and the gradient that reaches ch1 is dL/dR itself (three channels x
1/3 each). That is the factor kappa_rm = 5/3 accounts for (SPLIT_THEORY A2-5).
"""

import tensorflow as tf

GROUPS = ("a", "n", "rm")
# group -> the stack slots its output fills (render_tf.MAP_SLOTS order)
SLOTS = {"a": (0,), "n": (1,), "rm": (2, 3)}
SLOT_NAMES = ("baseColor", "normal", "roughness", "metallic", "height")
RM_CONST = 0.0       # channel 0 of the packed RM image


def _grey(x):
    """(B, H, W, 3) grey map -> (B, H, W, 1) by channel mean (== channel 0 on GGG)."""
    return tf.reduce_mean(x, axis=-1, keepdims=True)


def pack_rm(rough, metal):
    """(B, H, W, 3) GGG roughness and metallic -> (B, H, W, 3) [0, R, M]."""
    r = _grey(rough)
    return tf.concat([tf.zeros_like(r) + RM_CONST, r, _grey(metal)], axis=-1)


def split_stack(x5):
    """(B, 5, H, W, 3) stack -> {group: (B, H, W, 3)} network inputs or targets."""
    return {"a": x5[:, 0], "n": x5[:, 1], "rm": pack_rm(x5[:, 2], x5[:, 3])}


def unpack_rm(y_rm):
    """(B, H, W, 3) RM output -> (roughness, metallic), each (B, H, W, 3) GGG."""
    r, m = y_rm[..., 1:2], y_rm[..., 2:3]
    return tf.tile(r, [1, 1, 1, 3]), tf.tile(m, [1, 1, 1, 3])


def compose(sr, gt5):
    """{group: SR output} + GT stack -> the (B, 5, H, W, 3) stack the renderer reads.

    Every slot a network owns comes from its SR output; height comes from gt5."""
    rough, metal = unpack_rm(sr["rm"])
    return tf.stack([sr["a"], sr["n"], rough, metal, gt5[:, 4]], axis=1)


def compose_one(sr_g, g, gt5):
    """The diagnostic stack: group g's slots from its SR output, every other slot GT.

    SPLIT_THEORY C-1: L_ren of this, per group, against L_ren of compose() shows
    how much the joint render error is below the sum of the per-group errors."""
    slots = [gt5[:, i] for i in range(5)]
    if g == "a":
        slots[0] = sr_g
    elif g == "n":
        slots[1] = sr_g
    elif g == "rm":
        slots[2], slots[3] = unpack_rm(sr_g)
    else:
        raise ValueError("unknown group %r" % (g,))
    return tf.stack(slots, axis=1)
