"""MatUp's map preparation, without the rendering CLI.

Port of comp1 variants/matup/baseline/matup_render.py (load_set, position3d,
crop, DISP_VAL, EVAL_LIGHT), unchanged. Runs in the matup container; the caller
must have put external/matup_egsr on sys.path (paths.use_matup()) first.

Two things are easy to get wrong and both fail silently (matup_render.py):

* Normal convention. MatSynth maps (LR and GT) and every DeepTextures arm folder
  are OpenGL and are loaded with opengl_normals=True; MatUp's own normal_opt.png
  is already in the internal DirectX convention and must not be flipped again.
* The position grid resolution. MatUp builds the XY grid at the FULL output size
  and crops afterwards, so a 496^2 render comes from a 512^2 grid cropped [8:-8].
"""
import os

import numpy as np
import torch

from matup import data_utils, dataLoader, file_utils  # noqa: E402
from matup.renderer import Renderer  # noqa: E402

# dataLoader.py:154 hardcodes this in the dataset constructor and never reads
# args.disp_val, so the value MatUp actually optimises against is this one.
DISP_VAL = 0.01

# data_utils.py:110-112
EVAL_LIGHT = np.array([[0.276172, 0.276172, 0.920575]])

MIN_ROUGHNESS = 0.045            # dataLoader.py:39-40


def load_set(path, suffix, opengl_normals, floor_roughness):
    """PNG folder -> the four material tensors the renderer wants.

    `suffix` is "" for an SVBRDF/GT/arm folder and "_opt" for a MatUp run folder.
    """
    def rd(name, multi):
        return file_utils.read_img_as_float(
            os.path.join(path, name + suffix + ".png"), multi)

    albedo = file_utils.srgb2linrgb(torch.from_numpy(rd("baseColor", True))).float()
    metallic = torch.from_numpy(rd("metallic", False)).float()[..., None]
    normal = dataLoader.unpack_normal(
        torch.from_numpy(rd("normal", True)), opengl_normals).float()

    rough = torch.from_numpy(rd("roughness", False)).float()
    if floor_roughness:
        # load_roughness (dataLoader.py:34-41) floors the raw map. roughness_opt
        # stores sqrt(alpha) rather than a raw roughness, so it must not be
        # floored - alpha = value^2 round-trips it exactly either way.
        rough = torch.clamp(rough, min=MIN_ROUGHNESS)
    alpha = torch.clamp(torch.pow(rough, 2.0),
                        min=Renderer.min_alpha, max=1.0).float()[..., None]

    return albedo, metallic, normal, alpha


def position3d(height_norm, grid_size):
    """XY pixel-centre grid over [-1,1] plus the displaced height as Z."""
    pos2d = data_utils.compute_2D_position_grid(grid_size).float()
    z = (DISP_VAL * (height_norm - 0.5)).float()[..., None]
    return torch.cat([pos2d, z], dim=-1).float()


def crop(t, b):
    return t[b:-b, b:-b] if b else t
