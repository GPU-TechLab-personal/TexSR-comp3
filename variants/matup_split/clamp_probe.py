"""Per-channel clamp statistics of one MatUp run, before training and after every epoch
(docs/CLAMP_DEATH.md section 5). run.py --probe installs it; probe_report.py reads it.

MatUp's output is y = clamp(s + f(x)) (data_utils.get_renorm_clamped_maps): albedo and
metallic in [0, 1], alpha in [Renderer.min_alpha, 1]. A channel whose pre-clamp value
z = s + f(x) is outside that range at EVERY texel gets no gradient at all ("dead"). This
records, for every clamped channel, on the whole material (the texels training samples):

  active        fraction of texels with lo <= z <= hi (0 = dead)
  below, above  fractions clamped at the lower / upper bound
  s_mean        mean of the bilinear initialisation s (the margin to the lower bound)
  d_mean, d_std mean and spatial std of the MLP output f(x) (mu_c, sigma_c)
  z_min, z_max
  row_step      L2 change of this channel's last-layer row (weights + bias) since the
                previous probe (0 = frozen)

and per sub-network (split: a / n / rm; shared: all) the L2 change of all its parameters
since the previous probe (param_step; 0 = the whole group is frozen).

Tags: 0 = before the first step (initialisation), 1..E = after epoch e (MatUp's own
testing.eval_model, called each epoch through --save_interval 1).
"""

import json
import os

import torch

# (name, channel in MatUp's 7-vector, lower bound or None = Renderer.min_alpha, upper bound)
CHANNELS = (("R", 0, 0.0, 1.0), ("G", 1, 0.0, 1.0), ("B", 2, 0.0, 1.0),
            ("metallic", 3, 0.0, 1.0), ("alpha", 6, None, 1.0))
# split_mlp.GROUP_CHANNELS: which sub-network and which row of its last layer owns a channel
SPLIT_ROW = {0: ("a", 0), 1: ("a", 1), 2: ("a", 2), 3: ("rm", 0), 6: ("rm", 1)}


def last_rows(model):
    """channel -> its last-layer row (weights and bias, flattened), shared or split MLP."""
    out = {}
    nets = getattr(model, "nets", None)
    for name, c, _, _ in CHANNELS:
        if nets is not None:
            g, r = SPLIT_ROW[c]
            last = nets[g].mlp[-1]
        else:
            last, r = model.mlp[-1], c
        out[name] = torch.cat([last.weight[r].reshape(-1), last.bias[r:r + 1]]).detach().clone()
    return out


def groups(model):
    nets = getattr(model, "nets", None)
    if nets is not None:
        return {g: torch.cat([p.detach().reshape(-1) for p in n.parameters()]).clone()
                for g, n in nets.items()}
    return {"all": torch.cat([p.detach().reshape(-1) for p in model.parameters()]).clone()}


class Probe(object):

    def __init__(self, path, min_alpha, tile_width):
        self.path, self.min_alpha, self.pad = path, float(min_alpha), (tile_width - 1) // 2
        self.rows, self.prev_rows, self.prev_groups = [], None, None

    @torch.no_grad()
    def __call__(self, tag, model, dataset):
        device = next(model.parameters()).device
        was_training = model.training
        model.eval()
        x = dataset.get_svbrdf_init().clone().to(device)               # (H+2p, W+2p, 7)
        d = model(x.permute(2, 0, 1)[None])[0].permute(1, 2, 0)        # (H, W, 7), as testing.eval_model
        model.train(was_training)
        p = self.pad
        s = x[p:-p, p:-p, :]
        z = s + d
        row = {"tag": tag, "channels": {}}
        rows_now, groups_now = last_rows(model), groups(model)
        for name, c, lo, hi in CHANNELS:
            lo = self.min_alpha if lo is None else lo
            zc = z[..., c]
            st = {"active": ((zc >= lo) & (zc <= hi)).float().mean().item(),
                  "below": (zc < lo).float().mean().item(),
                  "above": (zc > hi).float().mean().item(),
                  "s_mean": s[..., c].mean().item(),
                  "d_mean": d[..., c].mean().item(),
                  "d_std": d[..., c].std().item(),
                  "z_min": zc.min().item(), "z_max": zc.max().item()}
            if self.prev_rows is not None:
                st["row_step"] = (rows_now[name] - self.prev_rows[name]).norm().item()
            row["channels"][name] = st
        if self.prev_groups is not None:
            row["param_step"] = {g: (v - self.prev_groups[g]).norm().item()
                                 for g, v in groups_now.items()}
        self.prev_rows, self.prev_groups = rows_now, groups_now
        self.rows.append(row)
        self.write()
        dead = [n for n, st in row["channels"].items() if st["active"] == 0.0]
        print("[probe] %s: dead %s" % (tag, ",".join(dead) or "-"), flush=True)

    def write(self, extra=None):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as fh:
            json.dump(dict({"probes": self.rows}, **(extra or {})), fh, indent=1)


def install(probe, nb_epochs):
    """Hook MatUp's modules: probe at init (train.train_model) and after every epoch
    (testing.eval_model with --save_interval 1); the per-epoch map PNGs are not written."""
    import matup.data_utils as du
    import matup.testing as testing
    import matup.train as train

    orig_train, orig_eval, orig_out = train.train_model, testing.eval_model, du.output_maps_from_list
    calls = {"n": 0}

    def train_model(args, model, optimizer, dataset, *rest):
        probe(0, model, dataset)
        return orig_train(args, model, optimizer, dataset, *rest)

    def eval_model(args, model, dataset, device):
        calls["n"] += 1
        if calls["n"] <= nb_epochs:                 # the call after training repeats epoch E
            probe(calls["n"], model, dataset)
        return orig_eval(args, model, dataset, device)

    def output_maps_from_list(maps, args, suffix="", *a, **k):
        if str(suffix).isdigit():                   # train.py's per-epoch maps (save_interval)
            return None
        return orig_out(maps, args, suffix, *a, **k)

    train.train_model = train_model
    testing.eval_model = eval_model
    du.output_maps_from_list = output_maps_from_list
