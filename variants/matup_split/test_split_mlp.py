#!/usr/bin/env python3
"""CPU checks of SplitMLP (no GPU, no data). Needs torch only.

    python variants/matup_split/test_split_mlp.py          (repo root; any torch)

1. parameter counts: shared 309,511, split 161,027 + 123,906 + 123,906 = 408,839
2. shape contract: (B, 7, 17, 17) -> (B, 7, 1, 1), and the full-image eval
   path (1, 7, H, W) -> (1, 7, H-16, W-16) as testing.py uses it
3. channel isolation: perturbing one group's input channels changes only that
   group's output channels (the split MLP never sees another group's maps)
4. output placement: each sub-MLP's output lands on its own channels
5. width variants: split_params == the built count for x1 (309,719) and x3 (927,302),
   and with_width keeps the 7 -> 7 contract
"""

import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *[os.pardir] * 2))
sys.path.insert(0, os.path.join(REPO, "external", "matup_egsr"))
sys.path.insert(0, REPO)

import torch  # noqa: E402

from matup.model import FullyConnected  # noqa: E402
from variants.matup_split.split_mlp import (GROUP_CHANNELS, WIDTHS, SplitMLP,  # noqa: E402
                                            count_params, split_params, with_width)

KW = dict(num_in=7, num_out=7, tile_width=17, nb_hidden_layers=4, hidden_layer_size=128,
          non_lin="LeakyReLU")


def main():
    torch.manual_seed(0)
    shared, split = FullyConnected(**KW), SplitMLP(**KW)
    per = {g: count_params(split.nets[g]) for g, _ in GROUP_CHANNELS}
    print("params shared=%d split=%s total=%d" % (count_params(shared), per, count_params(split)))
    assert count_params(shared) == 309511
    assert per == {"a": 161027, "n": 123906, "rm": 123906}
    assert count_params(split) == 408839

    x = torch.randn(4, 7, 17, 17)
    assert split(x).shape == (4, 7, 1, 1)
    assert split(torch.randn(1, 7, 40, 48)).shape == (1, 7, 24, 32)

    y0 = split(x)
    for g, ch in GROUP_CHANNELS:
        x2 = x.clone()
        x2[:, list(ch)] += torch.randn(4, len(ch), 17, 17)
        changed = (split(x2) - y0).abs().amax(dim=(0, 2, 3)) > 0
        assert changed.tolist() == [c in ch for c in range(7)], (g, changed.tolist())

    for g, ch in GROUP_CHANNELS:
        sub = split.nets[g](x[:, list(ch)])
        assert torch.allclose(y0[:, list(ch)], sub), g

    assert split_params(128) == 408839
    want = {"x1": 309719, "x3": 927302}
    for key, w in WIDTHS.items():
        m = with_width(w)(**KW)
        n = count_params(m)
        print("split %s width=%d params=%d (%.3fx shared)" % (key, w, n, n / 309511.0))
        assert n == split_params(w) == want[key], (key, n)
        assert m(x).shape == (4, 7, 1, 1)
    print("OK")


if __name__ == "__main__":
    main()
