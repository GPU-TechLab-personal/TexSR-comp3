"""MatUp's per-material MLP split into three group MLPs (docs/SPLIT_THEORY.md B1).

MatUp builds ONE FullyConnected(num_in=7, num_out=7, tile_width=17, ...)
(main_optim.py:59-66): a 17x17 conv over the 7-channel bilinear init
[albedo 0-2, metallic 3, normal_xy 4-5, alpha 6] (data_utils.maps_from_data),
then 1x1 convs, giving a 7-channel residual for the tile centre.

SplitMLP has the same constructor and the same (B, 7, h, w) -> (B, 7, h', w')
contract, so MatUp's training loop, renderer, metallic regulariser and CUDA
graphs run unchanged. Inside are three FullyConnected of MatUp's own class,
each seeing ONLY its group's channels and writing only those channels:

  group  input channels        output channels
  a      0, 1, 2  (albedo)     0, 1, 2
  n      4, 5     (normal xy)  4, 5
  rm     3, 6     (metallic, alpha = roughness^2)   3, 6

Each sub-MLP keeps MatUp's width (hid_layer_width 128), depth (hid_layer_size
4), tile width and non-linearity; only num_in / num_out change. Parameters:
shared 309,511; split 161,027 + 123,906 + 123,906 = 408,839 (x1.32: most
parameters sit in the 17x17 first layer, whose input channels the groups divide).
"""

import torch

from matup.model import FullyConnected

GROUP_CHANNELS = (("a", (0, 1, 2)), ("n", (4, 5)), ("rm", (3, 6)))


def _inverse(order):
    inv = [0] * len(order)
    for pos, c in enumerate(order):
        inv[c] = pos
    return inv


class SplitMLP(torch.nn.Module):

    def __init__(self, num_in, num_out, tile_width, nb_hidden_layers, hidden_layer_size, non_lin):
        super(SplitMLP, self).__init__()
        if num_in != 7 or num_out != 7:
            raise ValueError("SplitMLP replaces MatUp's 7 -> 7 MLP, got %d -> %d" % (num_in, num_out))
        self.nets = torch.nn.ModuleDict()
        order = []
        for g, ch in GROUP_CHANNELS:
            self.nets[g] = FullyConnected(len(ch), len(ch), tile_width, nb_hidden_layers,
                                          hidden_layer_size, non_lin)
            self.register_buffer("idx_" + g, torch.tensor(ch, dtype=torch.long), persistent=False)
            order.extend(ch)
        # concat order (a, n, rm) = channels [0,1,2,4,5,3,6]; inv puts them back at 0..6
        self.register_buffer("inv", torch.tensor(_inverse(order), dtype=torch.long),
                             persistent=False)

    def forward(self, x):
        outs = [self.nets[g](torch.index_select(x, 1, getattr(self, "idx_" + g)))
                for g, _ in GROUP_CHANNELS]
        return torch.index_select(torch.cat(outs, dim=1), 1, self.inv)


def count_params(model):
    return sum(p.numel() for p in model.parameters())
