import torch
from torch import nn
import spconv.pytorch as spconv


def share_indices(x: spconv.SparseConvTensor, y: spconv.SparseConvTensor
                  ) -> tuple[spconv.SparseConvTensor, spconv.SparseConvTensor]:
    x_z = spconv.SparseConvTensor(torch.zeros_like(x.features), x.indices,
                                  x.spatial_shape, x.batch_size)
    y_z = spconv.SparseConvTensor(torch.zeros_like(y.features),
                                  y.indices, y.spatial_shape, y.batch_size)
    x = spconv.functional.sparse_add(x, y_z)
    y = spconv.functional.sparse_add(y, x_z)
    return x, y


def loss_fn(x, y, weights):
    x, y = share_indices(x, y)
    pos_weight = weights * (y.features.sum(dim=0) > 0)
    return nn.functional.binary_cross_entropy_with_logits(
        x.features, y.features, pos_weight=pos_weight)


def make_block(name, input_size, output_size, stride, dilation, algo=None):
    modules = [
        spconv.SparseConv2d(
            input_size, 8, 3, stride=stride, dilation=dilation,
            indice_key=f"{name}1", algo=algo),
        nn.ReLU(),
        spconv.SparseConv2d(
            8, 16, 3, stride=stride, dilation=dilation, indice_key=f"{name}2",
            algo=algo),
        nn.ReLU(),
        spconv.SparseInverseConv2d(
            16, 8, 3, indice_key=f"{name}2", algo=algo),
        nn.ReLU(),
        spconv.SparseInverseConv2d(
            8, output_size, 3, indice_key=f"{name}1", algo=algo),
        nn.ReLU(),
    ]
    return spconv.SparseSequential(*modules)


class SparseFCN(nn.Module):

    def __init__(self, input_features: int, num_classes: int,
                 stride: int, dilation: int):
        super().__init__()
        self.num_classes = num_classes
        self.blocks = [
            (True, make_block("b1", input_features, num_classes, 4, 4)),
            (True, make_block("b2", num_classes + input_features, num_classes, 3, 3)),
            (False, make_block("b3", num_classes, num_classes, 2, 2)),
            (False, make_block("b4", num_classes, num_classes, 1, 1)),
        ]
        for i, block in enumerate(self.blocks):
            setattr(self, f"b{i}", block)
        self.last = spconv.SparseConv2d(
            num_classes * len(self.blocks), num_classes, 1)

    def forward(self, input: spconv.SparseConvTensor):
        outs = None
        x = None
        for add_input, block in self.blocks:
            if x is None:
                x = input
            elif add_input:
                x = x.replace_feature(
                    torch.cat([x.features, input.features], dim=1))
            x = block(x)
            if outs is None:
                outs = x
            else:
                outs = outs.replace_feature(
                    torch.cat([outs.features, x.features], dim=1))
        return self.last(outs)
