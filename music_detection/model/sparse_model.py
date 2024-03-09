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


def loss_fn(x, y):
    x, y = share_indices(x, y)
    return nn.functional.binary_cross_entropy_with_logits(
        x.features, y.features)


class SparseFCN(nn.Module):

    def __init__(self, input_features: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        algo = spconv.ConvAlgo.Native
        self.net = spconv.SparseSequential(
            spconv.SparseConv2d(
                input_features, 8, 3, stride=2, dilation=2, indice_key="cp1",
                algo=algo),
            nn.ReLU(),
            spconv.SparseConv2d(
                8, 16, 3, stride=2, dilation=2, indice_key="cp2", algo=algo),
            nn.ReLU(),
            spconv.SparseInverseConv2d(
                16, 8, 3, indice_key="cp2", algo=algo),
            nn.ReLU(),
            spconv.SparseInverseConv2d(
                8, num_classes, 3, indice_key="cp1", algo=algo),
        )

    def forward(self, x):
        return self.net(x)
