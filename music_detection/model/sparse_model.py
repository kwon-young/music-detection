from torch import nn
import spconv.pytorch as spconv


class SparseFCN(nn.Module):

    def __init__(self):
        self.net = spconv.SparseSequential(
            spconv.SparseConv2d(
                4, 8, 3, stride=2, dilation=2, indice_key="cp1",
                algo=spconv.ConvAlgo.Native),
            nn.ReLU(),
            spconv.SparseConv2d(
                8, 16, 3, stride=2, dilation=2, indice_key="cp2",
                algo=spconv.ConvAlgo.Native),
            nn.ReLU(),
            spconv.SparseInverseConv2d(
                16, 8, 3, indice_key="cp2", algo=spconv.ConvAlgo.Native),
            nn.ReLU(),
            spconv.SparseInverseConv2d(
                8, 2, 3, indice_key="cp1", algo=spconv.ConvAlgo.Native),
        )
