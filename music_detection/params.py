from pathlib import Path
from dataclasses import dataclass


@dataclass
class Train:
    model: str
    batch_size: int
    epochs: int
    lr: float
    weights_backbone: str


@dataclass
class TrainSparse:
    output_dir: Path
    seed: int
    dataset: Path
    batch_size: int
    lr: float
    epochs: int
    weights: Path | None
    stride: int
    dilation: int

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.dataset = Path(self.dataset)


@dataclass
class Params:
    train: Train
    train_sparse: TrainSparse

    def __post_init__(self):
        self.train = Train(**self.train)
        self.train_sparse = TrainSparse(**self.train_sparse)
