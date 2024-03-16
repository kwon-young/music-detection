import yaml
import json

import torch
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

from music_detection.data.sparse_dataset import SparseCoco
from music_detection.params import Params


def main():
    with open("params.yaml") as f:
        params = Params(**yaml.load(f, Loader=yaml.Loader))

    train_params = params.train_sparse

    dataset = SparseCoco(
        train_params.dataset / 'train2017',
        train_params.dataset / 'annotations/music_keypoints_train2017.json')

    weights = torch.zeros(dataset.num_classes, dtype=torch.float32)
    size = 0
    for _, target in tqdm(dataset):
        values = target.coalesce().values().to(torch.uint8)
        weights += values.sum(dim=0)
        size += values.size(0)
    weights = (size - weights) / weights
    name = f"data/{train_params.dataset.stem}_weights.json"
    label_weights = {
        dataset.cat[i]: weight for i, weight in enumerate(weights.tolist())
    }
    with open(name, 'w') as f:
        json.dump(label_weights, f, indent=4)


if __name__ == "__main__":
    main()
