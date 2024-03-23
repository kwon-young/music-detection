import yaml
import json

import torch
from tqdm import tqdm

from music_detection.data.sparse_dataset import SparseCoco
from music_detection.params import Params


def main():
    with open("params.yaml") as f:
        params = Params(**yaml.load(f, Loader=yaml.Loader))

    train_params = params.train_sparse

    dataset = SparseCoco(
        train_params.dataset / 'train2017',
        train_params.dataset / 'annotations/music_keypoints_train2017.json',
        label_whitelist=train_params.label_whitelist,
        image_whitelist=train_params.image_whitelist)

    weights = torch.zeros(dataset.num_classes, dtype=torch.float32)
    sizes = torch.zeros(dataset.num_classes, dtype=torch.int)
    for img, target in tqdm(dataset):
        target = target.coalesce()
        zero_values = torch.zeros(
            [img.values().size(0), target.values().size(1)],
            dtype=target.dtype)
        target = torch.sparse_coo_tensor(
            torch.cat([target.indices(), img.indices()], dim=-1),
            torch.cat([target.values(), zero_values], dim=0),
            target.shape)
        values = target.coalesce().values().to(torch.uint8)
        summed_values = values.sum(dim=0)
        weights += summed_values
        sizes += values.size(0) * (summed_values > 0)
    final_weights = (sizes.sum() - weights.sum()) / (
        dataset.num_classes * weights)
    name = f"data/{train_params.dataset.stem}_weights.json"
    final_weights = [weight if weight < 25 else 25
                     for weight in weights.tolist()]
    label_weights = {
        dataset.categories[i]['name']: weight
        for i, weight in enumerate(final_weights)
    }

    with open(name, 'w') as f:
        json.dump(label_weights, f, indent=4)


if __name__ == "__main__":
    main()
