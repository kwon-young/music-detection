import os
from typing import Optional, Callable
import json
from collections import defaultdict

import torch
import torch.utils.data as data
from torchvision.transforms.v2 import ToDtype
from torchvision.io import read_image, ImageReadMode
from spconv import SparseConvTensor

from music_detection.train.coco_utils import convert_coco_poly_to_mask


def invert(image: torch.Tensor):
    return 255 - image


def channel_last(image: torch.Tensor):
    return image.permute(1, 2, 0)


def from_dense(x: torch.Tensor, device: str = 'cpu'):
    """create sparse tensor fron channel last dense tensor by to_sparse
    x must be NHWC tensor, channel last
    """
    x_sp = x.to_sparse(x.ndim - 1).to(device)
    spatial_shape = x_sp.shape[1:-1]
    batch_size = x_sp.shape[0]
    indices_th = x_sp.indices().permute(1, 0).contiguous().int()
    features_th = x_sp.values()
    return SparseConvTensor(features_th, indices_th, spatial_shape, batch_size)


class SparseCoco(data.Dataset):

    def __init__(self, root: str, annFile: str,
                 transforms: Optional[Callable] = None):
        self.root = root
        with open(annFile, 'r') as f:
            self.coco = json.load(f)
        image_index = {image['image_id']: i
                       for i, image in enumerate(self.coco['images'])}
        cat_index = {
            cat['id']: i for i, cat in enumerate(self.coco['categories'])}
        self.index: dict[int, dict[int, list[int]]] = defaultdict(
            lambda: defaultdict(list))
        for i, annot in enumerate(self.coco['annotations']):
            image_i = image_index[annot['image_id']]
            cat_i = cat_index[annot['category_id']]
            self.index[image_i][cat_i].append(i)

    def __len__(self):
        return len(self.index)

    def __getitem(self, index):
        info = self.coco['images'][index]
        width, height = info['width'], info['height']
        path = info['file_name']
        image = read_image(os.path.join(self.root, path), ImageReadMode.GRAY)
        image = channel_last(
            ToDtype(torch.float32, scale=True)(
                invert(image)))
        image_sparse = image.to_sparse(image.ndim - 1)

        masks = []
        for cat_i, annots in self.index[index].items():
            cat_mask = None
            for i in annots:
                annot = self.coco['annotations'][i]
                mask = convert_coco_poly_to_mask([annot['segmentation']],
                                                 height, width)
                if cat_mask is None:
                    cat_mask = mask
                else:
                    cat_mask |= mask
            cat_mask = channel_last(cat_mask)
            masks.append(cat_mask.to_sparse(cat_mask.ndim - 1))
        target_sparse = torch.cat(masks, -1)

        if self.transforms is not None:
            image_sparse, target_sparse = self.transforms(
                image_sparse, target_sparse)

        return image_sparse, target_sparse
