import os
from typing import Callable
import json
from pathlib import Path

import torch
import torch.utils.data as data
from torchvision.transforms.v2 import ToDtype
from torchvision.io import read_image, ImageReadMode, write_jpeg
import spconv.pytorch as spconv

from music_detection.train.coco_utils import convert_coco_poly_to_mask
from music_detection.model.sparse_model import share_indices


def invert(image: torch.Tensor):
    return 255 - image


def channel_last(image: torch.Tensor):
    return image.permute(1, 2, 0)


def from_torch_sparse(x_sp: torch.Tensor):
    """create SparseConvTensor tensor from channel last sparse tensor
    x_sp must be NHWC tensor, channel last
    """
    spatial_shape = x_sp.shape[1:-1]
    batch_size = x_sp.shape[0]
    indices_th = x_sp.indices().permute(1, 0).contiguous().int()
    features_th = x_sp.values()
    return spconv.SparseConvTensor(features_th, indices_th, spatial_shape,
                                   batch_size)


def to_torch_sparse(x: spconv.SparseConvTensor):
    return torch.sparse_coo_tensor(x.indices.T, x.features)


def sparse_collate(batch: list[tuple[torch.Tensor, torch.Tensor]],
                   device: torch.device):
    result = []
    for tensors in zip(*batch):
        tensor = torch.stack(tensors, dim=0)
        tensor = tensor.coalesce()
        tensor = tensor.to(device)
        tensor = from_torch_sparse(tensor)
        result.append(tensor)
    return tuple(result)


def write_mask(mask, last=True):
    if last:
        mask = mask.sum(axis=2).expand(1, -1, -1)
    mask = (mask * 255).to(torch.uint8)
    mask = torch.cat([mask, mask, mask], axis=0)
    write_jpeg(mask, "/tmp/test.jpeg")


def make_imgs(g, r, indices, spatial_shape, batch_size):
    g = spconv.SparseConvTensor(g, indices, spatial_shape, batch_size)
    g = g.dense().to('cpu')
    r = spconv.SparseConvTensor(r, indices, spatial_shape, batch_size)
    r = r.dense().to('cpu')
    b = torch.zeros_like(r)
    return (torch.cat([r, g, b], dim=1) * 255).to(torch.uint8)


@torch.no_grad()
def write_xy(x: spconv.SparseConvTensor, y: spconv.SparseConvTensor
             ) -> list[torch.Tensor]:
    g_feat = x.features * y.features
    r_feat = x.features * (1 - y.features)

    g_sum = g_feat.sum(dim=1, keepdim=True)
    r_sum = r_feat.sum(dim=1, keepdim=True)

    imgs = make_imgs(g_sum, r_sum, x.indices, x.spatial_shape, x.batch_size)

    res = []
    for i, img in enumerate(imgs):
        nonzero = torch.nonzero(img)
        x1 = nonzero[:, 1].min()
        y1 = nonzero[:, 2].min()
        x2 = nonzero[:, 1].max()
        y2 = nonzero[:, 2].max()
        img = img[:, x1:x2, y1:y2]
        # print(img.shape)
        # label_imgs = [img]
        # for j in range(g_feat.size(dim=1)):
        #     g_label = g_feat[:, j:j+1]
        #     r_label = r_feat[:, j:j+1]
        #     img_label = make_imgs(g_label, r_label, x.indices, x.spatial_shape,
        #                           x.batch_size)
        #     img_label = img_label[i, :, x1:x2, y1:y2]
        #     print(img_label.shape)
        #     label_imgs.append(img_label)
        # img = torch.cat(label_imgs, dim=1)
        res.append(img)
    return res


@torch.no_grad()
def write_x(x):
    x = to_torch_sparse(x)
    imgs = []
    for img in x:
        xs, ys = img.coalesce().indices()
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        img = img.to('cpu').to_dense()
        img = img[x1:x2, y1:y2]
        img = img.permute(2, 0, 1)
        img = torch.cat(
            [torch.cat([img[i*j] for j in range(7)], dim=0) for i in range(3)],
            dim=1)
        h, w = img.shape
        img = img.expand(3, h, w) * 255
        imgs.append(img)
    return imgs


class SparseCoco(data.Dataset):

    def __init__(self, root: Path, annFile: Path,
                 transforms: Callable | None = None,
                 categories: dict | None = None):
        super().__init__()
        self.root = root
        self.transforms = transforms
        with open(annFile, 'r') as f:
            self.coco = json.load(f)
        image_index = {image['id']: i
                       for i, image in enumerate(self.coco['images'])}
        self.cat = {
            i: cat['name'] for i, cat in enumerate(self.coco['categories'])}
        cat_index = {
            cat['id']: i for i, cat in enumerate(self.coco['categories'])}
        self.num_classes = len(self.coco['categories'])
        self.index: list[list[list[int]]] = [
            [[] for _ in range(self.num_classes)]
            for _ in range(len(image_index))]
        for i, annot in enumerate(self.coco['annotations']):
            image_i = image_index[annot['image_id']]
            cat_i = cat_index[annot['category_id']]
            self.index[image_i][cat_i].append(i)
        self.cache: dict = {}

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        if index in self.cache:
            return self.cache[index]
        info = self.coco['images'][index]
        width, height = info['width'], info['height']
        path = info['file_name']
        image = read_image(os.path.join(self.root, path), ImageReadMode.GRAY)
        image = invert(image)
        image = channel_last(image)
        image = ToDtype(torch.float32, scale=True)(image)
        image_sparse = image.to_sparse(image.ndim - 1)

        masks = []
        for annots in self.index[index]:
            cat_mask = torch.zeros(1, height, width, dtype=torch.uint8)
            for i in annots:
                annot = self.coco['annotations'][i]
                mask = convert_coco_poly_to_mask([annot['segmentation']],
                                                 height, width)
                cat_mask |= mask
            cat_mask = channel_last(cat_mask)
            cat_mask = cat_mask.to_sparse(cat_mask.ndim - 1)
            masks.append(cat_mask)
        target_sparse = torch.cat(masks, -1).to(torch.float32)

        if self.transforms is not None:
            image_sparse, target_sparse = self.transforms(
                image_sparse, target_sparse)

        self.cache[index] = (image_sparse, target_sparse)
        return image_sparse, target_sparse
