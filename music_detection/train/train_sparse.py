import argparse
from pathlib import Path
import random
from functools import partial
from multiprocessing import set_start_method
from collections import defaultdict

import yaml
import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy_with_logits
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from dvclive import Live
import torcheval.metrics

from music_detection.data.sparse_dataset import SparseCoco, sparse_collate, \
    write_xy
from music_detection.model.sparse_model import SparseFCN, loss_fn, share_indices
from music_detection.params import Params


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'],
                        type=str, help="PyTorch device")
    parser.add_argument('--num_workers', default=4, type=int,
                        help="Number of parallel workers for dataloaders")
    parser.add_argument('--resume', default=None, type=Path,
                        help="Resume from this checkpoint")
    args = parser.parse_args()
    return args


@torch.inference_mode()
def evaluate(epoch: int, model: SparseFCN, name: str,
             loader: torch.utils.data.DataLoader,
             device: torch.device, writer: SummaryWriter, live: Live) -> float:
    model.eval()
    total_loss = 0.
    cpt = 0
    metric_fn = {
        "Accuracy": torcheval.metrics.BinaryAccuracy,
        "Precision": torcheval.metrics.BinaryPrecision,
        "Recall": torcheval.metrics.BinaryRecall,
        "F1": torcheval.metrics.BinaryF1Score,
    }

    metrics = defaultdict(dict)
    for metric_name, fn in metric_fn.items():
        metrics[metric_name]['total'] = fn()
        for label in loader.dataset.cat.values():
            metrics[metric_name][label] = fn()
    for i, (images, targets) in enumerate(tqdm(loader, desc=name), start=1):
        outputs = model(images)
        outputs, targets = share_indices(outputs, targets)
        imgs = write_xy(outputs, targets)
        for img in imgs:
            writer.add_image(f"{name}_{cpt}", img, global_step=epoch)
            cpt += 1
        loss = binary_cross_entropy_with_logits(
            outputs.features, targets.features)
        total_loss += loss.item()
        out_features = torch.sigmoid(outputs.features)
        target_features = targets.features > 0.5
        for metric in metrics.values():
            metric['total'].update(out_features.flatten(),
                                   target_features.flatten())
            for j, label in loader.dataset.cat.items():
                metric[label].update(out_features[:, j],
                                     target_features[:, j])
    total_loss /= i
    writer.add_scalars(name, {"loss": total_loss}, global_step=epoch)
    live.log_metric(f"{name}/loss", total_loss)
    for metric_name, metric in metrics.items():
        for label, value in metric.items():
            writer.add_scalars(f"{name}_{metric_name}",
                               {label: value.compute()}, global_step=epoch)
            live.log_metric(f"{name}/{metric_name}", value.compute().item())
    return total_loss


def main(args):
    with open("params.yaml") as f:
        params = Params(**yaml.load(f, Loader=yaml.Loader))
    train_params = params.train_sparse

    output_dir = train_params.output_dir
    output_dir.mkdir(exist_ok=True)

    device = torch.device(args.device)

    torch.use_deterministic_algorithms(True)
    random.seed(train_params.seed)
    np.random.seed(train_params.seed)
    torch.manual_seed(train_params.seed)

    train_dataset = SparseCoco(
        train_params.dataset / 'train2017',
        train_params.dataset / 'annotations/music_keypoints_train2017.json')
    val_dataset = SparseCoco(
        train_params.dataset / 'val2017',
        train_params.dataset / 'annotations/music_keypoints_val2017.json')

    sparse_collate_ = partial(sparse_collate, device=device)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_params.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=sparse_collate_,
        drop_last=False, persistent_workers=True)
    train_inference_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_params.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=sparse_collate_,
        drop_last=False, persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=train_params.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=sparse_collate_,
        drop_last=False, persistent_workers=True)

    model = SparseFCN(1, train_dataset.num_classes)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_params.lr)

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1

    writer = SummaryWriter(log_dir=train_params.output_dir,
                           purge_step=start_epoch)
    live = Live(dir=train_params.output_dir, resume=bool(args.resume),
                report='md')
    epoch_tqdm = tqdm(range(start_epoch, train_params.epochs), "epoch",
                      total=train_params.epochs, initial=start_epoch)
    for epoch in epoch_tqdm:
        model.train()
        for images, targets in tqdm(train_loader, 'train'):
            model.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
        }
        torch.save(checkpoint, train_params.output_dir / "checkpoint.pth")

        train_loss = evaluate(epoch, model, "train_infer",
                              train_inference_loader, device, writer, live)
        val_loss = evaluate(epoch, model, "val", val_loader, device, writer,
                            live)
        epoch_tqdm.set_postfix(train_loss=train_loss, val_loss=val_loss)
    writer.close()
    live.end()


if __name__ == "__main__":
    import autodebug
    set_start_method('spawn')
    main(make_args())
