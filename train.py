###############################################################################
# https://github.com/pytorch/vision/blob/master/references/detection/train.py #
###############################################################################

r"""PyTorch Detection Training.

To run in a multi-gpu environment, use the distributed launcher::

    python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env \
        train.py ... --world-size $NGPU

The default hyperparameters are tuned for training on 8 gpus and 2 images per gpu.
    --lr 0.02 --batch-size 2 --world-size 8
If you use different number of gpus, the learning rate should be changed to 0.02/8*$NGPU.

On top of that, for training Faster/Mask R-CNN, the default hyperparameters are
    --epochs 26 --lr-steps 16 22 --aspect-ratio-group-factor 3

Also, if you train Keypoint R-CNN, the default hyperparameters are
    --epochs 46 --lr-steps 36 43 --aspect-ratio-group-factor 3
Because the number of images is smaller in the person keypoint subset of COCO,
the number of epochs should be adapted so that we have the same number of iterations.
"""
import datetime
import os
import time
from typing import List
import numpy as np

import torch
import torch.utils.data
from torch import nn
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn

from engine import train_one_epoch, evaluate
from dataset import CocoDataset, Resizer
import utils
import transforms as T
import torchvision.transforms as trsf
from tensorboardX import SummaryWriter


def get_dataset(image_dir, json_path, resize_dim):
    cocoset = CocoDataset(image_dir, json_path, transforms=trsf.Compose([Resizer(resize_dim)]))
    return cocoset, cocoset.num_classes


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def parse_resize(resize_str: str) -> List[int]:
    dims = resize_str.split(",")
    dims = list(map(int, dims))
    if len(dims) < 2:
        dims.append(dims[0])
    return dims


def main(args):
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # Data loading code
    print("Loading data")
    img_dim = parse_resize(args.resize)
    print(img_dim)
    dataset, num_classes = get_dataset(args.image_dir, args.train_json_path, img_dim)
    dataset_test, _ = get_dataset(args.image_dir, args.val_json_path, img_dim)

    writer = SummaryWriter(logdir=args.output_dir)
    best_map = np.NINF

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    train_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, args.batch_size, drop_last=True
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=train_batch_sampler,
        num_workers=args.workers,
        collate_fn=utils.collate_fn,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        sampler=test_sampler,
        num_workers=args.workers,
        collate_fn=utils.collate_fn,
    )

    print("Creating model")
    model = torchvision.models.detection.__dict__[args.model](
        num_classes=num_classes + 1, pretrained=args.pretrained
    )
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.lr_steps, gamma=args.lr_gamma
    )

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1

    if args.test_only:
        evaluate(model, data_loader_test, device=device)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq, writer)
        lr_scheduler.step()

        # evaluate after every epoch
        stats = evaluate(model, data_loader_test, device=device)
        map_avg, map_50, map_75, map_small, map_medium, map_large = stats[:6]
        if utils.is_main_process():
            writer.add_scalar("eval/map@0.5:0.95", map_avg, epoch)
            writer.add_scalar("eval/map@0.5", map_50, epoch)
            writer.add_scalar("eval/map@0.75", map_75, epoch)
            writer.add_scalar("eval/map_small", map_small, epoch)
            writer.add_scalar("eval/map_medium", map_medium, epoch)
            writer.add_scalar("eval/map_large", map_large, epoch)

        if args.output_dir:
            if map_50 > best_map:
                utils.save_on_master(
                    {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "args": args,
                        "epoch": epoch,
                    },
                    os.path.join(args.output_dir, "model_{}.pth".format(epoch)),
                )
                best_map = map_50

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--image-dir", type=str, help="path to the images")
    parser.add_argument("--train-json-path", help="Path to COCO directory")
    parser.add_argument("--val-json-path", help="Path to COCO directory")
    parser.add_argument("--model", default="maskrcnn_resnet50_fpn", help="model")
    parser.add_argument("--device", default="cuda", help="device")
    parser.add_argument("--resize", type=str, help="training image size", default="512")
    parser.add_argument(
        "-b",
        "--batch-size",
        default=2,
        type=int,
        help="images per gpu, the total batch size is $NGPU x batch_size",
    )
    parser.add_argument(
        "--epochs", default=26, type=int, metavar="N", help="number of total epochs to run"
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--lr",
        default=0.02,
        type=float,
        help="initial learning rate, 0.02 is the default value for training "
        "on 8 gpus and 2 images_per_gpu",
    )
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--lr-step-size", default=8, type=int, help="decrease lr every step-size epochs"
    )
    parser.add_argument(
        "--lr-steps",
        default=[16, 22],
        nargs="+",
        type=int,
        help="decrease lr every step-size epochs",
    )
    parser.add_argument(
        "--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma"
    )
    parser.add_argument("--print-freq", default=20, type=int, help="print frequency")
    parser.add_argument("--output-dir", default=".", help="path where to save")
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")
    parser.add_argument("--aspect-ratio-group-factor", default=3, type=int)
    parser.add_argument(
        "--test-only", dest="test_only", help="Only test the model", action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument(
        "--dist-url", default="env://", help="url used to set up distributed training"
    )

    args = parser.parse_args()

    if args.output_dir:
        utils.mkdir(args.output_dir)

    main(args)
