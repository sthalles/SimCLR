import argparse
import builtins
import math
import os
import shutil
import sys
import time
from functools import partial

import yaml

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as torchvision_models
from torch.utils.tensorboard import SummaryWriter

import simclr.builder
import simclr.loader
import simclr.optimizer
import utils
import vits


torchvision_model_names = sorted(
    name
    for name in torchvision_models.__dict__
    if name.islower()
    and not name.startswith("__")
    and callable(torchvision_models.__dict__[name])
)

model_names = [
    "vit_small",
    "vit_base",
    "vit_conv_small",
    "vit_conv_base",
] + torchvision_model_names

parser = argparse.ArgumentParser(description="SimCLR ImageNet Pre-Training")
parser.add_argument("data", metavar="DIR", help="path to dataset")
parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="resnet50",
    choices=model_names,
    help="model architecture: " + " | ".join(model_names) + " (default: resnet50)",
)
parser.add_argument(
    "-j",
    "--workers",
    default=8,
    type=int,
    metavar="N",
    help="number of data loading workers per node (default: 8)",
)
parser.add_argument(
    "--epochs", default=100, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b",
    "--batch-size-per-gpu",
    default=256,
    type=int,
    metavar="N",
    help="mini-batch size per node. Official SimCLR uses a global batch size of 4096 images.",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.3,
    type=float,
    metavar="LR",
    help="initial (base) learning rate",
    dest="lr",
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-6,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-6)",
    dest="weight_decay",
)
parser.add_argument(
    "-p",
    "--log-freq",
    default=100,
    type=int,
    metavar="N",
    help="Log frequency (default: 100)",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument("--seed", default=0, type=int, help="Random seed.")
parser.add_argument(
    "--dist_url",
    default="env://",
    type=str,
    help="""url used to set up
                    distributed training; see https://pytorch.org/docs/stable/distributed.html""",
)

parser.add_argument(
    "--feat-dim", default=128, type=int, help="feature dimension (default: 128)"
)
parser.add_argument(
    "--proj-mlp-dim",
    default=2048,
    type=int,
    help="hidden dimension in MLPs (default: 2048)",
)
parser.add_argument(
    "--softmax-t", default=0.1, type=float, help="softmax temperature (default: 0.1)"
)

# other upgrades
parser.add_argument(
    "--optimizer",
    default="lars",
    type=str,
    choices=["lars", "adamw"],
    help="optimizer used (default: lars)",
)
parser.add_argument(
    "--warmup-epochs", default=10, type=int, metavar="N", help="number of warmup epochs"
)
parser.add_argument(
    "--crop-min",
    default=0.08,
    type=float,
    help="minimum scale for random cropping (default: 0.08)",
)
parser.add_argument(
    "--save-checkpoint-every-epochs",
    default=50,
    type=int,
    help="minimum scale for random cropping (default: 0.08)",
)


def main():
    args = parser.parse_args()

    # setup distributed training
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
    )
    cudnn.benchmark = True

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch.startswith("vit"):
        model = simclr.builder.SimCLR_ViT(
            partial(vits.__dict__[args.arch]),
            args.feat_dim,
            args.proj_mlp_dim,
            args.softmax_t,
        )
    else:
        print("Creating model!")
        model = simclr.builder.SimCLR_ResNet(
            partial(torchvision_models.__dict__[args.arch], zero_init_residual=True),
            args.feat_dim,
            args.proj_mlp_dim,
            args.softmax_t,
        )

    # infer learning rate before changing batch size
    args.lr = args.lr * args.batch_size_per_gpu / 256

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.cuda()
    # DistributedDataParallel will divide and allocate batch_size_per_gpu to all
    # available GPUs if device_ids are not set
    model = torch.nn.parallel.DistributedDataParallel(model)

    if args.optimizer == "lars":
        optimizer = simclr.optimizer.LARS(
            model.parameters(),
            args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
        )
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), args.lr, weight_decay=args.weight_decay
        )

    scaler = torch.cuda.amp.GradScaler()
    summary_writer = None
    if utils.is_main_process():
        summary_writer = SummaryWriter()
        stats_file = open(
            os.path.join(summary_writer.log_dir, "stats.txt"), "a", buffering=1
        )
        print(" ".join(sys.argv))
        print(" ".join(sys.argv), file=stats_file)
        with open(os.path.join(summary_writer.log_dir, "metadata.txt"), "a") as f:
            yaml.dump(args, f, allow_unicode=True)
            f.write(str(model))

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scaler.load_state_dict(checkpoint["scaler"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading code
    traindir = os.path.join(args.data, "train")

    # SimCLR augmentaion protocol
    augmentations = utils.get_augmentations(strategy='pre-training')

    train_dataset = datasets.ImageFolder(
        traindir,
        simclr.loader.TwoCropsTransform(
            transforms.Compose(augmentations), transforms.Compose(augmentations)
        ),
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size_per_gpu,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )

    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, optimizer, scaler, summary_writer, epoch, args)

        if utils.is_main_process():  # only the first GPU saves checkpoint
            filename = "checkpoint.pth.tar"

            if (epoch + 1) % args.save_checkpoint_every_epochs == 0:
                filename = "checkpoint_%04d.pth.tar" % epoch

            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                },
                is_best=False,
                filename=os.path.join(summary_writer.log_dir, filename),
            )

    if utils.is_main_process():
        summary_writer.close()


def train(train_loader, model, optimizer, scaler, summary_writer, epoch, args):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    learning_rates = AverageMeter("LR", ":.4e")
    losses = AverageMeter("Loss", ":.4e")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, learning_rates, losses],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()

    end = time.time()
    iters_per_epoch = len(train_loader)

    for i, (images, _) in enumerate(train_loader):
        # global stop
        it = len(train_loader) * epoch + i

        # measure data loading time
        data_time.update(time.time() - end)

        # adjust learning rate and momentum coefficient per iteration
        lr = adjust_learning_rate(optimizer, epoch + i / iters_per_epoch, args)
        learning_rates.update(lr)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(True):
            loss = model(torch.cat(images))

        losses.update(loss.item(), images[0].size(0))
        if utils.is_main_process() and it % args.log_freq == 0:
            summary_writer.add_scalar("loss", loss.item(), it)
            progress.display(i)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def adjust_learning_rate(optimizer, epoch, args):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = (
            args.lr
            * 0.5
            * (
                1.0
                + math.cos(
                    math.pi
                    * (epoch - args.warmup_epochs)
                    / (args.epochs - args.warmup_epochs)
                )
            )
        )
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


if __name__ == "__main__":
    main()
