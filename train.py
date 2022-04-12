import argparse
import os
from typing import Union, Optional

import timm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision.datasets import ImageFolder


# solver settings
OPT = 'adam'  # adam, sgd
WEIGHT_DECAY = 0.0001
MOMENTUM = 0.9  # only when OPT is sgd
BASE_LR = 0.001
LR_SCHEDULER = 'step'  # step, multistep, reduce_on_plateau
LR_DECAY_RATE = 0.1
LR_STEP_SIZE = 5  # only when LR_SCHEDULER is step
LR_STEP_MILESTONES = [10, 15]  # only when LR_SCHEDULER is multistep


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train classifier.')
    parser.add_argument('--dataset', '-d', type=str, required=True, help='Root directory of dataset')
    parser.add_argument('--outdir', '-o', type=str, default='results', help='Output directory')
    parser.add_argument('--model-name', '-m', type=str, default='resnet18', help='Model name (timm)')
    parser.add_argument('--img-size', '-i', type=int, default=112, help='Input size of image')
    parser.add_argument('--epochs', '-e', type=str, default=100, help='Number of training epochs')
    parser.add_argument('--save-interval', '-s', type=int, default=10, help='Save interval (epoch)')
    parser.add_argument('--batch-size', '-b', type=int, default=8, help='Batch size')
    parser.add_argument('--num-workers', '-w', type=int, default=12, help='Number of workers')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--gpu-ids', type=int, default=None, nargs='+', help='GPU IDs to use')
    group.add_argument('--n-gpu', type=int, default=None, help='Number of GPUs')
    parser.add_argument('--seed', type=int, default=42, help='Seed')
    args = parser.parse_args()
    return args


def get_optimizer(parameters) -> torch.optim.Optimizer:
    if OPT == 'adam':
        optimizer = torch.optim.Adam(parameters,
                                     lr=BASE_LR,
                                     weight_decay=WEIGHT_DECAY)
    elif OPT == 'sgd':
        optimizer = torch.optim.SGD(parameters,
                                    lr=BASE_LR,
                                    weight_decay=WEIGHT_DECAY,
                                    momentum=MOMENTUM)
    else:
        raise NotImplementedError()

    return optimizer


def get_lr_scheduler_config(optimizer: torch.optim.Optimizer) -> dict:
    if LR_SCHEDULER == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=LR_STEP_SIZE,
            gamma=LR_DECAY_RATE)
        lr_scheduler_config = {
            'scheduler': scheduler,
            'interval': 'epoch',
            'frequency': 1,
        }
    elif LR_SCHEDULER == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=LR_STEP_MILESTONES,
            gamma=LR_DECAY_RATE)
        lr_scheduler_config = {
            'scheduler': scheduler,
            'interval': 'epoch',
            'frequency': 1,
        }
    elif LR_SCHEDULER == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.1,
            patience=10,
            threshold=0.0001)
        lr_scheduler_config = {
            'scheduler': scheduler,
            'monitor': 'val/loss',
            'interval': 'epoch',
            'frequency': 1,
        }
    else:
        raise NotImplementedError

    return lr_scheduler_config


class ImageTransform:
    def __init__(self, is_train: bool, img_size: Union[int, tuple] = 112):
        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self.transform(img)


class SimpleData(LightningDataModule):
    def __init__(self, root_dir: str, img_size: int = 112, batch_size: int = 8, num_workers: int = 16):
        super().__init__()
        self.root_dir = root_dir
        self.root_train = os.path.join(root_dir, 'train')
        self.root_val = os.path.join(root_dir, 'val')
        self.num_classes = len(os.listdir(self.root_train))
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self) -> DataLoader:
        train_dataset = ImageFolder(root=self.root_train,
                                    transform=ImageTransform(is_train=True, img_size=self.img_size))
        dataloader = DataLoader(train_dataset,
                                batch_size=self.batch_size,
                                shuffle=True,
                                drop_last=True,
                                num_workers=self.num_workers)
        return dataloader

    def val_dataloader(self) -> DataLoader:
        val_dataset = ImageFolder(root=self.root_val,
                                  transform=ImageTransform(is_train=False, img_size=self.img_size))
        dataloader = DataLoader(val_dataset,
                                batch_size=self.batch_size,
                                shuffle=False,
                                drop_last=False,
                                num_workers=self.num_workers)
        return dataloader


class SimpleModel(LightningModule):
    def __init__(self, model_name: str = 'resnet18',
                 pretrained: bool = True, num_classes: Optional[int] = None):
        super().__init__()
        self.save_hyperparameters()
        self.model = timm.create_model(model_name=model_name,
                                       pretrained=pretrained,
                                       num_classes=num_classes)
        self.train_loss = nn.CrossEntropyLoss()
        self.train_acc = Accuracy()
        self.val_loss = nn.CrossEntropyLoss()
        self.val_acc = Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, target = batch

        out = self(x)
        _, pred = out.max(1)

        loss = self.train_loss(out, target)
        acc = self.train_acc(pred, target)
        self.log_dict({'train/loss': loss, 'train/acc': acc}, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, target = batch

        out = self(x)
        _, pred = out.max(1)

        loss = self.val_loss(out, target)
        acc = self.val_acc(pred, target)
        self.log_dict({'val/loss': loss, 'val/acc': acc})

    def configure_optimizers(self):
        optimizer = get_optimizer(self.parameters())
        lr_scheduler_config = get_lr_scheduler_config(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}


def get_basic_callbacks(checkpoint_interval: int = 1) -> list:
    lr_callback = LearningRateMonitor(logging_interval='epoch')
    ckpt_callback = ModelCheckpoint(filename='epoch{epoch:03d}',
                                    auto_insert_metric_name=False,
                                    save_top_k=-1,
                                    every_n_epochs=checkpoint_interval)
    return [ckpt_callback, lr_callback]


def get_gpu_settings(args: argparse.Namespace) -> tuple:
    if args.gpu_ids is not None:
        # list
        gpus = args.gpu_ids
        strategy = "ddp" if len(gpus) > 1 else None
    elif args.n_gpu is not None:
        # int
        gpus = args.n_gpu
        strategy = "ddp" if gpus > 1 else None
    else:
        gpus = 1
        strategy = None
    gpus = gpus if torch.cuda.is_available() else None
    strategy = None if gpus is None else strategy
    return gpus, strategy


def get_trainer(args: argparse.Namespace) -> Trainer:
    callbacks = get_basic_callbacks(checkpoint_interval=args.save_interval)
    gpus, strategy = get_gpu_settings(args)
    trainer = Trainer(
        max_epochs=args.epochs,
        callbacks=callbacks,
        default_root_dir=args.outdir,
        gpus=gpus,
        strategy=strategy,
        logger=True,
    )
    return trainer


if __name__ == '__main__':
    args = get_args()
    seed_everything(args.seed)
    data = SimpleData(root_dir=args.dataset, img_size=args.img_size,
                      batch_size=args.batch_size, num_workers=args.num_workers)
    model = SimpleModel(model_name=args.model_name, num_classes=data.num_classes)
    trainer = get_trainer(args)
    trainer.fit(model, data)
