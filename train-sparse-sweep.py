import argparse

import lightning as L
import torch
import torchvision
from cizm import Compression, SparseWeightUnstructured
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


SCHEDULE_SLOTS = 10


class ImageNetDataModule(L.LightningDataModule):
    def __init__(self, train_dir: str, val_dir: str, batch_size: int, num_workers: int) -> None:
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str | None = None) -> None:
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        val_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.train_set = ImageFolder(root=self.train_dir, transform=train_transform)
        self.val_set = ImageFolder(root=self.val_dir, transform=val_transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )


class SparseResNet50Module(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.model = torchvision.models.resnet50(weights="DEFAULT")
        self.compression = Compression(self.model)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.compression(x)

    def training_step(self, batch, batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        top1 = (logits.argmax(dim=1) == y).float().mean()
        self.log("train/loss", loss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/top1", top1, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        top1 = (logits.argmax(dim=1) == y).float().mean()
        self.log("val/loss", loss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/top1", top1, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)


class SweepPruneAndLRCallback(Callback):
    def __init__(self, lr_factor: float, lrs: list[float], sps: list[float]) -> None:
        super().__init__()
        self.lr_factor = lr_factor
        self.lrs = lrs
        self.sps = sps

    @staticmethod
    def _target_filter(module: torch.nn.Module) -> bool:
        if not isinstance(module, torch.nn.Conv2d):
            return False
        if module.in_channels <= 3:
            return False
        if module.groups == module.in_channels:
            return False
        return True

    @staticmethod
    def _detach_all(pl_module: SparseResNet50Module) -> None:
        for sparsifier in pl_module.compression.registrations:
            sparsifier.detach(pl_module.model)
        pl_module.compression.registrations.clear()

    def _value_for_epoch(self, values: list[float], epoch: int) -> float:
        if epoch < len(values):
            return values[epoch]
        return values[-1]

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: SparseResNet50Module) -> None:
        epoch = trainer.current_epoch
        lr = self._value_for_epoch(self.lrs, epoch) * self.lr_factor
        sp = self._value_for_epoch(self.sps, epoch)

        optimizer = trainer.optimizers[0]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        self._detach_all(pl_module)
        pl_module.compression.attach(
            SparseWeightUnstructured,
            self._target_filter,
            sparsity=sp,
        )
        pl_module.log("lr", lr, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: SparseResNet50Module) -> None:
        self._detach_all(pl_module)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default="/datasets/imagenet224/train")
    parser.add_argument("--val_dir", type=str, default="/datasets/imagenet224/val")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--gpus", type=int, default=-1)
    parser.add_argument("--project", type=str, default="imagenet-sweep")
    parser.add_argument("--strategy", type=str, default="ddp_spawn")

    parser.add_argument("--lr_default", type=float, default=0.1)
    parser.add_argument("--sp_default", type=float, default=0.9)
    for i in range(SCHEDULE_SLOTS):
        parser.add_argument(f"--lr{i}", type=float, default=None)
        parser.add_argument(f"--sp{i}", type=float, default=None)
    return parser.parse_args()


def _build_schedule(args: argparse.Namespace, prefix: str, default: float) -> list[float]:
    values: list[float] = []
    for i in range(SCHEDULE_SLOTS):
        v = getattr(args, f"{prefix}{i}")
        values.append(default if v is None else float(v))
    return values


def main() -> None:
    args = _parse_args()
    lr_factor = args.batch_size / 128

    datamodule = ImageNetDataModule(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    module = SparseResNet50Module()
    callback = SweepPruneAndLRCallback(
        lr_factor=lr_factor,
        lrs=_build_schedule(args, "lr", args.lr_default),
        sps=_build_schedule(args, "sp", args.sp_default),
    )

    logger = WandbLogger(project=args.project)
    logger.log_hyperparams(vars(args))

    trainer = L.Trainer(
        accelerator="gpu",
        devices=args.gpus,
        strategy=args.strategy,
        max_epochs=args.epochs,
        logger=logger,
        callbacks=[callback],
        precision="16-mixed",
        enable_checkpointing=False,
        log_every_n_steps=10,
    )
    trainer.fit(module, datamodule=datamodule)

    metrics = trainer.callback_metrics
    last_loss = float(metrics["val/loss"].detach().cpu())
    last_top1 = float(metrics["val/top1"].detach().cpu())
    logger.experiment.log({"last/loss": last_loss, "last/top1": last_top1})


if __name__ == "__main__":
    main()
