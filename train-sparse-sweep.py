from dataclasses import dataclass

import lightning as L
import torch
import torchvision
import wandb
from cizm import Compression, SparseWeightUnstructured
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


@dataclass
class Args:
    train_dir: str = "/datasets/imagenet224/train"
    val_dir: str = "/datasets/imagenet224/val"
    epochs: int = 10
    sweep_id: str | None = None
    sp: float = 0.9
    count: int = 100
    batch_size: int = 128
    num_workers: int = 12
    gpus: int = -1


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
    def __init__(self, epochs: int, lr_factor: float, sweep_cfg: dict) -> None:
        super().__init__()
        self.model = torchvision.models.resnet50(weights="DEFAULT")
        self.compression = Compression(self.model)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.epochs = epochs
        self.lr_factor = lr_factor
        self.sweep_cfg = sweep_cfg

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
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        return optimizer


class SweepPruneAndLRCallback(Callback):
    def __init__(self, sweep_cfg: dict, lr_factor: float) -> None:
        super().__init__()
        self.sweep_cfg = sweep_cfg
        self.lr_factor = lr_factor

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

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: SparseResNet50Module) -> None:
        epoch = trainer.current_epoch
        lr = float(self.sweep_cfg[f"lr{epoch}"]) * self.lr_factor
        sp = float(self.sweep_cfg[f"sp{epoch}"])

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


def build_sweep_config(epochs: int, sp: float) -> dict:
    parameters = {}
    for epoch_idx in range(epochs):
        parameters[f"lr{epoch_idx}"] = {"distribution": "log_uniform_values", "min": 1e-6, "max": 1.0}
        parameters[f"sp{epoch_idx}"] = {"distribution": "q_uniform", "min": 0.0, "max": sp, "q": 0.01}
    parameters[f"sp{epochs - 1}"] = {"values": [sp]}
    return {"method": "bayes", "metric": {"name": "last/top1", "goal": "maximize"}, "parameters": parameters}


def run_one(args: Args) -> None:
    with wandb.init(project="imagenet-sweep") as run:
        cfg = dict(run.config)
        lr_factor = args.batch_size / 128

        datamodule = ImageNetDataModule(
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        module = SparseResNet50Module(epochs=args.epochs, lr_factor=lr_factor, sweep_cfg=cfg)
        callback = SweepPruneAndLRCallback(sweep_cfg=cfg, lr_factor=lr_factor)

        trainer = L.Trainer(
            accelerator="gpu",
            devices=args.gpus,
            max_epochs=args.epochs,
            logger=WandbLogger(experiment=run),
            callbacks=[callback],
            precision="16-mixed",
            enable_checkpointing=False,
            log_every_n_steps=10,
        )
        trainer.fit(module, datamodule=datamodule)

        metrics = trainer.callback_metrics
        last_loss = float(metrics["val/loss"].detach().cpu())
        last_top1 = float(metrics["val/top1"].detach().cpu())
        wandb.log({"last/loss": last_loss, "last/top1": last_top1})


if __name__ == "__main__":
    args = Args()
    sweep_id = args.sweep_id or wandb.sweep(build_sweep_config(args.epochs, args.sp), project="imagenet-sweep")
    wandb.agent(sweep_id, lambda: run_one(args), count=args.count, project="imagenet-sweep")
