import argparse
import warnings

warnings.filterwarnings(
    "ignore",
    message=r"`isinstance\(treespec, LeafSpec\)` is deprecated.*",
    category=DeprecationWarning,
)

import lightning as L
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


class ResNet50TrainModule(L.LightningModule):
    def __init__(self, lr: float, lr_min: float, epochs: int, pretrained: bool) -> None:
        super().__init__()
        self.model = torchvision.models.resnet50(
            weights="DEFAULT" if pretrained else None
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        self.lr = lr
        self.lr_min = lr_min
        self.epochs = epochs

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        pred = torch.argmax(logits, dim=1)
        acc = (pred == y).float().mean()
        lr = self.optimizers().param_groups[0]["lr"]
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_acc", acc, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("lr", lr, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.epochs,
            eta_min=self.lr_min,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class ImageNetTrainDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.train_set = ImageFolder(root=self.data_dir, transform=transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/datasets/imagenet224/train/")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lr_min", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()

    datamodule = ImageNetTrainDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    module = ResNet50TrainModule(
        lr=args.lr,
        lr_min=args.lr_min,
        epochs=args.epochs,
        pretrained=args.pretrained,
    )

    trainer = L.Trainer(
        accelerator="gpu",
        devices=args.gpus,
        precision="16-mixed",
        max_epochs=args.epochs,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    main()
