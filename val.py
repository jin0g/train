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


class ResNet50ValModule(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.model = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        )
        self.model.eval()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        pred = torch.argmax(logits, dim=1)
        acc = (pred == y).float().mean()
        self.log("val_acc", acc, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)


class ImageNetValDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.val_set = ImageFolder(root=self.data_dir, transform=transform)

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/datasets/imagenet/val/")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()

    datamodule = ImageNetValDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    module = ResNet50ValModule()

    trainer = L.Trainer(
        accelerator="gpu",
        devices=args.gpus,
        precision="16-mixed",
        logger=False,
        enable_checkpointing=False,
    )
    trainer.validate(module, datamodule=datamodule)


if __name__ == "__main__":
    main()
