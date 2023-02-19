import pytorch_lightning as pl
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FakeData

from moonshine.models.unet import UNet


class SegmentationModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = UNet(name="unet50_fmow_rgb")
        self.backbone.load_weights(
            encoder_weights="unet50_fmow_rgb", decoder_weights="unet50_fmow_rgb"
        )
        self.classifier = nn.Conv2d(32, 2, (1, 1))

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x.mean((2, 3))


class LightningModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = SegmentationModel()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        return nn.CrossEntropyLoss(reduction="none")(y_hat, y).mean()

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-4)


def main():
    model = LightningModel()
    dataset = FakeData(
        size=128,
        image_size=(3, 256, 256),
        num_classes=2,
        transform=transforms.ToTensor(),
    )
    data_loader = DataLoader(dataset, batch_size=8)

    trainer = pl.Trainer(accelerator="auto", max_epochs=1)
    trainer.fit(model=model, train_dataloaders=data_loader)


if __name__ == "__main__":
    main()
