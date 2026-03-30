"""ResNet18-based Autoencoder for unsupervised anomaly detection."""

import torch
import torch.nn as nn
from torchvision import models


class ResNetEncoder(nn.Module):
    """Encoder: pretrained ResNet18, output 512-d feature vector."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)
        # Remove avgpool + fc, keep conv layers
        self.features = nn.Sequential(
            resnet.conv1,    # -> (64, 128, 128)
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,  # -> (64, 64, 64)
            resnet.layer1,   # -> (64, 64, 64)
            resnet.layer2,   # -> (128, 32, 32)
            resnet.layer3,   # -> (256, 16, 16)
            resnet.layer4,   # -> (512, 8, 8)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class Decoder(nn.Module):
    """Decoder: transpose convolutions to reconstruct 256x256 RGB image."""

    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            # (512, 8, 8) -> (256, 16, 16)
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # (256, 16, 16) -> (128, 32, 32)
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # (128, 32, 32) -> (64, 64, 64)
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # (64, 64, 64) -> (32, 128, 128)
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # (32, 128, 128) -> (3, 256, 256)
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),  # output range [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class AnomalyAutoencoder(nn.Module):
    """Full autoencoder: ResNet18 encoder + transpose conv decoder.

    Trained on normal images only. At inference, anomalous regions
    produce high reconstruction error.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.encoder = ResNetEncoder(pretrained=pretrained)
        self.decoder = Decoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon


if __name__ == "__main__":
    model = AnomalyAutoencoder(pretrained=False)
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    params = sum(p.numel() for p in model.parameters())
    print(f"Params: {params:,}")
