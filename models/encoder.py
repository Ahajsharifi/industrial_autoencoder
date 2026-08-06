import torch.nn as nn

from models.blocks import ConvBlock


class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            ConvBlock(3, 32),
            nn.MaxPool2d(2),

            ConvBlock(32, 64),
            nn.MaxPool2d(2),

            ConvBlock(64, 128),
            nn.MaxPool2d(2),

            ConvBlock(128, 256),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.features(x)